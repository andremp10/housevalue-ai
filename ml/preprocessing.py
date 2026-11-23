"""Fase 2 — Preparação dos dados (PRD §8.3, §9.3).

Princípios:
- toda transformação com parâmetros aprendidos (imputação, escala, vocabulário do
  one-hot) fica dentro de um sklearn.Pipeline e é ajustada apenas nos dados de
  treino do fold/da partição de desenvolvimento;
- o preenchimento de ausências *estruturais* é uma regra fixa derivada do
  dicionário de dados (sem parâmetros aprendidos), logo não gera vazamento.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml import config

NONE_CATEGORY = "None"

# Coluna categórica com NA estrutural → coluna numérica que indica a existência do
# elemento (0 = inexistente). None = sem coluna indicadora: segue o dicionário.
STRUCTURAL_RULES: dict[str, str | None] = {
    "BsmtQual": "TotalBsmtSF",
    "BsmtCond": "TotalBsmtSF",
    "BsmtExposure": "TotalBsmtSF",
    "BsmtFinType1": "TotalBsmtSF",
    "BsmtFinType2": "TotalBsmtSF",
    "GarageType": "GarageArea",
    "GarageFinish": "GarageArea",
    "GarageQual": "GarageArea",
    "GarageCond": "GarageArea",
    "FireplaceQu": "Fireplaces",
    "PoolQC": "PoolArea",
    "Alley": None,
    "Fence": None,
    "MiscFeature": None,
}

NOMINAL_NUMERIC_CODES = ["MSSubClass"]
EXCLUDED_COLUMNS = [config.ID_COLUMN, config.TARGET]


class StructuralNAFiller(BaseEstimator, TransformerMixin):
    """Preenche ausências estruturais com a categoria "None" (regra fixa, sem fit).

    - Se a coluna indicadora estiver presente: preenche apenas quando o elemento
      não existe (indicador == 0). Ausências com elemento existente permanecem
      NaN e são imputadas depois, com parâmetros do treino.
    - Se a coluna indicadora não estiver presente (ex.: subconjunto do formulário),
      segue o dicionário: NA = inexistência.
    Também converte códigos numéricos nominais (MSSubClass) em texto e
    normaliza colunas textuais para dtype object com np.nan como ausente.
    """

    def __init__(self, rules: dict[str, str | None] | None = None):
        self.rules = rules

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        rules = STRUCTURAL_RULES if self.rules is None else self.rules
        X = X.copy()
        for c in NOMINAL_NUMERIC_CODES:
            if c in X.columns:
                X[c] = X[c].map(lambda v: np.nan if pd.isna(v) else str(int(v)))
        for c in X.columns:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
        for col, parent in rules.items():
            if col not in X.columns:
                continue
            na = X[col].isna()
            if parent is not None and parent in X.columns:
                mask = na & (pd.to_numeric(X[parent], errors="coerce") == 0)
            else:
                mask = na
            X.loc[mask, col] = NONE_CATEGORY
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


def split_feature_types(columns: list[str], reference: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separa numéricas/categóricas pelo tipo no CSV e pelo dicionário (MSSubClass)."""
    cat = [c for c in columns if c in NOMINAL_NUMERIC_CODES or not pd.api.types.is_numeric_dtype(reference[c])]
    num = [c for c in columns if c not in cat]
    return num, cat


def build_preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> Pipeline:
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(num_steps), numeric))
    if categorical:
        transformers.append(("cat", Pipeline(cat_steps), categorical))
    ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)
    return Pipeline([("structural_na", StructuralNAFiller()), ("columns", ct)])


def build_hgb_native_preprocessor(numeric: list[str], categorical: list[str]) -> Pipeline:
    """Pré-processamento para o HistGradientBoosting com categorias nativas (v2).

    - Ausências estruturais: mesma regra fixa (StructuralNAFiller).
    - Numéricas: passam sem alteração; o HGB trata NaN nativamente (aprende para
      que lado da divisão vão os valores ausentes, usando só os dados de treino).
    - Categóricas: OrdinalEncoder (vocabulário aprendido só no treino); categoria
      desconhecida ou ausente vira NaN, tratado como ausente pelo HGB.
    A saída é um DataFrame com os nomes originais das colunas, para que o HGB
    receba `categorical_features` e `monotonic_cst` pelo nome.
    """
    from sklearn.preprocessing import OrdinalEncoder

    transformers = []
    if numeric:
        transformers.append(("num", "passthrough", numeric))
    if categorical:
        transformers.append(
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan), categorical)
        )
    ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False).set_output(transform="pandas")
    return Pipeline([("structural_na", StructuralNAFiller()), ("columns", ct)])


# Restrições monotônicas (+1: com o resto fixo, aumentar a variável nunca reduz o preço
# previsto). Escolhidas por conhecimento de domínio, não pelos dados. Não incluem
# BedroomAbvGr (com a área fixa, mais quartos significa cômodos menores) nem os anos.
MONOTONIC_INCREASING = ["OverallQual", "OverallCond", "GrLivArea", "TotalBsmtSF", "LotArea", "GarageCars", "Fireplaces", "FullBath"]


def assert_no_leakage_columns(columns: list[str]) -> None:
    bad = [c for c in columns if c in EXCLUDED_COLUMNS]
    if bad:
        raise ValueError(f"Colunas proibidas como preditoras: {bad}")


def full_feature_list(train: pd.DataFrame) -> list[str]:
    return [c for c in train.columns if c not in EXCLUDED_COLUMNS]


def dev_holdout_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partição única e reprodutível 80/20 (PRD §9.5)."""
    dev, hold = train_test_split(train, test_size=config.HOLDOUT_FRACTION, random_state=config.RANDOM_STATE, shuffle=True)
    return dev.sort_values(config.ID_COLUMN), hold.sort_values(config.ID_COLUMN)
