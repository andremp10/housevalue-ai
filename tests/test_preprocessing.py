"""Testes de preparação (PRD §8.3, §18.2): sem vazamento, NA estrutural tratado por regra."""
import numpy as np
import pandas as pd
import pytest

from ml import config
from ml.features import APP_FEATURES
from ml.preprocessing import (
    NONE_CATEGORY,
    StructuralNAFiller,
    assert_no_leakage_columns,
    build_preprocessor,
    dev_holdout_split,
    full_feature_list,
    split_feature_types,
)


def test_target_and_id_never_features(train_df):
    feats = full_feature_list(train_df)
    assert config.TARGET not in feats and config.ID_COLUMN not in feats
    assert len(feats) == train_df.shape[1] - 2
    assert config.TARGET not in APP_FEATURES and config.ID_COLUMN not in APP_FEATURES
    with pytest.raises(ValueError):
        assert_no_leakage_columns(["GrLivArea", config.TARGET])


def test_structural_fill_respects_existence_indicator():
    X = pd.DataFrame({"GarageArea": [0, 400, 0], "GarageType": [np.nan, np.nan, "Attchd"], "Fence": [np.nan, "GdPrv", np.nan]})
    out = StructuralNAFiller().fit_transform(X)
    assert out.loc[0, "GarageType"] == NONE_CATEGORY      # sem garagem → estrutural
    assert pd.isna(out.loc[1, "GarageType"])                # garagem existe → desconhecido
    assert out.loc[2, "GarageType"] == "Attchd"
    assert out.loc[0, "Fence"] == NONE_CATEGORY             # sem indicador → dicionário


def test_structural_filler_is_stateless(train_df):
    a = StructuralNAFiller().fit(train_df.head(10)).transform(train_df.tail(50))
    b = StructuralNAFiller().fit(train_df.tail(500)).transform(train_df.tail(50))
    pd.testing.assert_frame_equal(a, b)


def test_mssubclass_is_categorical(train_df):
    num, cat = split_feature_types(full_feature_list(train_df), train_df)
    assert "MSSubClass" in cat and "MSSubClass" not in num
    assert "Neighborhood" in cat and "GrLivArea" in num


def test_split_is_disjoint_and_reproducible(train_df):
    d1, h1 = dev_holdout_split(train_df)
    d2, h2 = dev_holdout_split(train_df)
    assert set(d1[config.ID_COLUMN]).isdisjoint(h1[config.ID_COLUMN])
    assert len(d1) + len(h1) == len(train_df)
    assert list(h1[config.ID_COLUMN]) == list(h2[config.ID_COLUMN])
    assert abs(len(h1) / len(train_df) - config.HOLDOUT_FRACTION) < 0.01


def test_preprocessor_learns_only_from_fit_data(train_df):
    """Imputação/escala/vocabulário dependem apenas dos dados de ajuste (dev), nunca do holdout."""
    dev, hold = dev_holdout_split(train_df)
    feats = full_feature_list(train_df)
    num, cat = split_feature_types(feats, train_df)
    pre = build_preprocessor(num, cat, scale=True).fit(dev[feats])
    imputer = pre.named_steps["columns"].named_transformers_["num"].named_steps["impute"]
    i = num.index("LotFrontage")
    assert imputer.statistics_[i] == dev["LotFrontage"].median()
    scaler = pre.named_steps["columns"].named_transformers_["num"].named_steps["scale"]
    assert np.isclose(scaler.mean_[num.index("GrLivArea")], dev["GrLivArea"].mean())
    # categoria presente só no holdout não entra no vocabulário e é ignorada sem erro
    hold2 = hold[feats].copy()
    hold2.iloc[0, hold2.columns.get_loc("Neighborhood")] = "CATEGORIA_INEXISTENTE"
    Xt = pre.transform(hold2)
    assert Xt.shape[0] == len(hold2) and not np.isnan(Xt).any()
