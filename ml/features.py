"""Conjuntos de atributos (PRD §7.3–7.5).

FULL: os 79 preditores do train.csv (exceto Id e SalePrice).
APP:  subconjunto do formulário web. Escolhido a partir da triagem feita
      somente na partição de desenvolvimento (artifacts/reports/eda_summary.json,
      chave app_candidate_screening) combinada com critérios de compreensão:

  Incluídos (força univariada na partição de desenvolvimento entre parênteses —
  ver eda_summary.json): OverallQual, Neighborhood, GrLivArea, GarageCars,
  KitchenQual, YearBuilt, FullBath, TotalBsmtSF, YearRemodAdd, Fireplaces,
  LotArea, CentralAir, BedroomAbvGr, OverallCond.

  Excluídos por redundância com um atributo mais intuitivo já incluído:
  GarageArea (≈ GarageCars), 1stFlrSF/2ndFlrSF/TotRmsAbvGrd (≈ GrLivArea),
  ExterQual (≈ KitchenQual/OverallQual). Excluído por difícil compreensão:
  BsmtQual (altura do porão). HouseStyle/BldgType/HalfBath: associação baixa.

  BedroomAbvGr e OverallCond têm associação univariada fraca, mas são
  características que um usuário espera informar; sua contribuição real é
  medida por importância por permutação no holdout (ml/explainability.py).
"""
from __future__ import annotations

APP_FEATURES: list[str] = [
    "OverallQual",
    "OverallCond",
    "GrLivArea",
    "TotalBsmtSF",
    "LotArea",
    "YearBuilt",
    "YearRemodAdd",
    "FullBath",
    "BedroomAbvGr",
    "GarageCars",
    "Fireplaces",
    "Neighborhood",
    "KitchenQual",
    "CentralAir",
]

# Metadados de interface (rótulos em PT-BR). Os limites e opções NÃO estão aqui:
# são derivados dos dados de treino em ml/train.py e salvos em form_schema.json.
# Áreas aparecem em m² (unit) mas o modelo recebe sq ft: ver ml/units.py.
FIELD_META: dict[str, dict] = {
    "OverallQual": {"api_name": "overall_qual", "label": "Qualidade geral do acabamento", "help": "Avaliação do material e acabamento da casa, de 1 (muito ruim) a 10 (excelente).", "unit": "nota 1–10", "step": 1, "group": "Qualidade"},
    "OverallCond": {"api_name": "overall_cond", "label": "Estado de conservação", "help": "Condição geral da casa, de 1 (muito ruim) a 10 (excelente).", "unit": "nota 1–10", "step": 1, "group": "Qualidade"},
    "KitchenQual": {"api_name": "kitchen_qual", "label": "Qualidade da cozinha", "help": "Qualidade da cozinha segundo o dicionário de dados.", "group": "Qualidade"},
    "GrLivArea": {"api_name": "gr_liv_area_m2", "label": "Área construída (sem porão)", "help": "Área habitável acima do nível do solo, sem contar o porão.", "unit": "m²", "step": 1, "group": "Áreas"},
    "TotalBsmtSF": {"api_name": "total_bsmt_m2", "label": "Área do porão", "help": "Área total do porão. Informe 0 se o imóvel não tem porão (comum nos EUA).", "unit": "m²", "step": 1, "group": "Áreas"},
    "LotArea": {"api_name": "lot_area_m2", "label": "Área do terreno", "help": "Tamanho total do lote.", "unit": "m²", "step": 1, "group": "Áreas"},
    "YearBuilt": {"api_name": "year_built", "label": "Ano de construção", "help": "Ano original de construção.", "unit": "ano", "step": 1, "group": "Idade"},
    "YearRemodAdd": {"api_name": "year_remod_add", "label": "Ano da última reforma", "help": "Ano de reforma/ampliação. Igual ao ano de construção se nunca foi reformado.", "unit": "ano", "step": 1, "group": "Idade"},
    "FullBath": {"api_name": "full_bath", "label": "Banheiros completos", "help": "Banheiros completos acima do nível do solo (não conta o porão).", "unit": "quantidade", "step": 1, "group": "Cômodos"},
    "BedroomAbvGr": {"api_name": "bedroom_abv_gr", "label": "Quartos", "help": "Quartos acima do nível do solo (não conta quartos no porão).", "unit": "quantidade", "step": 1, "group": "Cômodos"},
    "GarageCars": {"api_name": "garage_cars", "label": "Vagas na garagem", "help": "Capacidade da garagem em número de carros. Informe 0 se não há garagem.", "unit": "carros", "step": 1, "group": "Cômodos"},
    "Fireplaces": {"api_name": "fireplaces", "label": "Lareiras", "help": "Número de lareiras.", "unit": "quantidade", "step": 1, "group": "Cômodos"},
    "Neighborhood": {"api_name": "neighborhood", "label": "Bairro (Ames, Iowa)", "help": "Localização física dentro dos limites da cidade de Ames.", "group": "Localização"},
    "CentralAir": {"api_name": "central_air", "label": "Ar-condicionado central", "help": "Se o imóvel possui ar-condicionado central.", "group": "Qualidade"},
}

# Traduções das categorias de qualidade usadas no formulário (códigos do dicionário).
QUALITY_PT = {"Ex": "Excelente", "Gd": "Bom", "TA": "Típico/médio", "Fa": "Regular", "Po": "Ruim"}
YES_NO_PT = {"Y": "Sim", "N": "Não"}
