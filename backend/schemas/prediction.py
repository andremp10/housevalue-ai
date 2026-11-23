"""Contrato de entrada/saída de /api/predict (PRD §13.4).

Os campos correspondem exatamente a ml.features.APP_FEATURES (as colunas com que
o pipeline da aplicação foi treinado), em snake_case. Os tipos são validados
pelo Pydantic; as faixas e categorias aceitas são validadas no serviço contra
artifacts/models/form_schema.json, derivado dos dados de treino.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    overall_qual: StrictInt = Field(..., description="OverallQual — qualidade geral do acabamento (1–10)")
    overall_cond: StrictInt = Field(..., description="OverallCond — estado de conservação (1–10)")
    gr_liv_area_m2: StrictFloat = Field(..., gt=0, description="GrLivArea — área construída acima do solo, em m² (o modelo recebe o valor convertido para pés²)")
    total_bsmt_m2: StrictFloat = Field(..., ge=0, description="TotalBsmtSF — área do porão em m²; 0 = sem porão")
    lot_area_m2: StrictFloat = Field(..., gt=0, description="LotArea — área do terreno em m²")
    year_built: StrictInt = Field(..., description="YearBuilt — ano de construção")
    year_remod_add: StrictInt = Field(..., description="YearRemodAdd — ano da reforma (igual ao de construção se não houve)")
    full_bath: StrictInt = Field(..., description="FullBath — banheiros completos acima do solo")
    bedroom_abv_gr: StrictInt = Field(..., description="BedroomAbvGr — quartos acima do solo")
    garage_cars: StrictInt = Field(..., description="GarageCars — vagas na garagem; 0 = sem garagem")
    fireplaces: StrictInt = Field(..., description="Fireplaces — número de lareiras")
    neighborhood: StrictStr = Field(..., description="Neighborhood — código do bairro em Ames (ver /api/form-options)")
    kitchen_qual: StrictStr = Field(..., description="KitchenQual — qualidade da cozinha (Ex, Gd, TA, Fa)")
    central_air: StrictStr = Field(..., description="CentralAir — ar-condicionado central (Y/N)")


class InputEcho(BaseModel):
    field: str
    column: str
    label: str
    value: int | float | str
    display: str


class PriceRange(BaseModel):
    low: float
    high: float
    level: float
    holdout_coverage: float
    explanation: str


class PredictionResponse(BaseModel):
    predicted_sale_price: float
    range: PriceRange | None = None
    similar_real_houses: list[dict] = []
    currency: str = "USD"
    target: str = "SalePrice"
    model_name: str
    model_config_name: str
    model_version: str
    trained_at: str
    is_estimate: bool = True
    dataset_context: str
    inputs: list[InputEcho]
    notes: list[str]

    model_config = ConfigDict(protected_namespaces=())


class FieldError(BaseModel):
    field: str
    message: str


class ErrorResponse(BaseModel):
    detail: str
    errors: list[FieldError] | None = None


FieldName = Literal[
    "overall_qual", "overall_cond", "gr_liv_area_m2", "total_bsmt_m2", "lot_area_m2", "year_built", "year_remod_add",
    "full_bath", "bedroom_abv_gr", "garage_cars", "fireplaces", "neighborhood", "kitchen_qual", "central_air",
]


class SensitivityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs: PredictionRequest
    field: FieldName = Field(..., description="Variável (nome da API) a variar; as demais ficam fixas")


class DecomposeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base: PredictionRequest = Field(..., description="Cenário A (referência)")
    target: PredictionRequest = Field(..., description="Cenário atual")
