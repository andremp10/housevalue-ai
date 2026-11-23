"""Conversão de unidades de área (pés² → m²).

Os dados originais (Ames, Iowa) e o modelo usam pés quadrados (sq ft). Para a
apresentação no Brasil, a API, a interface e os documentos mostram metros
quadrados. A conversão acontece só na fronteira: o modelo recebe sempre sq ft,
exatamente como no treino.
"""
from __future__ import annotations

SQFT_TO_M2 = 0.09290304  # 1 pé² = 0,09290304 m² (definição exata)

# Colunas de área usadas no formulário/simulador → nome do campo em m² na API.
AREA_FIELDS = {
    "GrLivArea": "gr_liv_area_m2",
    "TotalBsmtSF": "total_bsmt_m2",
    "LotArea": "lot_area_m2",
}


def sqft_to_m2(v: float) -> float:
    return float(v) * SQFT_TO_M2


def m2_to_sqft(v: float) -> float:
    return float(v) / SQFT_TO_M2
