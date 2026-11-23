# Auditoria dos dados — House Prices (HouseValue AI)

> Documento gerado automaticamente por `python -m ml.inspect_data` em 2026-11-22T22:56:24+00:00 (UTC). Todos os números abaixo foram calculados a partir dos arquivos locais; nada foi presumido. Perfil completo em `artifacts/data_profile.json`.

## 1. Origem e localização dos arquivos

- **Locais pesquisados:** `workspace:C:\Users\ozile\OneDrive\Área de Trabalho\SIG_ML`, `downloads:C:\Users\ozile\Downloads`
- **ZIPs candidatos (contendo CSV com `SalePrice`):** 1
- **Decisão:** Nenhum ZIP no workspace; ZIP com a base House Prices encontrado em Downloads. O workspace contém os arquivos já extraídos (verificados por SHA-256).
- **ZIP original:** `house-prices-advanced-regression-techniques.zip` (user Downloads folder), 203,809 bytes, SHA-256 `65f769a9157a2581671957ed08da8a8162d53e67b4e9970ee856b634deb11d9f`. O arquivo original não foi movido nem modificado.

### Conteúdo do ZIP

| Membro | Tamanho (bytes) | Compactado | SHA-256 |
|---|---|---|---|
| data_description.txt | 13,370 | 4,446 | `356a4670aa688afa8ba44b1365b633fbbac2dae535fe4638532630607d64070d` |
| sample_submission.csv | 31,939 | 15,648 | `f19e99699b925aa3085c8d2ae43c62f15fbcd35e4a953b98025f8392c3ffa807` |
| test.csv | 451,405 | 87,058 | `8fdd3d829d4d986b58f845c9553b225e67dd8383624d90fb6ca1d4bed5798c1e` |
| train.csv | 460,676 | 96,055 | `1e18addf81e5e4d347cc17ee6075bbe4a42b7fa26b9e5b063e8f692a5f929d41` |

### Arquivos já extraídos na raiz do workspace

| Arquivo | Tamanho | Idêntico ao membro do ZIP? |
|---|---|---|
| train.csv | 460,676 | sim (SHA-256 igual) |
| test.csv | 451,405 | sim (SHA-256 igual) |
| data_description.txt | 13,370 | sim (SHA-256 igual) |
| sample_submission.csv | 31,939 | sim (SHA-256 igual) |

### Cópia de trabalho utilizada pelo pipeline

| Arquivo | Caminho | Origem | Ação |
|---|---|---|---|
| train.csv | data/raw/train.csv | zip | kept_existing_identical |
| test.csv | data/raw/test.csv | zip | kept_existing_identical |
| data_description.txt | data/raw/data_description.txt | zip | kept_existing_identical |
| sample_submission.csv | data/raw/sample_submission.csv | zip | kept_existing_identical |

## 2. Formato e estrutura

| Arquivo | Codificação | BOM | Fim de linha | Linhas | Colunas |
|---|---|---|---|---|---|
| train.csv | ascii | não | LF | 1,460 | 81 |
| test.csv | ascii | não | LF | 1,459 | 80 |
| sample_submission.csv | ascii | não | LF | 1,459 | 2 |
| data_description.txt | ascii | não | LF | — | 79 |

- `train.csv` contém `SalePrice`: **True**; `test.csv` contém `SalePrice`: **False** (portanto `test.csv` não serve para métricas supervisionadas).
- `sample_submission.csv` tem colunas ['Id', 'SalePrice'].
- Preditores (excluindo `Id` e `SalePrice`): **79** — 35 numéricos e 44 categóricos (incluindo `MSSubClass`, código numérico nominal).
- Dicionário (`data_description.txt`): 79 colunas descritas. Colunas do CSV sem descrição: nenhuma. Nomes divergentes mapeados: `Bedroom` → `BedroomAbvGr`, `Kitchen` → `KitchenAbvGr`.

## 3. Política de leitura e tokens de ausência

- Parâmetros: `{'keep_default_na': False, 'na_values': ['NA']}`. Somente o token literal 'NA' é ausente. O padrão do pandas converteria o texto 'None' (categoria válida de MasVnrType) em NaN.
- Evidência: com a leitura padrão do pandas, `MasVnrType` teria **872** ausentes; com a política do projeto, **8**. O restante é a categoria `None` (sem revestimento de alvenaria).
- Tokens brutos encontrados no `train.csv`: `'NA'` em 19 coluna(s); `'None'` em 1 coluna(s).

## 4. Dicionário de colunas (tipo inferido × esperado, ausências)

| Coluna | Descrição (dicionário) | dtype inferido | Tipo esperado | Únicos | Ausentes treino | % treino | Ausentes teste | Natureza da ausência |
|---|---|---|---|---|---|---|---|---|
| Id | — | int64 | identificador | 1,460 | 0 | 0.00 | 0 |  |
| MSSubClass | Identifies the type of dwelling involved in the sale. | int64 | categórica nominal (códigos numéricos) | 15 | 0 | 0.00 | 0 |  |
| MSZoning | Identifies the general zoning classification of the sale. | str | categórica | 5 | 0 | 0.00 | 4 | desconhecida / não registrada |
| LotFrontage | Linear feet of street connected to property | float64 | numérica | 110 | 259 | 17.74 | 227 | desconhecida / não registrada |
| LotArea | Lot size in square feet | int64 | numérica | 1,073 | 0 | 0.00 | 0 |  |
| Street | Type of road access to property | str | categórica | 2 | 0 | 0.00 | 0 |  |
| Alley | Type of alley access to property | str | categórica | 2 | 1,369 | 93.77 | 1,352 | estrutural — dicionário: NA = 'No alley access' |
| LotShape | General shape of property | str | categórica | 4 | 0 | 0.00 | 0 |  |
| LandContour | Flatness of the property | str | categórica | 4 | 0 | 0.00 | 0 |  |
| Utilities | Type of utilities available | str | categórica | 2 | 0 | 0.00 | 2 | desconhecida / não registrada |
| LotConfig | Lot configuration | str | categórica | 5 | 0 | 0.00 | 0 |  |
| LandSlope | Slope of property | str | categórica | 3 | 0 | 0.00 | 0 |  |
| Neighborhood | Physical locations within Ames city limits | str | categórica | 25 | 0 | 0.00 | 0 |  |
| Condition1 | Proximity to various conditions | str | categórica | 9 | 0 | 0.00 | 0 |  |
| Condition2 | Proximity to various conditions (if more than one is present) | str | categórica | 8 | 0 | 0.00 | 0 |  |
| BldgType | Type of dwelling | str | categórica | 5 | 0 | 0.00 | 0 |  |
| HouseStyle | Style of dwelling | str | categórica | 8 | 0 | 0.00 | 0 |  |
| OverallQual | Rates the overall material and finish of the house | int64 | numérica ordinal (escala 1–10) | 10 | 0 | 0.00 | 0 |  |
| OverallCond | Rates the overall condition of the house | int64 | numérica ordinal (escala 1–10) | 9 | 0 | 0.00 | 0 |  |
| YearBuilt | Original construction date | int64 | numérica | 112 | 0 | 0.00 | 0 |  |
| YearRemodAdd | Remodel date (same as construction date if no remodeling or additions) | int64 | numérica | 61 | 0 | 0.00 | 0 |  |
| RoofStyle | Type of roof | str | categórica | 6 | 0 | 0.00 | 0 |  |
| RoofMatl | Roof material | str | categórica | 8 | 0 | 0.00 | 0 |  |
| Exterior1st | Exterior covering on house | str | categórica | 15 | 0 | 0.00 | 1 | desconhecida / não registrada |
| Exterior2nd | Exterior covering on house (if more than one material) | str | categórica | 16 | 0 | 0.00 | 1 | desconhecida / não registrada |
| MasVnrType | Masonry veneer type | str | categórica | 4 | 8 | 0.55 | 16 | desconhecida / não registrada |
| MasVnrArea | Masonry veneer area in square feet | float64 | numérica | 327 | 8 | 0.55 | 15 | desconhecida / não registrada |
| ExterQual | Evaluates the quality of the material on the exterior | str | categórica | 4 | 0 | 0.00 | 0 |  |
| ExterCond | Evaluates the present condition of the material on the exterior | str | categórica | 5 | 0 | 0.00 | 0 |  |
| Foundation | Type of foundation | str | categórica | 6 | 0 | 0.00 | 0 |  |
| BsmtQual | Evaluates the height of the basement | str | categórica | 4 | 37 | 2.53 | 44 | estrutural — dicionário: NA = 'No Basement' |
| BsmtCond | Evaluates the general condition of the basement | str | categórica | 4 | 37 | 2.53 | 45 | estrutural — dicionário: NA = 'No Basement' |
| BsmtExposure | Refers to walkout or garden level walls | str | categórica | 4 | 38 | 2.60 | 44 | estrutural — dicionário: NA = 'No Basement' |
| BsmtFinType1 | Rating of basement finished area | str | categórica | 6 | 37 | 2.53 | 42 | estrutural — dicionário: NA = 'No Basement' |
| BsmtFinSF1 | Type 1 finished square feet | int64 | numérica | 637 | 0 | 0.00 | 1 | desconhecida / não registrada |
| BsmtFinType2 | Rating of basement finished area (if multiple types) | str | categórica | 6 | 38 | 2.60 | 42 | estrutural — dicionário: NA = 'No Basement' |
| BsmtFinSF2 | Type 2 finished square feet | int64 | numérica | 144 | 0 | 0.00 | 1 | desconhecida / não registrada |
| BsmtUnfSF | Unfinished square feet of basement area | int64 | numérica | 780 | 0 | 0.00 | 1 | desconhecida / não registrada |
| TotalBsmtSF | Total square feet of basement area | int64 | numérica | 721 | 0 | 0.00 | 1 | desconhecida / não registrada |
| Heating | Type of heating | str | categórica | 6 | 0 | 0.00 | 0 |  |
| HeatingQC | Heating quality and condition | str | categórica | 5 | 0 | 0.00 | 0 |  |
| CentralAir | Central air conditioning | str | categórica | 2 | 0 | 0.00 | 0 |  |
| Electrical | Electrical system | str | categórica | 5 | 1 | 0.07 | 0 | desconhecida / não registrada |
| 1stFlrSF | First Floor square feet | int64 | numérica | 753 | 0 | 0.00 | 0 |  |
| 2ndFlrSF | Second floor square feet | int64 | numérica | 417 | 0 | 0.00 | 0 |  |
| LowQualFinSF | Low quality finished square feet (all floors) | int64 | numérica | 24 | 0 | 0.00 | 0 |  |
| GrLivArea | Above grade (ground) living area square feet | int64 | numérica | 861 | 0 | 0.00 | 0 |  |
| BsmtFullBath | Basement full bathrooms | int64 | numérica | 4 | 0 | 0.00 | 2 | desconhecida / não registrada |
| BsmtHalfBath | Basement half bathrooms | int64 | numérica | 3 | 0 | 0.00 | 2 | desconhecida / não registrada |
| FullBath | Full bathrooms above grade | int64 | numérica | 4 | 0 | 0.00 | 0 |  |
| HalfBath | Half baths above grade | int64 | numérica | 3 | 0 | 0.00 | 0 |  |
| BedroomAbvGr | Bedrooms above grade (does NOT include basement bedrooms) | int64 | numérica | 8 | 0 | 0.00 | 0 |  |
| KitchenAbvGr | Kitchens above grade | int64 | numérica | 4 | 0 | 0.00 | 0 |  |
| KitchenQual | Kitchen quality | str | categórica | 4 | 0 | 0.00 | 1 | desconhecida / não registrada |
| TotRmsAbvGrd | Total rooms above grade (does not include bathrooms) | int64 | numérica | 12 | 0 | 0.00 | 0 |  |
| Functional | Home functionality (Assume typical unless deductions are warranted) | str | categórica | 7 | 0 | 0.00 | 2 | desconhecida / não registrada |
| Fireplaces | Number of fireplaces | int64 | numérica | 4 | 0 | 0.00 | 0 |  |
| FireplaceQu | Fireplace quality | str | categórica | 5 | 690 | 47.26 | 730 | estrutural — dicionário: NA = 'No Fireplace' |
| GarageType | Garage location | str | categórica | 6 | 81 | 5.55 | 76 | estrutural — dicionário: NA = 'No Garage' |
| GarageYrBlt | Year garage was built | float64 | numérica | 97 | 81 | 5.55 | 78 | desconhecida / não registrada |
| GarageFinish | Interior finish of the garage | str | categórica | 3 | 81 | 5.55 | 78 | estrutural — dicionário: NA = 'No Garage' |
| GarageCars | Size of garage in car capacity | int64 | numérica | 5 | 0 | 0.00 | 1 | desconhecida / não registrada |
| GarageArea | Size of garage in square feet | int64 | numérica | 441 | 0 | 0.00 | 1 | desconhecida / não registrada |
| GarageQual | Garage quality | str | categórica | 5 | 81 | 5.55 | 78 | estrutural — dicionário: NA = 'No Garage' |
| GarageCond | Garage condition | str | categórica | 5 | 81 | 5.55 | 78 | estrutural — dicionário: NA = 'No Garage' |
| PavedDrive | Paved driveway | str | categórica | 3 | 0 | 0.00 | 0 |  |
| WoodDeckSF | Wood deck area in square feet | int64 | numérica | 274 | 0 | 0.00 | 0 |  |
| OpenPorchSF | Open porch area in square feet | int64 | numérica | 202 | 0 | 0.00 | 0 |  |
| EnclosedPorch | Enclosed porch area in square feet | int64 | numérica | 120 | 0 | 0.00 | 0 |  |
| 3SsnPorch | Three season porch area in square feet | int64 | numérica | 20 | 0 | 0.00 | 0 |  |
| ScreenPorch | Screen porch area in square feet | int64 | numérica | 76 | 0 | 0.00 | 0 |  |
| PoolArea | Pool area in square feet | int64 | numérica | 8 | 0 | 0.00 | 0 |  |
| PoolQC | Pool quality | str | categórica | 3 | 1,453 | 99.52 | 1,456 | estrutural — dicionário: NA = 'No Pool' |
| Fence | Fence quality | str | categórica | 4 | 1,179 | 80.75 | 1,169 | estrutural — dicionário: NA = 'No Fence' |
| MiscFeature | Miscellaneous feature not covered in other categories | str | categórica | 4 | 1,406 | 96.30 | 1,408 | estrutural — dicionário: NA = 'None' |
| MiscVal | $Value of miscellaneous feature | int64 | numérica | 21 | 0 | 0.00 | 0 |  |
| MoSold | Month Sold (MM) | int64 | numérica | 12 | 0 | 0.00 | 0 |  |
| YrSold | Year Sold (YYYY) | int64 | numérica | 5 | 0 | 0.00 | 0 |  |
| SaleType | Type of sale | str | categórica | 9 | 0 | 0.00 | 1 | desconhecida / não registrada |
| SaleCondition | Condition of sale | str | categórica | 6 | 0 | 0.00 | 0 |  |
| SalePrice | — | int64 | alvo (numérica, USD) | 663 | 0 | 0.00 | — |  |

## 5. Interpretação das ausências (PRD §6.4)

O dicionário define explicitamente `NA` como **inexistência do elemento** nas colunas abaixo (ausência estrutural):

| Coluna | Significado de NA no dicionário |
|---|---|
| Alley | No alley access |
| BsmtQual | No Basement |
| BsmtCond | No Basement |
| BsmtExposure | No Basement |
| BsmtFinType1 | No Basement |
| BsmtFinType2 | No Basement |
| FireplaceQu | No Fireplace |
| GarageType | No Garage |
| GarageFinish | No Garage |
| GarageQual | No Garage |
| GarageCond | No Garage |
| PoolQC | No Pool |
| Fence | No Fence |
| MiscFeature | None |

Verificação de consistência (ausência estrutural deveria coincidir com o elemento inexistente):


**train.csv**

| Grupo | Coluna | NA total | NA e elemento ausente | NA mas elemento presente | Valor mas elemento ausente |
|---|---|---|---|---|---|
| Porão (TotalBsmtSF == 0) | BsmtQual | 37 | 37 | 0 | 0 |
| Porão (TotalBsmtSF == 0) | BsmtCond | 37 | 37 | 0 | 0 |
| Porão (TotalBsmtSF == 0) | BsmtExposure | 38 | 37 | 1 | 0 |
| Porão (TotalBsmtSF == 0) | BsmtFinType1 | 37 | 37 | 0 | 0 |
| Porão (TotalBsmtSF == 0) | BsmtFinType2 | 38 | 37 | 1 | 0 |
| Garagem (GarageArea == 0) | GarageType | 81 | 81 | 0 | 0 |
| Garagem (GarageArea == 0) | GarageFinish | 81 | 81 | 0 | 0 |
| Garagem (GarageArea == 0) | GarageQual | 81 | 81 | 0 | 0 |
| Garagem (GarageArea == 0) | GarageCond | 81 | 81 | 0 | 0 |
| Garagem (GarageArea == 0) | GarageYrBlt | 81 | 81 | 0 | 0 |
| Lareira (Fireplaces == 0) | FireplaceQu | 690 | 690 | 0 | 0 |
| Piscina (PoolArea == 0) | PoolQC | 1,453 | 1,453 | 0 | 0 |

**test.csv**

| Grupo | Coluna | NA total | NA e elemento ausente | NA mas elemento presente | Valor mas elemento ausente |
|---|---|---|---|---|---|
| Porão (TotalBsmtSF == 0) | BsmtQual | 44 | 42 | 2 | 0 |
| Porão (TotalBsmtSF == 0) | BsmtCond | 45 | 42 | 3 | 0 |
| Porão (TotalBsmtSF == 0) | BsmtExposure | 44 | 42 | 2 | 0 |
| Porão (TotalBsmtSF == 0) | BsmtFinType1 | 42 | 42 | 0 | 0 |
| Porão (TotalBsmtSF == 0) | BsmtFinType2 | 42 | 42 | 0 | 0 |
| Garagem (GarageArea == 0) | GarageType | 76 | 76 | 0 | 1 |
| Garagem (GarageArea == 0) | GarageFinish | 78 | 77 | 1 | 0 |
| Garagem (GarageArea == 0) | GarageQual | 78 | 77 | 1 | 0 |
| Garagem (GarageArea == 0) | GarageCond | 78 | 77 | 1 | 0 |
| Garagem (GarageArea == 0) | GarageYrBlt | 78 | 77 | 1 | 0 |
| Lareira (Fireplaces == 0) | FireplaceQu | 730 | 730 | 0 | 0 |
| Piscina (PoolArea == 0) | PoolQC | 1,456 | 1,453 | 3 | 0 |

Interpretação: `NA` com elemento ausente → ausência **estrutural** (preenchida com a categoria `"None"` por regra fixa, sem aprendizado). `NA` com elemento presente → informação **desconhecida/inconsistente** (mantida como ausente e imputada dentro do pipeline com parâmetros aprendidos só no treino). `Alley`, `Fence` e `MiscFeature` não têm coluna de existência; seguem o dicionário (`NA` = inexistente). `LotFrontage`, `MasVnrType`, `MasVnrArea` e `Electrical` não têm `NA` documentado: ausência tratada como desconhecida. `GarageYrBlt` é numérica e fica ausente quando não há garagem; imputada com indicador de ausência.

## 6. Variáveis com alto percentual de ausência (≥ 15% no treino)

| Coluna | % ausente (treino) | Natureza |
|---|---|---|
| LotFrontage | 17.74 | desconhecida / não registrada |
| Alley | 93.77 | estrutural — dicionário: NA = 'No alley access' |
| FireplaceQu | 47.26 | estrutural — dicionário: NA = 'No Fireplace' |
| PoolQC | 99.52 | estrutural — dicionário: NA = 'No Pool' |
| Fence | 80.75 | estrutural — dicionário: NA = 'No Fence' |
| MiscFeature | 96.30 | estrutural — dicionário: NA = 'None' |

Como as ausências altas são majoritariamente estruturais (informativas: indicam inexistência), as colunas são mantidas no conjunto completo com a categoria `None`; nenhuma é descartada só pelo percentual.

## 7. Duplicidades e identificadores

| Verificação | Resultado |
|---|---|
| train_full_rows | 0 |
| train_excluding_id | 0 |
| train_features_only | 0 |
| test_excluding_id | 0 |
| train_id_unique | True |
| test_id_unique | True |
| id_overlap_train_test | 0 |
| train_id_range | [1, 1460] |
| test_id_range | [1461, 2919] |

## 8. Variável-alvo `SalePrice` (USD)

| Estatística | Valor |
|---|---|
| count | 1,460.00 |
| mean | 180,921.20 |
| std | 79,442.50 |
| min | 34,900.00 |
| 25% | 129,975.00 |
| 50% | 163,000.00 |
| 75% | 214,000.00 |
| max | 755,000.00 |
| assimetria | 1.88 |
| curtose | 6.54 |
| assimetria log1p | 0.12 |
| curtose log1p | 0.81 |
| ausentes | 0 |
| ≤ 0 | 0 |

Quantis:

| Quantil | SalePrice (USD) |
|---|---|
| 0.01 | 61,815.97 |
| 0.05 | 88,000.00 |
| 0.1 | 106,475.00 |
| 0.25 | 129,975.00 |
| 0.5 | 163,000.00 |
| 0.75 | 214,000.00 |
| 0.9 | 278,000.00 |
| 0.95 | 326,100.00 |
| 0.99 | 442,567.01 |

Por ano de venda:

| YrSold | n | Mediana (USD) |
|---|---|---|
| 2006 | 314 | 163,995.00 |
| 2007 | 329 | 167,000.00 |
| 2008 | 304 | 164,000.00 |
| 2009 | 338 | 162,000.00 |
| 2010 | 175 | 155,000.00 |

A distribuição é assimétrica à direita; `log1p` reduz a assimetria (valores acima). Isso motiva testar `TransformedTargetRegressor(log1p/expm1)` na modelagem (PRD §9.4), sem aplicá-lo automaticamente.

## 9. Estatísticas descritivas — numéricas (treino)

| Coluna | n | média | dp | mín | 25% | 50% | 75% | máx | assim. | % zeros | outliers IQR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LotFrontage | 1,201.00 | 70.05 | 24.28 | 21.00 | 59.00 | 69.00 | 80.00 | 313.00 | 2.16 | 0.00 | 88 |
| LotArea | 1,460.00 | 10,516.83 | 9,981.26 | 1,300.00 | 7,553.50 | 9,478.50 | 11,601.50 | 215,245.00 | 12.21 | 0.00 | 69 |
| OverallQual | 1,460.00 | 6.10 | 1.38 | 1.00 | 5.00 | 6.00 | 7.00 | 10.00 | 0.22 | 0.00 | 2 |
| OverallCond | 1,460.00 | 5.58 | 1.11 | 1.00 | 5.00 | 5.00 | 6.00 | 9.00 | 0.69 | 0.00 | 125 |
| YearBuilt | 1,460.00 | 1,971.27 | 30.20 | 1,872.00 | 1,954.00 | 1,973.00 | 2,000.00 | 2,010.00 | -0.61 | 0.00 | 7 |
| YearRemodAdd | 1,460.00 | 1,984.87 | 20.65 | 1,950.00 | 1,967.00 | 1,994.00 | 2,004.00 | 2,010.00 | -0.50 | 0.00 | 0 |
| MasVnrArea | 1,452.00 | 103.69 | 181.07 | 0.00 | 0.00 | 0.00 | 166.00 | 1,600.00 | 2.67 | 58.97 | 96 |
| BsmtFinSF1 | 1,460.00 | 443.64 | 456.10 | 0.00 | 0.00 | 383.50 | 712.25 | 5,644.00 | 1.69 | 31.99 | 7 |
| BsmtFinSF2 | 1,460.00 | 46.55 | 161.32 | 0.00 | 0.00 | 0.00 | 0.00 | 1,474.00 | 4.26 | 88.56 | 167 |
| BsmtUnfSF | 1,460.00 | 567.24 | 441.87 | 0.00 | 223.00 | 477.50 | 808.00 | 2,336.00 | 0.92 | 8.08 | 29 |
| TotalBsmtSF | 1,460.00 | 1,057.43 | 438.71 | 0.00 | 795.75 | 991.50 | 1,298.25 | 6,110.00 | 1.52 | 2.53 | 61 |
| 1stFlrSF | 1,460.00 | 1,162.63 | 386.59 | 334.00 | 882.00 | 1,087.00 | 1,391.25 | 4,692.00 | 1.38 | 0.00 | 20 |
| 2ndFlrSF | 1,460.00 | 346.99 | 436.53 | 0.00 | 0.00 | 0.00 | 728.00 | 2,065.00 | 0.81 | 56.78 | 2 |
| LowQualFinSF | 1,460.00 | 5.84 | 48.62 | 0.00 | 0.00 | 0.00 | 0.00 | 572.00 | 9.01 | 98.22 | 26 |
| GrLivArea | 1,460.00 | 1,515.46 | 525.48 | 334.00 | 1,129.50 | 1,464.00 | 1,776.75 | 5,642.00 | 1.37 | 0.00 | 31 |
| BsmtFullBath | 1,460.00 | 0.43 | 0.52 | 0.00 | 0.00 | 0.00 | 1.00 | 3.00 | 0.60 | 58.63 | 1 |
| BsmtHalfBath | 1,460.00 | 0.06 | 0.24 | 0.00 | 0.00 | 0.00 | 0.00 | 2.00 | 4.10 | 94.38 | 82 |
| FullBath | 1,460.00 | 1.57 | 0.55 | 0.00 | 1.00 | 2.00 | 2.00 | 3.00 | 0.04 | 0.62 | 0 |
| HalfBath | 1,460.00 | 0.38 | 0.50 | 0.00 | 0.00 | 0.00 | 1.00 | 2.00 | 0.68 | 62.53 | 0 |
| BedroomAbvGr | 1,460.00 | 2.87 | 0.82 | 0.00 | 2.00 | 3.00 | 3.00 | 8.00 | 0.21 | 0.41 | 35 |
| KitchenAbvGr | 1,460.00 | 1.05 | 0.22 | 0.00 | 1.00 | 1.00 | 1.00 | 3.00 | 4.49 | 0.07 | 68 |
| TotRmsAbvGrd | 1,460.00 | 6.52 | 1.63 | 2.00 | 5.00 | 6.00 | 7.00 | 14.00 | 0.68 | 0.00 | 30 |
| Fireplaces | 1,460.00 | 0.61 | 0.64 | 0.00 | 0.00 | 1.00 | 1.00 | 3.00 | 0.65 | 47.26 | 5 |
| GarageYrBlt | 1,379.00 | 1,978.51 | 24.69 | 1,900.00 | 1,961.00 | 1,980.00 | 2,002.00 | 2,010.00 | -0.65 | 0.00 | 0 |
| GarageCars | 1,460.00 | 1.77 | 0.75 | 0.00 | 1.00 | 2.00 | 2.00 | 4.00 | -0.34 | 5.55 | 5 |
| GarageArea | 1,460.00 | 472.98 | 213.80 | 0.00 | 334.50 | 480.00 | 576.00 | 1,418.00 | 0.18 | 5.55 | 21 |
| WoodDeckSF | 1,460.00 | 94.24 | 125.34 | 0.00 | 0.00 | 0.00 | 168.00 | 857.00 | 1.54 | 52.12 | 32 |
| OpenPorchSF | 1,460.00 | 46.66 | 66.26 | 0.00 | 0.00 | 25.00 | 68.00 | 547.00 | 2.36 | 44.93 | 77 |
| EnclosedPorch | 1,460.00 | 21.95 | 61.12 | 0.00 | 0.00 | 0.00 | 0.00 | 552.00 | 3.09 | 85.75 | 208 |
| 3SsnPorch | 1,460.00 | 3.41 | 29.32 | 0.00 | 0.00 | 0.00 | 0.00 | 508.00 | 10.30 | 98.36 | 24 |
| ScreenPorch | 1,460.00 | 15.06 | 55.76 | 0.00 | 0.00 | 0.00 | 0.00 | 480.00 | 4.12 | 92.05 | 116 |
| PoolArea | 1,460.00 | 2.76 | 40.18 | 0.00 | 0.00 | 0.00 | 0.00 | 738.00 | 14.83 | 99.52 | 7 |
| MiscVal | 1,460.00 | 43.49 | 496.12 | 0.00 | 0.00 | 0.00 | 0.00 | 15,500.00 | 24.48 | 96.44 | 52 |
| MoSold | 1,460.00 | 6.32 | 2.70 | 1.00 | 5.00 | 6.00 | 8.00 | 12.00 | 0.21 | 0.00 | 0 |
| YrSold | 1,460.00 | 2,007.82 | 1.33 | 2,006.00 | 2,007.00 | 2,008.00 | 2,009.00 | 2,010.00 | 0.10 | 0.00 | 0 |

## 10. Frequência das categorias principais (treino)

| Coluna | nº categorias | Top categorias (contagem) | % dominante | Só no teste |
|---|---|---|---|---|
| MSSubClass | 15 | 20 (536), 60 (299), 50 (144), 120 (87) | 36.71 | 150 |
| MSZoning | 5 | RL (1151), RM (218), FV (65), RH (16) | 78.84 | — |
| Street | 2 | Pave (1454), Grvl (6) | 99.59 | — |
| Alley | 2 | None (1369), Grvl (50), Pave (41) | 93.77 | — |
| LotShape | 4 | Reg (925), IR1 (484), IR2 (41), IR3 (10) | 63.36 | — |
| LandContour | 4 | Lvl (1311), Bnk (63), HLS (50), Low (36) | 89.79 | — |
| Utilities | 2 | AllPub (1459), NoSeWa (1) | 99.93 | — |
| LotConfig | 5 | Inside (1052), Corner (263), CulDSac (94), FR2 (47) | 72.05 | — |
| LandSlope | 3 | Gtl (1382), Mod (65), Sev (13) | 94.66 | — |
| Neighborhood | 25 | NAmes (225), CollgCr (150), OldTown (113), Edwards (100) | 15.41 | — |
| Condition1 | 9 | Norm (1260), Feedr (81), Artery (48), RRAn (26) | 86.30 | — |
| Condition2 | 8 | Norm (1445), Feedr (6), Artery (2), RRNn (2) | 98.97 | — |
| BldgType | 5 | 1Fam (1220), TwnhsE (114), Duplex (52), Twnhs (43) | 83.56 | — |
| HouseStyle | 8 | 1Story (726), 2Story (445), 1.5Fin (154), SLvl (65) | 49.73 | — |
| RoofStyle | 6 | Gable (1141), Hip (286), Flat (13), Gambrel (11) | 78.15 | — |
| RoofMatl | 8 | CompShg (1434), Tar&Grv (11), WdShngl (6), WdShake (5) | 98.22 | — |
| Exterior1st | 15 | VinylSd (515), HdBoard (222), MetalSd (220), Wd Sdng (206) | 35.27 | — |
| Exterior2nd | 16 | VinylSd (504), MetalSd (214), HdBoard (207), Wd Sdng (197) | 34.52 | — |
| MasVnrType | 4 | None (864), BrkFace (445), Stone (128), BrkCmn (15) | 59.18 | — |
| ExterQual | 4 | TA (906), Gd (488), Ex (52), Fa (14) | 62.05 | — |
| ExterCond | 5 | TA (1282), Gd (146), Fa (28), Ex (3) | 87.81 | — |
| Foundation | 6 | PConc (647), CBlock (634), BrkTil (146), Slab (24) | 44.32 | — |
| BsmtQual | 4 | TA (649), Gd (618), Ex (121), None (37) | 44.45 | — |
| BsmtCond | 4 | TA (1311), Gd (65), Fa (45), None (37) | 89.79 | — |
| BsmtExposure | 4 | No (953), Av (221), Gd (134), Mn (114) | 65.27 | — |
| BsmtFinType1 | 6 | Unf (430), GLQ (418), ALQ (220), BLQ (148) | 29.45 | — |
| BsmtFinType2 | 6 | Unf (1256), Rec (54), LwQ (46), None (38) | 86.03 | — |
| Heating | 6 | GasA (1428), GasW (18), Grav (7), Wall (4) | 97.81 | — |
| HeatingQC | 5 | Ex (741), TA (428), Gd (241), Fa (49) | 50.75 | — |
| CentralAir | 2 | Y (1365), N (95) | 93.49 | — |
| Electrical | 5 | SBrkr (1334), FuseA (94), FuseF (27), FuseP (3) | 91.37 | — |
| KitchenQual | 4 | TA (735), Gd (586), Ex (100), Fa (39) | 50.34 | — |
| Functional | 7 | Typ (1360), Min2 (34), Min1 (31), Mod (15) | 93.15 | — |
| FireplaceQu | 5 | None (690), Gd (380), TA (313), Fa (33) | 47.26 | — |
| GarageType | 6 | Attchd (870), Detchd (387), BuiltIn (88), None (81) | 59.59 | — |
| GarageFinish | 3 | Unf (605), RFn (422), Fin (352), None (81) | 41.44 | — |
| GarageQual | 5 | TA (1311), None (81), Fa (48), Gd (14) | 89.79 | — |
| GarageCond | 5 | TA (1326), None (81), Fa (35), Gd (9) | 90.82 | — |
| PavedDrive | 3 | Y (1340), N (90), P (30) | 91.78 | — |
| PoolQC | 3 | None (1453), Gd (3), Ex (2), Fa (2) | 99.52 | — |
| Fence | 4 | None (1179), MnPrv (157), GdPrv (59), GdWo (54) | 80.75 | — |
| MiscFeature | 4 | None (1406), Shed (49), Gar2 (2), Othr (2) | 96.30 | — |
| SaleType | 9 | WD (1267), New (122), COD (43), ConLD (9) | 86.78 | — |
| SaleCondition | 6 | Normal (1198), Partial (125), Abnorml (101), Family (20) | 82.05 | — |

Categóricas quase constantes (≥ 99% numa categoria): Street, Utilities, PoolQC.

### Categorias observadas que não constam no dicionário


**train.csv**

| Coluna | Observado e não documentado | Documentado e não observado |
|---|---|---|
| MSZoning | C (all) | A, C, I, RP |
| Neighborhood | NAmes | Names |
| BldgType | 2fmCon, Duplex, Twnhs | 2FmCon, Duplx, TwnhsI |
| Exterior2nd | Brk Cmn, CmentBd, Wd Shng | BrkComm, CemntBd, PreCast, WdShing |

**test.csv**

| Coluna | Observado e não documentado | Documentado e não observado |
|---|---|---|
| MSZoning | C (all) | A, C, I, RP |
| Neighborhood | NAmes | Names |
| BldgType | 2fmCon, Duplex, Twnhs | 2FmCon, Duplx, TwnhsI |
| Exterior2nd | Brk Cmn, CmentBd, Wd Shng | BrkComm, CemntBd, Other, PreCast, WdShing |

Essas divergências são de grafia entre o CSV e o dicionário (ex.: abreviações diferentes). O modelo usa os valores **como aparecem no CSV**; o formulário só oferece categorias observadas no treino.

## 11. Valores extremos e inconsistências lógicas

Imóveis com `GrLivArea` > 4000 sq ft no treino:

| Id | GrLivArea | SalePrice | SaleCondition | OverallQual | Neighborhood |
|---|---|---|---|---|---|
| 524 | 4,676 | 184,750 | Partial | 10 | Edwards |
| 692 | 4,316 | 755,000 | Normal | 10 | NoRidge |
| 1183 | 4,476 | 745,000 | Abnorml | 10 | NoRidge |
| 1299 | 5,642 | 160,000 | Partial | 10 | Edwards |

No teste: 1 imóvel(is) com `GrLivArea` > 4000.

Verificações lógicas:

| Arquivo | Regra | Ocorrências | Ids (até 10) |
|---|---|---|---|
| train | YearRemodAdd < YearBuilt | 0 | — |
| train | YrSold < YearBuilt | 0 | — |
| train | YrSold < YearRemodAdd | 1 | 524 |
| train | GarageYrBlt > YrSold | 0 | — |
| train | GarageYrBlt < YearBuilt | 9 | 30, 94, 325, 601, 737, 1104, 1377, 1415, 1419 |
| train | MasVnrType == 'None' e MasVnrArea > 0 | 5 | 625, 774, 1231, 1301, 1335 |
| train | MasVnrType != 'None' e MasVnrArea == 0 | 2 | 689, 1242 |
| train | 1stFlrSF+2ndFlrSF+LowQualFinSF != GrLivArea | 0 | — |
| train | BsmtFinSF1+BsmtFinSF2+BsmtUnfSF != TotalBsmtSF | 0 | — |
| test | YearRemodAdd < YearBuilt | 1 | 1877 |
| test | YrSold < YearBuilt | 1 | 2550 |
| test | YrSold < YearRemodAdd | 2 | 2296, 2550 |
| test | GarageYrBlt > YrSold | 2 | 2550, 2593 |
| test | GarageYrBlt < YearBuilt | 9 | 1522, 1577, 1806, 1841, 1896, 1898, 2123, 2264, 2510 |
| test | MasVnrType == 'None' e MasVnrArea > 0 | 2 | 1670, 2453 |
| test | MasVnrType != 'None' e MasVnrArea == 0 | 1 | 2320 |
| test | 1stFlrSF+2ndFlrSF+LowQualFinSF != GrLivArea | 0 | — |
| test | BsmtFinSF1+BsmtFinSF2+BsmtUnfSF != TotalBsmtSF | 1 | 2121 |

## 12. Decisões de uso e exclusão

| Item | Decisão | Justificativa |
|---|---|---|
| `SalePrice` | Variável-alvo | Presente, numérica, sem ausências no `train.csv`. Nunca usada como entrada. |
| `Id` | Excluído dos preditores | Identificador sequencial sem significado físico (PRD §7.4). |
| `MSSubClass` | Tratada como categórica | Códigos numéricos nominais segundo o dicionário. |
| `OverallQual`, `OverallCond` | Numéricas ordinais | Escala 1–10 com ordem significativa. |
| Colunas com NA estrutural | Mantidas; NA → `"None"` quando o elemento inexiste | Ausência é informativa (PRD §6.4). |
| Ausências desconhecidas | Imputadas dentro do pipeline (mediana / moda), ajustadas só no treino | Evita vazamento (PRD §8.3). |
| Linhas extremas (`GrLivArea` > 4000) | **Mantidas** no protocolo principal | Removê-las do holdout tornaria a avaliação otimista; são vendas reais (ver `SaleCondition`). Registradas como limitação. |
| Quase constantes (Street, Utilities, PoolQC) | Mantidas no conjunto completo | Impacto desprezível; o one-hot e os modelos lidam com elas. Não entram no formulário. |
| Colunas com vazamento direto do alvo | Nenhuma identificada | Nenhuma coluna é derivada do preço. `MiscVal` é valor de um item acessório, não do imóvel. `SaleType`/`SaleCondition`/`MoSold`/`YrSold` são conhecidos no momento da venda; ficam no conjunto completo, mas não no formulário (não descrevem o imóvel). |

## 13. Limitações identificadas

- Base restrita a Ames, Iowa (EUA), vendas de 2006 a 2010, preços em USD nominais da época — não representa o mercado brasileiro nem preços atuais.
- Tamanho moderado (1,460 imóveis rotulados), com bairros e categorias raras pouco representados.
- `test.csv` não tem rótulos: só serve para gerar previsões (submissão Kaggle), nunca para métricas.
- Divergências de grafia entre CSV e dicionário (seção 10) e pequenas inconsistências entre colunas relacionadas (seções 5 e 11).
- Vendas atípicas (ex.: `SaleCondition = Partial/Abnorml`) e imóveis muito grandes com preço baixo podem gerar erros altos.

## 14. Decisão de viabilidade (PRD §6.5)

| Critério | Resultado |
|---|---|
| labelled_file_with_numeric_target | True |
| n_labelled_rows | 1460 |
| holdout_rows_at_20pct | 292 |
| predictors_interpretable_via_dictionary | True |
| critical_inconsistencies_treatable | True |
| apt_for_regression | True |

**Conclusão:** a base é apta para regressão supervisionada com `SalePrice` como alvo.
