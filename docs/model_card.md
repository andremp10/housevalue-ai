# Model card — HouseValue AI

| Campo | Valor |
|---|---|
| Modelo | `HistGradientBoostingRegressor (nativo, monotônico) + log1p(y)` |
| Hiperparâmetros | learning_rate=0.03, max_leaf_nodes=15 |
| Versão | 2.0.0 |
| Treinado em | 2026-11-23T00:34:27+00:00 (UTC) |
| Artefato | `artifacts/models/house_price_pipeline.joblib` (SHA-256 `e75046fbc0f03cc16e1b5aa4a145d6a455a85d6082fbd3c036468bb708372eb5`) |
| Dados de treino | partição de desenvolvimento (1168 linhas), de `train.csv` (SHA-256 `1e18addf81e5e4d347cc17ee6075bbe4a42b7fa26b9e5b063e8f692a5f929d41`) |
| Alvo | `SalePrice`, USD nominal da época |
| Entradas | `OverallQual`, `OverallCond`, `GrLivArea`, `TotalBsmtSF`, `LotArea`, `YearBuilt`, `YearRemodAdd`, `FullBath`, `BedroomAbvGr`, `GarageCars`, `Fireplaces`, `Neighborhood`, `KitchenQual`, `CentralAir` |
| Bibliotecas | python 3.13.3, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, joblib 1.6.0 |

## Finalidade
Demonstração acadêmica (TK0231/UFC) de um fluxo completo de Machine Learning: estimar o preço histórico de venda de imóveis de Ames, Iowa, a partir de 14 características que uma pessoa consegue informar.

## População representada
1460 vendas residenciais em Ames, Iowa (EUA), entre 2006 e 2010 (Kaggle — House Prices: Advanced Regression Techniques). Faixa de preços observada: US$ 34.900 a US$ 755.000 (mediana US$ 163.000).

## Métricas (holdout de 292 imóveis, avaliação única)
| Métrica | Modelo | Baseline (mediana) |
|---|---|---|
| RMSLE (primária) | 0,1367 | 0,4323 |
| MAE | US$ 16.298 | US$ 59.568 |
| RMSE | US$ 26.766 | US$ 88.667 |
| R² | 0,907 | -0,025 |

Validação cruzada (5 folds, desenvolvimento): RMSLE 0,1325 ± 0,0068.

## Usos apropriados
- Ensino e demonstração de regressão, validação e implantação de modelos.
- Explorar como características físicas se associam a preços nesta base histórica.

## Usos não recomendados
- Avaliar imóveis reais, no Brasil ou em qualquer mercado atual; decisões de compra, venda, crédito ou tributação.
- Converter a estimativa para reais como se fosse um preço local.
- Imóveis com características fora das faixas do treino (a API rejeita esses valores).
- Inferências causais ("reformar aumenta o preço em X").

## Limitações e riscos conhecidos
- Subestima imóveis de alto padrão e erra mais em vendas `Partial`/`Abnorml` (ver `docs/model_evaluation.md`).
- A faixa típica de erro (80%) é a mesma para todas as casas (não varia por segmento); conferida no holdout, onde cobriu 79,1% dos preços reais.
- Categorias e faixas aceitas limitadas ao que existe na partição de desenvolvimento.
- Dados de 2006–2010 incluem o início da crise imobiliária americana de 2008.
