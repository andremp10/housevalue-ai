# Metodologia — HouseValue AI

> Gerado por `python -m ml.build_docs` a partir dos artefatos. Valores numéricos não são digitados à mão.

## 1. Problema
Regressão supervisionada: estimar `SalePrice` (preço de venda, USD, imóveis de Ames, Iowa, vendas de 2006 a 2010) a partir das características do imóvel.

## 2. Dados
- Fonte operacional: `house-prices-advanced-regression-techniques.zip` (SHA-256 `65f769a9157a2581671957ed08da8a8162d53e67b4e9970ee856b634deb11d9f`), cópia de trabalho em `data/raw/`. Detalhes em `docs/data_audit.md`.
- `train.csv`: 1460 imóveis rotulados, 81 colunas (79 preditores + `Id` + `SalePrice`).
- `test.csv`: 1459 imóveis **sem** `SalePrice` — usado só para gerar `submission.csv`.
- Leitura: `keep_default_na=False, na_values=["NA"]`, porque o padrão do pandas transformaria a categoria `"None"` de `MasVnrType` em ausente (872 → 8 ausentes reais).

## 3. Variável-alvo e exclusões
- Alvo: `SalePrice` (numérico, sem ausências, mínimo US$ 34.900, mediana US$ 163.000, máximo US$ 755.000; assimetria 1,88, em `log1p` 0,12).
- Excluídos dos preditores: `SalePrice` e `Id` (verificado por `assert_no_leakage_columns` e por testes).
- Nenhuma coluna derivada do preço foi identificada.

## 4. Conjuntos de atributos
- **full**: os 79 preditores (35 numéricos, 44 categóricos; `MSSubClass` tratado como categórico).
- **app** (formulário e API): 14 variáveis, escolhidas com triagem univariada **apenas na partição de desenvolvimento** (ρ de Spearman para numéricas e √η² para categóricas, contra `log1p(SalePrice)`) e critérios de compreensão/realismo de preenchimento. Justificativas em `ml/features.py`.

| Variável | Campo da API | Rótulo | Tipo | Faixa / opções (dev) | Força univariada (dev) |
|---|---|---|---|---|---|
| OverallQual | `overall_qual` | Qualidade geral do acabamento | inteiro | 1–10 | 0,801 |
| OverallCond | `overall_cond` | Estado de conservação | inteiro | 1–9 | 0,124 |
| GrLivArea | `gr_liv_area_m2` | Área construída (sem porão) | número (m²) | 31–524 m² | 0,723 |
| TotalBsmtSF | `total_bsmt_m2` | Área do porão | número (m²) | 0–568 m² | 0,595 |
| LotArea | `lot_area_m2` | Área do terreno | número (m²) | 121–19.997 m² | 0,464 |
| YearBuilt | `year_built` | Ano de construção | inteiro | 1872–2010 | 0,643 |
| YearRemodAdd | `year_remod_add` | Ano da última reforma | inteiro | 1950–2010 | 0,563 |
| FullBath | `full_bath` | Banheiros completos | inteiro | 0–3 | 0,626 |
| BedroomAbvGr | `bedroom_abv_gr` | Quartos | inteiro | 0–8 | 0,224 |
| GarageCars | `garage_cars` | Vagas na garagem | inteiro | 0–4 | 0,687 |
| Fireplaces | `fireplaces` | Lareiras | inteiro | 0–3 | 0,514 |
| Neighborhood | `neighborhood` | Bairro (Ames, Iowa) | categoria | 25 categorias | 0,744 |
| KitchenQual | `kitchen_qual` | Qualidade da cozinha | categoria | 4 categorias | 0,669 |
| CentralAir | `central_air` | Ar-condicionado central | categoria | 2 categorias | 0,365 |

As faixas aceitas são o mínimo e o máximo observados na partição de desenvolvimento: o modelo não foi treinado fora delas. Regra adicional: `YearRemodAdd ≥ YearBuilt` (0 violações no treino).

## 5. Pré-processamento (dentro do `Pipeline`)
1. `StructuralNAFiller` — regra fixa, sem parâmetros aprendidos: `NA` → `"None"` quando o dicionário define `NA` como inexistência **e** a coluna de existência confirma (ex.: `GarageType` com `GarageArea == 0`). `NA` com elemento existente permanece ausente. `Alley`, `Fence` e `MiscFeature` seguem o dicionário.
2. Numéricas: `SimpleImputer(median, add_indicator=True)`; `StandardScaler` apenas para o Ridge.
3. Categóricas: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")`.
4. Alvo (variantes testadas): `TransformedTargetRegressor(func=log1p, inverse_func=expm1)`; a previsão volta para USD.

Todos os parâmetros aprendidos (medianas, modas, médias/desvios, vocabulário do one-hot) são ajustados só nos dados de treino de cada fold e, no final, só na partição de desenvolvimento. Testes em `tests/test_preprocessing.py` verificam isso.

## 6. Divisão e validação
- Holdout: 292 imóveis (20%) separados com `train_test_split(shuffle=True, random_state=42)` antes de qualquer análise de modelagem; desenvolvimento: 1168. Ids em `artifacts/metrics/split.json`.
- Divisão aleatória justificada como estimativa de generalização para imóveis da mesma população (Ames, 2006–2010) — não é previsão do futuro. A mediana de preço por ano de venda é estável (ver `docs/data_audit.md`, seção 8), o que não indica deriva temporal forte nesse período.
- `KFold(n_splits=5, shuffle=True, random_state=42)` idêntico para todos os modelos e os dois conjuntos de atributos.
- Hiperparâmetros: `GridSearchCV` nesses folds, só no desenvolvimento. As métricas de CV do melhor ponto da grade são levemente otimistas; o holdout é a estimativa independente.

## 7. Algoritmos
| Algoritmo | Papel | Grade |
|---|---|---|
| `DummyRegressor(strategy="median")` | baseline obrigatório | — |
| `Ridge` | referência linear (regularização ajuda com a multicolinearidade das áreas e do one-hot) | alpha ∈ {0,1; 0,3; 1; 3; 10; 30; 100; 300; 1000} |
| `RandomForestRegressor(n_estimators=300)` | não linear, interações | max_features ∈ {0,33; 0,6; 1,0}, min_samples_leaf ∈ {1; 3} |
| `HistGradientBoostingRegressor(max_iter=500)` | boosting, candidato adicional | learning_rate ∈ {0,03; 0,1}, max_leaf_nodes ∈ {7; 15; 31} |

Cada algoritmo (exceto o baseline) foi testado com o alvo original e com `log1p(y)`.

## 8. Métricas
Calculadas na escala original (USD) para todos os modelos: MAE, RMSE, R² e RMSLE (`sqrt(mean((log1p(y) − log1p(ŷ))²))`; previsões negativas seriam truncadas em 0 e contadas). **Métrica primária de seleção: RMSLE** (alinhada à métrica da competição Kaggle), desempate por MAE — decisão registrada antes dos experimentos em `ml/config.py`.

## 9. Reprodutibilidade
Semente 42 em partição, folds e modelos; versões: Python 3.13.3, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, joblib 1.6.0. Hash SHA-256 do dado de origem e do artefato em `artifacts/models/model_metadata.json`.
