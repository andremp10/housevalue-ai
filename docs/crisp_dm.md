# CRISP-DM — HouseValue AI

> Gerado por `python -m ml.build_docs`. Cada etapa aponta para os artefatos reais que a evidenciam.

## 1. Entendimento do negócio
- **Contexto:** o preço de um imóvel depende de área, qualidade, idade, localização e outros fatores. A base é de Ames, Iowa (EUA), 2006–2010, em USD — não representa o mercado brasileiro nem preços atuais.
- **Problema:** estimar `SalePrice` com erro mensurável a partir das características do imóvel.
- **Pergunta:** em que medida características físicas, estruturais e de localização permitem estimar preços históricos de venda por regressão?
- **Hipóteses (H1–H4):** ver PRD §3.6; respostas na seção 5.
- **Usuários:** professor e estudantes; demonstrador do projeto.
- **Critérios de sucesso:** analítico — superar o baseline (mediana) em validação cruzada e no holdout, na métrica primária RMSLE e no MAE; funcional — API e interface entregando previsões do pipeline salvo, com testes passando.

## 2. Entendimento dos dados
- ZIP localizado em `user Downloads folder` (`house-prices-advanced-regression-techniques.zip`); os CSVs da raiz do workspace são idênticos aos membros do ZIP (SHA-256).
- 1460 imóveis rotulados × 81 colunas; 1459 sem rótulo. 0 duplicatas; `Id` único.
- Ausências: majoritariamente **estruturais** (dicionário: `NA` = "não possui"), p. ex. `PoolQC` 99.52%; desconhecidas de fato: `LotFrontage` 17.74%, `MasVnrType`/`MasVnrArea` 0,55%, `Electrical` 0,07%.
- Alvo assimétrico (assimetria 1,88; 0,12 em log1p).
- Divergências de grafia CSV × dicionário (ex.: `NAmes`/`Names`, `Twnhs`/`TwnhsI`) e inconsistências lógicas pontuais documentadas.
- **Evidências:** `docs/data_audit.md`, `artifacts/data_profile.json`, `artifacts/figures/eda_*.png`, `artifacts/reports/eda_summary.json`.

## 3. Preparação dos dados
- `Id` e `SalePrice` fora dos preditores; `MSSubClass` como categórico.
- `NA` estrutural → `"None"` por regra fixa; `NA` desconhecido → imputação dentro do pipeline (mediana/moda, com indicador de ausência nas numéricas).
- One-hot com `handle_unknown="ignore"`; padronização só para o Ridge.
- Holdout de 20% separado antes de qualquer ajuste; transformadores ajustados só no treino de cada fold.
- Dois conjuntos de atributos: 79 (referência) e 14 (aplicação), este escolhido só com dados de desenvolvimento.
- **Evidências:** `ml/preprocessing.py`, `ml/features.py`, `tests/test_preprocessing.py`, `docs/methodology.md`.

## 4. Modelagem
- Baseline (mediana), Ridge, Random Forest e HistGradientBoosting, com e sem `log1p(y)`, nos mesmos 5 folds e mesmos atributos; hiperparâmetros por GridSearchCV no desenvolvimento.
- Selecionado (aplicação): **HistGradientBoostingRegressor (nativo, monotônico) + log1p(y)** — RMSLE CV 0,1325 ± 0,0068.
- **Evidências:** `ml/train.py`, `artifacts/metrics/model_comparison.{json,csv}`.

## 5. Avaliação
- Holdout (292 imóveis): RMSLE 0,1367, MAE US$ 16.298, RMSE US$ 26.766, R² 0,907 — baseline: RMSLE 0,4323, MAE US$ 59.568.
- Supera o baseline em todas as métricas: sim.
- Hipóteses: H1, H2 e H4 suportadas; H3 suportada entre famílias de modelos (detalhes em `docs/model_evaluation.md`).
- Limitações: subestimação de imóveis caros, erros maiores em vendas `Partial`, holdout pequeno.
- **Evidências:** `ml/evaluate.py`, `artifacts/metrics/holdout_evaluation.json`, `artifacts/figures/eval_*.png`, `docs/model_evaluation.md`.

## 6. Implantação
- Pipeline completo (pré-processamento + modelo + transformação do alvo) salvo em `artifacts/models/house_price_pipeline.joblib`, com metadados e hash.
- API FastAPI (`backend/`) carrega o artefato (verificando o SHA-256), valida entradas com Pydantic e com as faixas/categorias de `form_schema.json`, e nunca treina durante requisições.
- Interface HTML/CSS/JS + Plotly.js (`frontend/`) servida pela própria API em `http://localhost:8000/`, com 3 abas: **Simular um imóvel** (estimativa ao vivo, faixa típica de erro e comparação Real × Modelo com casas parecidas que foram vendidas), **Imóveis reais** (o modelo roda nas casas de teste e cada estimativa é comparada ao preço da venda) e **Como o modelo funciona**. Áreas em m² na interface; a conversão para pés² (unidade do treino) acontece só na fronteira da API.
- Testes: `pytest` (dados, preparação, modelo, API) e `tests/e2e_ui.py` (navegador real).
- **Evidências:** `README.md`, `docs/model_card.md`, `docs/presentation.md`.
