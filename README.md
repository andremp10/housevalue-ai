# HouseValue AI

Estimativa **experimental** do preço de venda de imóveis de **Ames, Iowa (EUA)**, com vendas de **2006–2010** em **USD**, a partir da base Kaggle *House Prices: Advanced Regression Techniques*.
Projeto acadêmico (TK0231 — Sistemas de Informação Gerencial, UFC) com metodologia CRISP-DM: auditoria dos dados → pipeline scikit-learn → API FastAPI → interface HTML/CSS/JS com Plotly.js.

> Não é laudo de avaliação e não vale para o mercado brasileiro nem para preços atuais.

**Sumário:** [Resultado](#resultado-holdout-de-292-imóveis-avaliação-única) · [Arquitetura](#arquitetura) · [Instalação](#instalação) · [Pipeline de ML](#pipeline-de-ml-offline-nesta-ordem) · [Executar](#executar-a-aplicação) · [Endpoints](#endpoints) · [Testes](#testes) · [Interface](#interface) · [Estrutura](#estrutura) · [Documentação](#documentação) · [Apresentação](#para-a-apresentação) · [Limitações](#limitações-conhecidas)

## Resultado (holdout de 292 imóveis, avaliação única)

| Modelo | RMSLE | MAE | RMSE | R² |
|---|---|---|---|---|
| v2 · HistGradientBoosting nativo com restrições monotônicas + log1p(y), 14 variáveis (**usado na aplicação**) | 0,1367 | US$ 16.298 | US$ 26.766 | 0,907 |
| v1 · HistGradientBoosting (one-hot) + log1p(y), 14 variáveis | 0,1390 | US$ 16.668 | US$ 28.228 | 0,896 |
| Baseline — mediana | 0,4323 | US$ 59.568 | US$ 88.667 | −0,025 |

Os números vêm de `artifacts/metrics/holdout_evaluation.json` (v2) e `artifacts/history/v1/` (v1). Na validação cruzada, o modelo v2 ficou em **empate técnico** com a variante sem restrições monotônicas (venceu 16 de 20 partições, mas a vantagem média de 0,0008 é menor que a variação entre partições, de 0,0020). O holdout já tinha sido consultado na v1; a v2 escolheu o modelo só pela validação cruzada. A análise completa está em [`docs/model_evaluation.md`](docs/model_evaluation.md). Se você retreinar o modelo, esta tabela precisa ser atualizada à mão; os documentos em `docs/` são regenerados por `python -m ml.build_docs`.

## Arquitetura

```text
 ZIP Kaggle ──► ml.inspect_data ──► data/raw/ + docs/data_audit.md          (auditoria, sem modelagem)
                     │
                     ▼
 ml.exploratory_analysis ─► ml.train ─► ml.evaluate ─► ml.app_assets ─► ml.model_analysis ─► ml.build_docs
   (EDA, triagem só        (holdout 20%   (avaliação    (faixa de erro,  (curva de aprendizado, (docs/ com os
    no desenvolvimento)     separado, CV   única no      casas reais,     empate, erro por       números dos
                            5 partes,      holdout)      distribuições)   segmento)              artefatos)
                            9 configs)
                     │
                     ▼
 artifacts/models/house_price_pipeline.joblib  (+ metadados, SHA-256, form_schema, error_band)
                     │   carregado uma vez, hash conferido, nunca treina em requisição
                     ▼
 backend/ FastAPI  ──  /api/predict, /api/real-houses, /api/analytics/* …   (áreas em m² ↔ pés² só aqui)
                     │
                     ▼
 frontend/ HTML + CSS + JS + Plotly.js  ──  3 abas: Simular · Imóveis reais · Como funciona
```

O treino é **offline**: a API só carrega o pipeline salvo (pré-processamento + modelo + transformação do alvo em um único objeto), então a interface usa exatamente a mesma lógica avaliada.

## Requisitos
- Python 3.11+ (testado com 3.13.3 no Windows 11)
- A base: o ZIP `house-prices-advanced-regression-techniques.zip` no workspace ou em `~/Downloads`, **ou** os CSVs já extraídos na raiz do projeto (`train.csv`, `test.csv`, `data_description.txt`, `sample_submission.csv`). Outro caminho pode ser indicado com a variável de ambiente `HOUSEVALUE_ZIP`.

## Instalação

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-train.txt   # (só rodar a API: requirements.txt)
```

## Pipeline de ML (offline, nesta ordem)

```powershell
python -m ml.inspect_data          # Fase 1: localiza o ZIP, extrai data/raw/, gera docs/data_audit.md e artifacts/data_profile.json
python -m ml.exploratory_analysis  # Fase 2: figuras da EDA e triagem das variáveis do formulário (só partição de desenvolvimento)
python -m ml.train                 # Fase 3: CV de 9 configurações × 2 conjuntos de atributos, seleção e salvamento (~13 min)
python -m ml.evaluate              # Fase 3: avaliação única no holdout, diagnóstico, importâncias, submission.csv
python -m ml.app_assets            # Dados da interface: distribuições, imóveis reais do holdout e faixa típica de erro (sem retreino)
python -m ml.model_analysis        # Curva de aprendizado, teste de empate (RepeatedKFold) e erro por segmento
python -m ml.figures               # (opcional) redesenha as figuras eval_*.png a partir dos artefatos, sem retreinar
python -m ml.importance_plain      # (opcional) importância em linguagem simples para os slides (erro típico sem cada característica)
python -m ml.build_docs            # Gera docs/ com os números dos artefatos
```

Quase todo o tempo de treino (~10 dos ~13 min) vai para as duas variantes de HistGradientBoosting com one-hot nas 79 variáveis, mantidas para comparar com a v1. As variantes com categorias nativas levam ~40 s cada. Os artefatos já gerados estão em `artifacts/`, então não é preciso retreinar para usar a aplicação.

## Executar a aplicação

```powershell
uvicorn backend.main:app --reload
```

- Interface: http://localhost:8000/ (a API também serve o frontend; não é preciso outro servidor)
- Documentação da API: http://localhost:8000/docs

Se preferir servir o frontend separadamente (ex.: `python -m http.server 5500 -d frontend`), abra `http://localhost:5500/?api=http://localhost:8000`. As origens permitidas por CORS ficam em `HOUSEVALUE_CORS_ORIGINS` (ver `.env.example`).

O Plotly.js 2.35.2 vem incluído em `frontend/assets/vendor/`, então a demonstração funciona sem internet. Só as fontes do Google Fonts dependem da rede; sem elas, o navegador usa fontes alternativas.

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| GET | `/api/health` | Estado da API e do modelo |
| GET | `/api/model/info` | Algoritmo, versão, datas, variáveis, métricas de CV e holdout |
| GET | `/api/form-options` | Campos, rótulos, faixas e categorias aceitas (derivados do treino) |
| POST | `/api/predict` | Estimativa em USD para as 14 variáveis (áreas em m²), com faixa típica de erro e as 4 casas reais vendidas mais parecidas |
| GET | `/api/real-houses` | Roda o modelo agora nas 292 casas vendidas do holdout e devolve real × modelo de cada uma, com o resumo |
| GET | `/api/error-band` | Como a faixa típica de erro (80%) foi estimada e a cobertura conferida no holdout |
| POST | `/api/predict/sensitivity` | Curva de sensibilidade (ICE): previsões do modelo variando uma variável e mantendo as demais, só dentro da faixa de treino |
| GET | `/api/samples` | Imóveis reais do holdout (14 variáveis, preço real e previsão do modelo salvo) |
| POST | `/api/predict/compare` | Previsão de cada algoritmo candidato para o mesmo imóvel (com o MAE de holdout de cada um, informativo) |
| POST | `/api/predict/decompose` | Efeito isolado de cada característica alterada entre o cenário A e o atual, mais a interação |
| GET | `/api/analytics/model-analysis` | Curva de aprendizado, teste de empate e erro por faixa de preço, condição de venda e bairro |
| GET | `/api/analytics/overview` | Indicadores da base e do modelo |
| GET | `/api/analytics/model-comparison` | Comparação dos modelos (CV e holdout) |
| GET | `/api/analytics/feature-importance` | Importâncias, com o tipo de medida |
| GET | `/api/analytics/evaluation` | Real × previsto, resíduos e maiores erros (holdout) |
| GET | `/api/analytics/eda` | Agregações da análise exploratória |

Erros: `422` (campo ausente, tipo inválido, categoria desconhecida, valor fora da faixa do treino, reforma antes da construção), `413` (corpo grande demais), `503` (modelo ou artefatos ausentes, ou hash do modelo que não confere), `500` (falha genérica, sem stack trace).

Exemplo de requisição com um imóvel real do holdout (Id 16, preço real US$ 132.000):

```powershell
curl.exe -X POST http://localhost:8000/api/predict -H "Content-Type: application/json" -d '{\"overall_qual\":7,\"overall_cond\":8,\"gr_liv_area_m2\":79.34,\"total_bsmt_m2\":77.30,\"lot_area_m2\":568.57,\"year_built\":1929,\"year_remod_add\":2001,\"full_bath\":1,\"bedroom_abv_gr\":2,\"garage_cars\":2,\"fireplaces\":0,\"neighborhood\":\"BrkSide\",\"kitchen_qual\":\"TA\",\"central_air\":\"Y\"}'
```

Resposta (resumida): `"predicted_sale_price": 143567.49, "range": {"low": 125341.4, "high": 165103.36}`. As áreas vão em m²; a API converte para pés², a unidade com que o modelo foi treinado (1 pé² = 0,09290304 m²).

## Testes

```powershell
pytest                                   # dados, pré-processamento, modelo e API (46 testes)
pip install -r requirements-dev.txt      # opcional: Playwright, para o teste no navegador
uvicorn backend.main:app --port 8000     # em outro terminal
python tests/e2e_ui.py                   # navega na interface com o Microsoft Edge instalado e salva capturas
```

## Interface

Três abas. Em todo o sistema, **laranja = Real (preço pelo qual a casa foi vendida)** e **azul = Modelo (estimativa)**; verde, âmbar e vermelho indicam só a situação do erro: perto do real (até 10%), erro moderado (10% a 20%) e erro grande (mais de 20%). Áreas em m².

1. **Simular um imóvel** (abre aqui): formulário ao vivo (sliders com a distribuição dos dados de treino, nota de 1 a 10, botões de − e +, busca de bairros), estimativa com a **faixa típica de erro** e as casas vendidas mais parecidas em **Real | Modelo | Erro**. Logo abaixo, a tabela **Sua casa × casas reais vendidas**, com as características lado a lado e, para cada casa, preço real, estimativa e diferença. Abrir uma casa vendida guarda a casa que você montou ("Voltar para a casa que você montou"). Em "Explorar o modelo": como o preço reage a cada característica, algoritmos lado a lado e comparação de cenários.
2. **Imóveis reais**: o modelo calcula o preço das 292 casas vendidas que nunca viu. Mostra os indicadores, o gráfico real × modelo, a tabela de todas as casas e, fixo ao lado, o detalhe da casa selecionada com **Real | diferença | Modelo**.
3. **Como o modelo funciona**: explicação em 5 passos, o que mais pesa no preço, onde o modelo erra e o erro de cada algoritmo; os detalhes técnicos ficam num painel recolhível.

A API serializa as previsões e limita o OpenMP a 1 thread. Sem isso, chamadas simultâneas do HistGradientBoosting levavam ~8 s cada (medido); com isso, 8 chamadas simultâneas terminam em ~0,7 s.

## Estrutura

```text
ml/            auditoria, EDA, pré-processamento, treino, avaliação, explicabilidade, figuras (plot_style, figures), geração de docs
backend/       FastAPI: core/config, schemas, api (rotas), services (previsão e indicadores)
frontend/      index.html, assets/css, assets/js (api, util, charts, components, realhouses, simulator, app), assets/vendor (Plotly.js)
data/raw/      cópia de trabalho extraída do ZIP (não versionar)
artifacts/     models/ (joblib + metadados + form_schema + error_band), metrics/, figures/, reports/, history/v1/
docs/          data_audit, crisp_dm, methodology, model_evaluation, model_card, presentation
tests/         pytest e teste E2E da interface
```

## Documentação
- [Auditoria dos dados](docs/data_audit.md)
- [CRISP-DM](docs/crisp_dm.md)
- [Metodologia](docs/methodology.md)
- [Avaliação dos modelos](docs/model_evaluation.md)
- [Model card](docs/model_card.md)
- [Roteiro de apresentação](docs/presentation.md)

Índice comentado em [`docs/README.md`](docs/README.md). Todos os arquivos de `docs/` (menos o índice) são gerados a partir dos artefatos; não edite à mão.

## Para a apresentação

1. Subir o servidor: `.venv\Scripts\python -m uvicorn backend.main:app --port 8000` e abrir http://localhost:8000/.
2. Ler o guia completo [`docs/HouseValue_Guia_da_Apresentacao.pdf`](docs/HouseValue_Guia_da_Apresentacao.pdf) (projeto explicado, fala de cada slide e perguntas prováveis) e seguir o roteiro em [`docs/presentation.md`](docs/presentation.md): 15 slides em ordem de história (problema e hipóteses → 5 etapas do CRISP-DM → sistema → resposta), ≈9 min de fala + ≈2,5 min de demo ao vivo. Deck (claude.ai, privado até ser compartilhado): https://claude.ai/artifact/JPueWzcQrYvz5YYtjkfdvW, com a mesma numeração e a fala de cada slide nas notas do apresentador.
3. Figuras prontas para slides ou relatório: `artifacts/figures/` (mesma convenção de cores da interface).

## Limitações conhecidas
- Base histórica e local (Ames, 2006–2010, USD nominal).
- Erros maiores em imóveis caros e em vendas `Partial`/`Abnorml`; a faixa típica de erro (80%) é a mesma para todas as casas, não varia por segmento.
- O formulário aceita apenas valores dentro das faixas observadas no treino.
- `submission.csv` (Kaggle) não pôde ser avaliada localmente porque `test.csv` não tem rótulos.
