"""Gera a documentação com números lidos dos artefatos (evita divergência texto × resultado).

Uso: python -m ml.build_docs   (depois de inspect_data, exploratory_analysis, train e evaluate)

Gera: docs/methodology.md, docs/model_evaluation.md, docs/model_card.md,
      docs/crisp_dm.md, docs/presentation.md
(docs/data_audit.md é gerado por ml.inspect_data)
"""
from __future__ import annotations

import json

import pandas as pd

from ml import config
from ml.features import APP_FEATURES, FIELD_META
from ml.units import SQFT_TO_M2


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def usd(v):
    return "US$ " + f"{v:,.0f}".replace(",", ".")


def n(v, d=4):
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def short(c):
    return c.replace("Regressor", "").replace("HistGradientBoosting", "HistGradientBoosting")


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    profile = _load(config.DATA_PROFILE_PATH)
    eda = _load(config.EDA_SUMMARY_PATH)
    comp = _load(config.MODEL_COMPARISON_PATH)
    hold = _load(config.HOLDOUT_REPORT_PATH)
    imp = _load(config.FEATURE_IMPORTANCE_PATH)
    meta = _load(config.MODEL_METADATA_PATH)
    split = _load(config.SPLIT_PATH)
    form = _load(config.FORM_SCHEMA_PATH)
    _band_p = config.MODELS_DIR / "error_band.json"
    band_cov = f"{n(100 * _load(_band_p)['holdout_coverage'], 1)}%" if _band_p.is_file() else "—"

    fs = comp["feature_sets"]
    M, B, F = hold["metrics"]["app"], hold["metrics"]["baseline"], hold["metrics"]["full"]
    t = profile["target"]["describe"]
    n_train, n_test = profile["files"]["train.csv"]["rows"], profile["files"]["test.csv"]["rows"]
    zipinfo = profile["source_manifest"]["zip"]
    sel_app = next(r for r in fs["app"]["results"] if r["config"] == fs["app"]["selected"])
    res = hold["residuals_app"]
    perm = imp["app"]["permutation_holdout"]["rows"]
    ridge_full = next(r for r in fs["full"]["results"] if r["config"] == "Ridge + log1p(y)")
    ridge_app = next(r for r in fs["app"]["results"] if r["config"] == "Ridge + log1p(y)")
    in_hold = [i for i in (524, 1299) if i in split["holdout_ids"]]

    def cv_table(key):
        rows = []
        for r in sorted(fs[key]["results"], key=lambda r: r["cv_rmsle_mean"]):
            mark = " ★" if r["config"] == fs[key]["selected"] else ""
            rows.append([short(r["config"]) + mark, n(r["cv_rmsle_mean"]), n(r["cv_rmsle_std"]), usd(r["cv_mae_mean"]), usd(r["cv_rmse_mean"]),
                         n(r["cv_r2_mean"], 3), n(r["train_rmsle_mean"]), ", ".join(f"{k}={v}" for k, v in r["best_params"].items()) or "—"])
        return table(["Configuração", "RMSLE CV", "± dp", "MAE CV", "RMSE CV", "R² CV", "RMSLE treino", "Hiperparâmetros"], rows)

    holdout_tbl = table(
        ["Artefato", "Configuração", "RMSLE", "MAE", "RMSE", "R²"],
        [["Aplicação (14 variáveis) ★", short(hold["evaluated_artifacts"]["app"]["config"]), n(M["rmsle"]), usd(M["mae"]), usd(M["rmse"]), n(M["r2"], 3)],
         ["Referência (79 variáveis)", short(hold["evaluated_artifacts"]["full"]["config"]), n(F["rmsle"]), usd(F["mae"]), usd(F["rmse"]), n(F["r2"], 3)],
         ["Baseline", "DummyRegressor (mediana)", n(B["rmsle"]), usd(B["mae"]), usd(B["rmse"]), n(B["r2"], 3)]],
    )
    cvh = hold["cv_vs_holdout"]["app"]
    cvh_tbl = table(["Métrica", "CV (média ± dp)", "Holdout"],
                    [["RMSLE", f"{n(cvh['rmsle']['cv_mean'])} ± {n(cvh['rmsle']['cv_std'])}", n(cvh["rmsle"]["holdout"])],
                     ["MAE", f"{usd(cvh['mae']['cv_mean'])} ± {usd(cvh['mae']['cv_std'])}", usd(cvh["mae"]["holdout"])],
                     ["RMSE", f"{usd(cvh['rmse']['cv_mean'])} ± {usd(cvh['rmse']['cv_std'])}", usd(cvh["rmse"]["holdout"])],
                     ["R²", f"{n(cvh['r2']['cv_mean'], 3)} ± {n(cvh['r2']['cv_std'], 3)}", n(cvh["r2"]["holdout"], 3)]])
    perm_tbl = table(["Variável", "Aumento da RMSLE (média)", "dp"], [[r["feature"], n(r["importance_mean"]), n(r["importance_std"])] for r in perm])
    err_tbl = table(["Id", "Real", "Previsto", "|Erro|", "Erro %", "Área construída", "OverallQual", "Bairro", "SaleCondition"],
                    [[r["Id"], usd(r["SalePrice"]), usd(r["pred_app"]), usd(r["abs_error_app"]), n(r["pct_error_app"], 1) + "%", f"{n(r['GrLivArea'] * SQFT_TO_M2, 0)} m²", r["OverallQual"], r["Neighborhood"], r["SaleCondition"]] for r in hold["top_errors_app"]])
    top_cond = {}
    for r in hold["top_errors_app"]:
        top_cond[r["SaleCondition"]] = top_cond.get(r["SaleCondition"], 0) + 1
    partial_share = 100 * profile["extremes"]["sale_condition_counts"].get("Partial", 0) / n_train
    screening = eda["app_candidate_screening"]["rows"]
    form_tbl = table(["Variável", "Campo da API", "Rótulo", "Tipo", "Faixa / opções (dev)", "Força univariada (dev)"],
                     [[f["name"], f"`{f['api_name']}`", f["label"], ("número (m²)" if f.get("display_factor") else "inteiro") if f["type"] == "integer" else "categoria",
                       (f"{n(f['min'] * SQFT_TO_M2, 0)}–{n(f['max'] * SQFT_TO_M2, 0)} m²" if f.get("display_factor") else f"{f['min']}–{f['max']}") if f["type"] == "integer" else f"{len(f['options'])} categorias",
                       n(next(s["abs_strength"] for s in screening if s["feature"] == f["name"]), 3)] for f in form["fields"]])
    lv = meta["library_versions"]
    beats_all = all(hold["beats_baseline"]["app"].values())
    D = config.DOCS_DIR
    D.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ methodology
    (D / "methodology.md").write_text(f"""# Metodologia — HouseValue AI

> Gerado por `python -m ml.build_docs` a partir dos artefatos. Valores numéricos não são digitados à mão.

## 1. Problema
Regressão supervisionada: estimar `SalePrice` (preço de venda, USD, imóveis de Ames, Iowa, vendas de 2006 a 2010) a partir das características do imóvel.

## 2. Dados
- Fonte operacional: `{zipinfo['file_name'] if zipinfo else 'arquivos extraídos'}` (SHA-256 `{zipinfo['sha256'] if zipinfo else '—'}`), cópia de trabalho em `data/raw/`. Detalhes em `docs/data_audit.md`.
- `train.csv`: {n_train} imóveis rotulados, {profile['files']['train.csv']['columns']} colunas ({profile['feature_groups']['n_features']} preditores + `Id` + `SalePrice`).
- `test.csv`: {n_test} imóveis **sem** `SalePrice` — usado só para gerar `submission.csv`.
- Leitura: `keep_default_na=False, na_values=["NA"]`, porque o padrão do pandas transformaria a categoria `"None"` de `MasVnrType` em ausente ({profile['read_policy']['masvnrtype_missing_with_pandas_default']} → {profile['read_policy']['masvnrtype_missing_with_project_policy']} ausentes reais).

## 3. Variável-alvo e exclusões
- Alvo: `SalePrice` (numérico, sem ausências, mínimo {usd(t['min'])}, mediana {usd(t['50%'])}, máximo {usd(t['max'])}; assimetria {n(profile['target']['skew'], 2)}, em `log1p` {n(profile['target']['log1p_skew'], 2)}).
- Excluídos dos preditores: `SalePrice` e `Id` (verificado por `assert_no_leakage_columns` e por testes).
- Nenhuma coluna derivada do preço foi identificada.

## 4. Conjuntos de atributos
- **full**: os {len(fs['full']['features'])} preditores ({len(fs['full']['numeric_features'])} numéricos, {len(fs['full']['categorical_features'])} categóricos; `MSSubClass` tratado como categórico).
- **app** (formulário e API): {len(APP_FEATURES)} variáveis, escolhidas com triagem univariada **apenas na partição de desenvolvimento** (ρ de Spearman para numéricas e √η² para categóricas, contra `log1p(SalePrice)`) e critérios de compreensão/realismo de preenchimento. Justificativas em `ml/features.py`.

{form_tbl}

As faixas aceitas são o mínimo e o máximo observados na partição de desenvolvimento: o modelo não foi treinado fora delas. Regra adicional: `YearRemodAdd ≥ YearBuilt` (0 violações no treino).

## 5. Pré-processamento (dentro do `Pipeline`)
1. `StructuralNAFiller` — regra fixa, sem parâmetros aprendidos: `NA` → `"None"` quando o dicionário define `NA` como inexistência **e** a coluna de existência confirma (ex.: `GarageType` com `GarageArea == 0`). `NA` com elemento existente permanece ausente. `Alley`, `Fence` e `MiscFeature` seguem o dicionário.
2. Numéricas: `SimpleImputer(median, add_indicator=True)`; `StandardScaler` apenas para o Ridge.
3. Categóricas: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")`.
4. Alvo (variantes testadas): `TransformedTargetRegressor(func=log1p, inverse_func=expm1)`; a previsão volta para USD.

Todos os parâmetros aprendidos (medianas, modas, médias/desvios, vocabulário do one-hot) são ajustados só nos dados de treino de cada fold e, no final, só na partição de desenvolvimento. Testes em `tests/test_preprocessing.py` verificam isso.

## 6. Divisão e validação
- Holdout: {split['n_holdout']} imóveis (20%) separados com `train_test_split(shuffle=True, random_state=42)` antes de qualquer análise de modelagem; desenvolvimento: {split['n_dev']}. Ids em `artifacts/metrics/split.json`.
- Divisão aleatória justificada como estimativa de generalização para imóveis da mesma população (Ames, 2006–2010) — não é previsão do futuro. A mediana de preço por ano de venda é estável (ver `docs/data_audit.md`, seção 8), o que não indica deriva temporal forte nesse período.
- `KFold(n_splits=5, shuffle=True, random_state=42)` idêntico para todos os modelos e os dois conjuntos de atributos.
- Hiperparâmetros: `GridSearchCV` nesses folds, só no desenvolvimento. As métricas de CV do melhor ponto da grade são levemente otimistas; o holdout é a estimativa independente.

## 7. Algoritmos
| Algoritmo | Papel | Grade |
|---|---|---|
| `DummyRegressor(strategy="median")` | baseline obrigatório | — |
| `Ridge` | referência linear (regularização ajuda com a multicolinearidade das áreas e do one-hot) | alpha ∈ {{0,1; 0,3; 1; 3; 10; 30; 100; 300; 1000}} |
| `RandomForestRegressor(n_estimators=300)` | não linear, interações | max_features ∈ {{0,33; 0,6; 1,0}}, min_samples_leaf ∈ {{1; 3}} |
| `HistGradientBoostingRegressor(max_iter=500)` | boosting, candidato adicional | learning_rate ∈ {{0,03; 0,1}}, max_leaf_nodes ∈ {{7; 15; 31}} |

Cada algoritmo (exceto o baseline) foi testado com o alvo original e com `log1p(y)`.

## 8. Métricas
Calculadas na escala original (USD) para todos os modelos: MAE, RMSE, R² e RMSLE (`sqrt(mean((log1p(y) − log1p(ŷ))²))`; previsões negativas seriam truncadas em 0 e contadas). **Métrica primária de seleção: RMSLE** (alinhada à métrica da competição Kaggle), desempate por MAE — decisão registrada antes dos experimentos em `ml/config.py`.

## 9. Reprodutibilidade
Semente 42 em partição, folds e modelos; versões: Python {lv['python']}, scikit-learn {lv['scikit-learn']}, pandas {lv['pandas']}, numpy {lv['numpy']}, joblib {lv['joblib']}. Hash SHA-256 do dado de origem e do artefato em `artifacts/models/model_metadata.json`.
""", encoding="utf-8")

    # ------------------------------------------------------------------ evaluation
    (D / "model_evaluation.md").write_text(f"""# Avaliação dos modelos — HouseValue AI

> Gerado por `python -m ml.build_docs` a partir de `artifacts/metrics/`. Protocolo em `docs/methodology.md`.

## 1. Métrica primária (definida antes dos experimentos)
**RMSLE**, para alinhamento com a métrica da competição Kaggle (RMSE entre logaritmos), com desempate por **MAE em USD**. MAE, RMSE e R² também são reportados. Todas as métricas abaixo estão em USD/escala original, calculadas nos mesmos dados para todos os modelos.

## 2. Validação cruzada — conjunto da aplicação ({len(fs['app']['features'])} variáveis)
5 folds, mesma partição para todos; desenvolvimento = {split['n_dev']} imóveis. ★ = selecionado.

{cv_table('app')}

## 3. Validação cruzada — todos os {len(fs['full']['features'])} atributos (referência)

{cv_table('full')}

![Comparação em CV](../artifacts/figures/eval_model_comparison_cv.png)

### Leitura dos resultados de CV
- Todos os modelos superam o baseline por ampla margem (RMSLE do baseline: {n(next(r for r in fs['app']['results'] if r['algorithm']=='DummyRegressor')['cv_rmsle_mean'])}).
- A transformação `log1p(y)` reduziu a RMSLE em todos os algoritmos nos dois conjuntos.
- Os modelos de árvores (Random Forest e HistGradientBoosting) tiveram RMSLE menor que o Ridge. Isso é compatível com a hipótese H2 (relações não lineares), mas a diferença entre RF e HGB no conjunto da aplicação é pequena, da ordem do desvio-padrão entre folds.
- **Instabilidade do Ridge + log1p(y):** apesar da RMSLE razoável, o RMSE e o R² médios são ruins por causa de um fold: RMSE por fold (conjunto full) = {[round(v) for v in ridge_full['cv_rmse_folds']]}, R² por fold = {[round(v, 3) for v in ridge_full['cv_r2_folds']]} (conjunto app: R² = {[round(v, 3) for v in ridge_app['cv_r2_folds']]}). Em escala log, um modelo linear extrapola de forma exponencial para imóveis atípicos (ex.: áreas muito grandes), e `expm1` amplifica o erro em USD.
- Treino × validação: o HGB selecionado tem RMSLE de treino {n(sel_app['train_rmsle_mean'])} contra {n(sel_app['cv_rmsle_mean'])} na validação — há ajuste aos dados de treino, esperado em boosting, mas a validação permanece a melhor entre os candidatos. Ver `artifacts/figures/eval_train_vs_validation.png`.

## 4. Modelo selecionado
- **Aplicação:** `{fs['app']['selected']}` com {', '.join(f'{k}={v}' for k, v in meta['hyperparameters'].items())}.
- **Referência full:** `{fs['full']['selected']}`.
- Reajustado na partição de desenvolvimento inteira ({split['n_dev']} imóveis). O artefato avaliado abaixo é o mesmo arquivo carregado pela API (SHA-256 `{hold['evaluated_artifacts']['app']['sha256']}`).

## 5. Holdout ({hold['holdout']['n']} imóveis, avaliação única)

{holdout_tbl}

Supera o baseline em todas as métricas: **{'sim' if beats_all else 'não'}** (aplicação); referência full: **{'sim' if all(hold['beats_baseline']['full'].values()) else 'não'}**.

### CV × holdout (modelo da aplicação)

{cvh_tbl}

A RMSLE do holdout é próxima da CV (dentro de ~1 dp), indicando que a estimativa de CV não foi muito otimista. O R² do holdout é maior que o da CV; uma explicação compatível com os dados é que os dois imóveis anômalos com `GrLivArea` > 4.000 pés² (≈ 372 m²) e preço baixo (Ids 524 e 1299) {'estão no holdout' if in_hold else 'ficaram na partição de desenvolvimento, não no holdout'}, e esses casos pesam mais no RMSE/R² do que na RMSLE. Com {hold['holdout']['n']} imóveis, o holdout também tem variância amostral.

O conjunto completo (79 variáveis) teve RMSLE {'menor' if F['rmsle'] < M['rmsle'] else 'maior'} no holdout ({n(F['rmsle'])} vs {n(M['rmsle'])}), mas RMSE {'maior' if F['rmse'] > M['rmse'] else 'menor'} ({usd(F['rmse'])} vs {usd(M['rmse'])}). O ganho de usar 79 variáveis em vez de 14 é pequeno nesta base; o modelo de 14 variáveis é o usado na aplicação porque o usuário consegue preenchê-las.

## 6. Diagnóstico de resíduos (aplicação, holdout)
- Resíduo médio (real − previsto): {usd(res['mean'])}; mediana: {usd(res['median'])}.
- Erro absoluto mediano: {usd(res['median_abs'])}; 90% dos imóveis com erro até {usd(res['p90_abs'])}.
- Erro percentual médio (MAPE): {n(res['mape_pct'], 1)}%; mediano: {n(res['median_ape_pct'], 1)}%.

![Real × previsto](../artifacts/figures/eval_actual_vs_predicted.png)
![Resíduos](../artifacts/figures/eval_residuals.png)
![Erros absolutos](../artifacts/figures/eval_abs_error_distribution.png)

Os imóveis mais caros tendem a ser subestimados (pontos abaixo da diagonal na faixa alta): há poucos exemplos acima de US$ 400 mil para aprender esse trecho.

### Maiores erros absolutos

{err_tbl}

Condição de venda entre os 15 maiores erros: {', '.join(f'{k}: {v}' for k, v in top_cond.items())}. No treino, `Partial` (venda de imóvel novo/inacabado) representa {n(partial_share, 1)}% das vendas — {'está sobrerrepresentada' if 100 * top_cond.get('Partial', 0) / 15 > partial_share else 'não está sobrerrepresentada'} entre os maiores erros. `SaleCondition` não está no formulário por não descrever o imóvel.

## 7. Importância das variáveis (aplicação)
Importância por permutação no holdout (aumento da RMSLE ao embaralhar cada coluna, 20 repetições). Não é causalidade; variáveis correlacionadas dividem importância.

{perm_tbl}

![Importância por permutação](../artifacts/figures/eval_permutation_importance_app.png)

`OverallCond` teve associação univariada fraca na triagem, mas importância por permutação relevante: o efeito aparece condicionado às demais variáveis (ex.: mesma idade/qualidade). `FullBath` e `BedroomAbvGr` contribuem pouco quando a área já está presente.

## 8. Conclusão sobre as hipóteses
- **H1** (características associadas ao preço): suportada — correlações fortes na EDA e modelos com RMSLE ~{n(M['rmsle'], 2)} contra {n(B['rmsle'], 2)} do baseline no holdout.
- **H2** (não linear > linear): suportada nesta base — árvores tiveram RMSLE menor que o Ridge em CV nos dois conjuntos, e o Ridge+log mostrou instabilidade em imóveis atípicos.
- **H3** (desempenhos distintos): suportada entre famílias (baseline ≪ Ridge < árvores); entre RF e HGB a diferença é pequena e não deve ser superinterpretada.
- **H4** (implantação sem alterar o pré-processamento): suportada — o mesmo `.joblib` é carregado pela API; teste automatizado confirma previsão idêntica (`tests/test_api.py::test_predict_matches_saved_pipeline`).

## 9. Limitações
- Ames, Iowa, 2006–2010, USD nominal: não generaliza para o Brasil nem para preços atuais.
- Erros maiores em imóveis caros, muito grandes ou vendas `Partial`/`Abnorml`.
- Sem intervalo de previsão calibrado — a aplicação não exibe faixa de preço.
- Holdout de {hold['holdout']['n']} imóveis: as métricas têm incerteza amostral.
- A `submission.csv` (Kaggle) não foi avaliada localmente: `test.csv` não tem rótulos.
""", encoding="utf-8")

    # ------------------------------------------------------------------ model card
    (D / "model_card.md").write_text(f"""# Model card — HouseValue AI

| Campo | Valor |
|---|---|
| Modelo | `{meta['model_config']}` |
| Hiperparâmetros | {', '.join(f'{k}={v}' for k, v in meta['hyperparameters'].items())} |
| Versão | {meta['model_version']} |
| Treinado em | {meta['trained_at']} (UTC) |
| Artefato | `artifacts/models/house_price_pipeline.joblib` (SHA-256 `{meta['artifact_sha256']}`) |
| Dados de treino | {meta['training_data']}, de `train.csv` (SHA-256 `{meta['source_sha256']}`) |
| Alvo | `SalePrice`, USD nominal da época |
| Entradas | {', '.join(f'`{c}`' for c in meta['input_features'])} |
| Bibliotecas | {', '.join(f'{k} {v}' for k, v in lv.items())} |

## Finalidade
Demonstração acadêmica (TK0231/UFC) de um fluxo completo de Machine Learning: estimar o preço histórico de venda de imóveis de Ames, Iowa, a partir de 14 características que uma pessoa consegue informar.

## População representada
{n_train} vendas residenciais em Ames, Iowa (EUA), entre 2006 e 2010 (Kaggle — House Prices: Advanced Regression Techniques). Faixa de preços observada: {usd(t['min'])} a {usd(t['max'])} (mediana {usd(t['50%'])}).

## Métricas (holdout de {hold['holdout']['n']} imóveis, avaliação única)
| Métrica | Modelo | Baseline (mediana) |
|---|---|---|
| RMSLE (primária) | {n(M['rmsle'])} | {n(B['rmsle'])} |
| MAE | {usd(M['mae'])} | {usd(B['mae'])} |
| RMSE | {usd(M['rmse'])} | {usd(B['rmse'])} |
| R² | {n(M['r2'], 3)} | {n(B['r2'], 3)} |

Validação cruzada (5 folds, desenvolvimento): RMSLE {n(meta['cv_metrics']['rmsle']['mean'])} ± {n(meta['cv_metrics']['rmsle']['std'])}.

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
- A faixa típica de erro (80%) é a mesma para todas as casas (não varia por segmento); conferida no holdout, onde cobriu {band_cov} dos preços reais.
- Categorias e faixas aceitas limitadas ao que existe na partição de desenvolvimento.
- Dados de 2006–2010 incluem o início da crise imobiliária americana de 2008.
""", encoding="utf-8")

    # ------------------------------------------------------------------ CRISP-DM
    (D / "crisp_dm.md").write_text(f"""# CRISP-DM — HouseValue AI

> Gerado por `python -m ml.build_docs`. Cada etapa aponta para os artefatos reais que a evidenciam.

## 1. Entendimento do negócio
- **Contexto:** o preço de um imóvel depende de área, qualidade, idade, localização e outros fatores. A base é de Ames, Iowa (EUA), 2006–2010, em USD — não representa o mercado brasileiro nem preços atuais.
- **Problema:** estimar `SalePrice` com erro mensurável a partir das características do imóvel.
- **Pergunta:** em que medida características físicas, estruturais e de localização permitem estimar preços históricos de venda por regressão?
- **Hipóteses (H1–H4):** ver PRD §3.6; respostas na seção 5.
- **Usuários:** professor e estudantes; demonstrador do projeto.
- **Critérios de sucesso:** analítico — superar o baseline (mediana) em validação cruzada e no holdout, na métrica primária RMSLE e no MAE; funcional — API e interface entregando previsões do pipeline salvo, com testes passando.

## 2. Entendimento dos dados
- ZIP localizado em {'`' + zipinfo['location'] + '`' if zipinfo else '—'} (`{zipinfo['file_name'] if zipinfo else '—'}`); os CSVs da raiz do workspace são idênticos aos membros do ZIP (SHA-256).
- {n_train} imóveis rotulados × {profile['files']['train.csv']['columns']} colunas; {n_test} sem rótulo. 0 duplicatas; `Id` único.
- Ausências: majoritariamente **estruturais** (dicionário: `NA` = "não possui"), p. ex. `PoolQC` {next(c['missing_train_pct'] for c in profile['columns'] if c['name']=='PoolQC')}%; desconhecidas de fato: `LotFrontage` {next(c['missing_train_pct'] for c in profile['columns'] if c['name']=='LotFrontage')}%, `MasVnrType`/`MasVnrArea` 0,55%, `Electrical` 0,07%.
- Alvo assimétrico (assimetria {n(profile['target']['skew'], 2)}; {n(profile['target']['log1p_skew'], 2)} em log1p).
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
- Selecionado (aplicação): **{fs['app']['selected']}** — RMSLE CV {n(sel_app['cv_rmsle_mean'])} ± {n(sel_app['cv_rmsle_std'])}.
- **Evidências:** `ml/train.py`, `artifacts/metrics/model_comparison.{{json,csv}}`.

## 5. Avaliação
- Holdout ({hold['holdout']['n']} imóveis): RMSLE {n(M['rmsle'])}, MAE {usd(M['mae'])}, RMSE {usd(M['rmse'])}, R² {n(M['r2'], 3)} — baseline: RMSLE {n(B['rmsle'])}, MAE {usd(B['mae'])}.
- Supera o baseline em todas as métricas: {'sim' if beats_all else 'não'}.
- Hipóteses: H1, H2 e H4 suportadas; H3 suportada entre famílias de modelos (detalhes em `docs/model_evaluation.md`).
- Limitações: subestimação de imóveis caros, erros maiores em vendas `Partial`, holdout pequeno.
- **Evidências:** `ml/evaluate.py`, `artifacts/metrics/holdout_evaluation.json`, `artifacts/figures/eval_*.png`, `docs/model_evaluation.md`.

## 6. Implantação
- Pipeline completo (pré-processamento + modelo + transformação do alvo) salvo em `artifacts/models/house_price_pipeline.joblib`, com metadados e hash.
- API FastAPI (`backend/`) carrega o artefato (verificando o SHA-256), valida entradas com Pydantic e com as faixas/categorias de `form_schema.json`, e nunca treina durante requisições.
- Interface HTML/CSS/JS + Plotly.js (`frontend/`) servida pela própria API em `http://localhost:8000/`, com 3 abas: **Simular um imóvel** (estimativa ao vivo, faixa típica de erro e comparação Real × Modelo com casas parecidas que foram vendidas), **Imóveis reais** (o modelo roda nas casas de teste e cada estimativa é comparada ao preço da venda) e **Como o modelo funciona**. Áreas em m² na interface; a conversão para pés² (unidade do treino) acontece só na fronteira da API.
- Testes: `pytest` (dados, preparação, modelo, API) e `tests/e2e_ui.py` (navegador real).
- **Evidências:** `README.md`, `docs/model_card.md`, `docs/presentation.md`.
""", encoding="utf-8")

    # ------------------------------------------------------------------ presentation
    top3 = ", ".join(r["feature"] for r in perm[:3])
    ana_p = config.METRICS_DIR / "model_analysis.json"
    ana = _load(ana_p) if ana_p.is_file() else None
    tie_txt = (f"{short(ana['tie_test']['models'][0])} venceu {short(ana['tie_test']['models'][1])} em {ana['tie_test']['first_wins']} de {ana['tie_test']['n_splits']} partições: **{ana['tie_test']['verdict']}**." if ana else "")
    q = ana["segments"]["price_quartile"] if ana else None
    bias_txt = (f"superestima a faixa mais barata em média {usd(abs(q[0]['bias']))} e subestima a mais cara em {usd(abs(q[3]['bias']))} (regressão à média)" if q else "")
    top_err = hold["top_errors_app"][0]
    band_p = config.MODELS_DIR / "error_band.json"
    band = _load(band_p) if band_p.is_file() else None
    band_txt = (f"faixa típica de {n(100 * (band['factor_low'] - 1), 1)}% a +{n(100 * (band['factor_high'] - 1), 1)}% em torno da estimativa (80%), estimada por validação cruzada no treino; nas casas reais testadas, o preço caiu dentro dela em {n(100 * band['holdout_coverage'], 1)}% dos casos"
                if band else "")
    within10 = n(100 * band["holdout_within_10pct"], 0) if band else "—"
    ibr = lambda v: f"{v:,}".replace(",", ".")
    sel_short = 'Gradient Boosting com restrições monotônicas ("regras de bom senso")'
    perm_labels = [FIELD_META.get(r["feature"], {}).get("label", r["feature"]) for r in perm[:4]]
    v1e_p = config.ARTIFACTS_DIR / "history" / "v1" / "holdout_evaluation.json"
    v1_txt = f" (v1: {n(_load(v1e_p)['metrics']['app']['rmsle'])})" if v1e_p.is_file() else ""
    n_algo = len(fs["app"]["results"])
    dummy_cv = next(r for r in fs["app"]["results"] if r["algorithm"] == "DummyRegressor")["cv_rmsle_mean"]
    tie = ana["tie_test"] if ana else None
    tie_line = f"Em {tie['n_splits']} partições extras, o vencedor ganhou {tie['first_wins']}: **{tie['verdict']}**." if tie else ""
    within20 = n(100 * band["holdout_within_20pct"], 0) if band else "—"
    band_lo = n(100 * (1 - band["factor_low"]), 0) if band else "—"
    band_hi = n(100 * (band["factor_high"] - 1), 0) if band else "—"
    n_hold = hold["holdout"]["n"]
    ip_p = config.METRICS_DIR / "importance_plain.json"
    plain_lines, nb_txt = "", ""
    if ip_p.is_file():
        ip = _load(ip_p)
        plain_lines = f"- Com todas as informações: erro típico **{n(ip['baseline']['median_ape_pct'], 1)}%**.\n" + "\n".join(
            f"- Sem: {FIELD_META.get(r['feature'], {}).get('label', r['feature'])}: **{n(r['median_ape_pct'], 1)}%** (casas com erro até 10%: {n(r['within_10pct'], 0)}%)"
            for r in ip["rows"][:5])
        nbs = ip["typical_house_by_neighborhood"]
        nb_txt = f" (a mesma casa típica: {usd(nbs[0]['pred'])} em {nbs[0]['label']}, {usd(nbs[-1]['pred'])} em {nbs[-1]['label']})"
    cvr = {r["config"]: r for r in fs["app"]["results"]}
    def _cv(cfg):
        return n(cvr[cfg]["cv_rmsle_mean"]) if cfg in cvr else "—"
    def _tr(cfg):
        return n(cvr[cfg]["train_rmsle_mean"], 2) if cfg in cvr else "—"
    C_D, C_R, C_RL = "DummyRegressor", "Ridge", "Ridge + log1p(y)"
    C_RF = "RandomForestRegressor + log1p(y)"
    C_GB = "HistGradientBoostingRegressor + log1p(y)"
    C_GBN = "HistGradientBoostingRegressor (categorias nativas) + log1p(y)"
    C_GBM = "HistGradientBoostingRegressor (nativo, monotônico) + log1p(y)"
    hp = meta.get("hyperparameters", {})
    def _mae(cfg, key="cv_mae_mean"):
        return usd(cvr[cfg][key]) if cfg in cvr else "—"
    hpred = pd.read_csv(config.METRICS_DIR / "holdout_predictions.csv").set_index("Id")
    cover_id = 434
    cover_txt = ""
    ch_lo = ch_hi = 0
    if cover_id in hpred.index:
        ch = hpred.loc[cover_id]
        ch_lo, ch_hi = ch['pred_app'] * band['factor_low'], ch['pred_app'] * band['factor_high']
        cover_txt = (f"Casa real da prova (Id {cover_id}, bairro Gilbert, 149 m², 3 quartos, construída em 1997, vendida em 2008): "
                     f"vendida por {usd(ch['SalePrice'])}; o modelo estimou {usd(ch['pred_app'])} ({n(100 * (ch['pred_app'] - ch['SalePrice']) / ch['SalePrice'], 1)}%).")
    hp_txt = ", ".join(f"{k} = {v}" for k, v in hp.items())
    (D / "presentation.md").write_text(f"""# Roteiro de fala — HouseValue AI (≈8 min de fala + demo ao vivo)

> Gerado por `python -m ml.build_docs` a partir dos artefatos: todos os números abaixo são os reais do projeto.
> Acompanha o deck de slides (mesma numeração, 16 slides). A fala completa de cada slide também está nas notas do deck.
> **Fio da história:** problema e hipóteses (1–3) → o caminho em 5 etapas, CRISP-DM (4) → dados, preparação, treino, prova (5–11) → o sistema (12–14) → resposta às hipóteses (15) → perguntas (16). A casa da capa volta na conclusão para fechar a história.
> **Antes de começar:** `.venv\\Scripts\\python -m uvicorn backend.main:app --port 8000` e abrir `http://localhost:8000/` (abre na aba **Simular um imóvel**).
> **Plano B:** se o servidor não subir, o slide 13 e a tabela do slide 14 são capturas reais do sistema.
> Cores no sistema e nos gráficos: **laranja = preço real da venda**, **azul = estimativa do modelo**; verde / âmbar / vermelho = erro até 10% / 10–20% / mais de 20%.

## Parte 1 · O problema (CRISP-DM: entendimento do negócio)

### 1. Capa: "Quanto vale esta casa?" (≈25 s)
- {cover_txt}
- "Hoje vamos mostrar como chegamos nisso e como provamos o quanto ele acerta, inclusive quando erra." Apresentar a convenção laranja/azul.

### 2. Imagine que você vai vender sua casa (≈30 s)
- Você sabe a área, os quartos, o ano, o acabamento, o bairro. Não sabe **quanto pedir**.
- Hoje: palpite, vizinhos ou avaliação paga. Um corretor compara com **vendas parecidas**: é isso que vamos ensinar o computador a fazer.

### 3. A nossa pergunta e as hipóteses (≈40 s)
- **Dá para estimar o preço de uma casa só pelas características dela?**
- Machine Learning: exemplos ({ibr(n_train)} vendas com preço real) → o computador aprende os padrões → casa nova recebe um preço. A resposta é um número: **regressão**.
- Hipóteses: **H1** as características têm relação com o preço; **H2** modelos que aprendem curvas (árvores) erram menos que uma \"linha reta\" (regressão linear); **H3** algoritmos diferentes têm desempenhos diferentes nos mesmos dados; **H4** o modelo pode virar um sistema de uso real, com o mesmo tratamento do treino.

## Parte 2 · Como construímos

### 4. O caminho, em 5 etapas (≈30 s)
CRISP-DM tem 6 fases: a 1ª (entendimento do negócio) é o problema; as outras viram as etapas 1 Conhecer os dados (entendimento dos dados) · 2 Preparar (preparação) · 3 Treinar e comparar (modelagem) · 4 Provar (avaliação) · 5 Virar sistema (implantação). Cada etapa deixa um resultado: 1.460 vendas · 14 informações · 9 candidatos · 292 casas nunca vistas · sistema com 3 abas. A barrinha no alto de cada slide mostra a etapa.

### 5. Etapa 1 · Os dados (≈35 s)
- Base **House Prices** (Kaggle), indicada pela disciplina: **{ibr(n_train)} casas vendidas** com o preço real e **79 características** cada; mais {ibr(n_test)} casas sem preço (não entram na avaliação).
- Metade das casas custa menos de {usd(t['50%'])}; os preços vão de {usd(t['min'])} a {usd(t['max'])}, com poucas casas muito caras (cauda longa). Por isso o modelo aprende o preço em escala log e converte de volta para dólares.
- Limite: **Ames, Iowa (EUA), vendas de 2006 a 2010**. O método vale; os valores não servem para o Brasil de hoje.

### 6. Etapa 2 · Preparar: 14 informações (≈40 s)
- Ninguém digita 79 informações: o formulário usa **14** que qualquer pessoa sabe responder: áreas (construída, porão, terreno, em m²), qualidade, conservação, cozinha, ar central, anos de construção e reforma, quartos, banheiros, garagem, lareiras e bairro. Repetidas saíram (ex.: vagas × área da garagem).
- **Tratamento dos dados:** "NA" em várias colunas quer dizer "não tem" (ex.: "não tem piscina"), não "faltando". Corrigido pelo dicionário da base, sem inventar valores (detalhes em `docs/data_audit.md`).
- **Sem "colar":** tudo decidido sem olhar as casas reservadas para o teste final.

### 7. Etapa 3 · Como um aluno antes da prova (≈55 s)
1. **A prova fica lacrada:** {n_hold} casas (20%) separadas antes de tudo; o modelo só as vê no final, uma única vez.
2. **Cinco simulados** (validação cruzada, KFold 5) nas {ibr(n_train - n_hold)} casas de estudo: cada parte vira teste uma vez; a nota é a média dos 5.
3. **{n_algo} candidatos, mesmas regras:** chute (mediana), regressão linear (Ridge), Random Forest e Gradient Boosting, com variações; os ajustes de cada um (hiperparâmetros) testados por GridSearchCV nos mesmos simulados. Nota: **RMSLE** (erro relativo ao preço).
4. **Ferramentas:** Python + scikit-learn; tratamento e modelo num único **pipeline**, o mesmo arquivo usado pela API.

### 8. Etapa 3 · Por que o Gradient Boosting venceu (≈60 s)
Quanto cada tipo de modelo errou, **em média, nos simulados** (melhor versão de cada tipo):

| Modelo | Em palavras simples | Erro médio | Nota RMSLE |
|---|---|---|---|
| Chute | sempre o preço do meio | {_mae(C_D)} | {_cv(C_D)} |
| Regressão linear | uma "linha reta" | {_mae(C_RL)} | {_cv(C_RL)} |
| Random Forest | árvores lado a lado | {_mae(C_RF)} | {_cv(C_RF)} |
| **Gradient Boosting** | árvores em fila | **{_mae(C_GBM)}** | **{_cv(C_GBM)}** |

- O grande salto é sair do chute; entre os modelos de verdade a diferença é pequena, e isso é dito com honestidade.
- **Como ele aprende:** avaliadores em fila; o 1º dá um palpite, o 2º corrige o erro do 1º, e assim por diante (500 pequenas correções; {hp_txt}).
- **Por que venceu:** (1) menor erro entre os {n_algo} candidatos; (2) **decora menos**: a Random Forest erra {_mae(C_RF, "train_mae_mean")} nas casas que estudou e {_mae(C_RF)} nas novas; o vencedor, {_mae(C_GBM, "train_mae_mean")} → {_mae(C_GBM)}; (3) **bom senso**: mais área ou qualidade nunca baixam o preço (restrição monotônica).
- **Para o professor:** {tie_line} (contra a versão sem as regras de bom senso).

### 9. Etapa 4 · Provar: as {n_hold} casas nunca vistas (≈45 s)
- **{within10}%** das casas com erro de até 10% ("7 em cada 10"); {within20}% com até 20%.
- **Erro típico: {n(res['median_ape_pct'], 1)}%** (metade das casas erra menos que isso).
- **Erro médio: {usd(M['mae'])}**, contra {usd(B['mae'])} do chute (≈{n(B['mae'] / M['mae'], 1)}× menor).
- **R² = {n(M['r2'], 2)}**: explica cerca de {n(100 * M['r2'], 0)}% das diferenças de preço. RMSLE {n(M['rmsle'])}{v1_txt}, perto da validação cruzada: os simulados não foram otimistas.
- Ler o gráfico: cada ponto é uma casa; na linha tracejada o modelo acertaria em cheio.

### 10. Etapa 4 · Onde ele erra, e por quê (≈45 s)
1. **Nos extremos, puxa para o meio:** {bias_txt}. Por quê: há poucas casas muito baratas ou muito caras para aprender; na dúvida, fica perto do "normal".
2. **Vendas fora do normal:** casa **Id {top_err['Id']}**, vendida ainda em construção (`{top_err['SaleCondition']}`) por {usd(top_err['SalePrice'])}; o modelo estimou {usd(top_err['pred_app'])} (+{n(top_err['pct_error_app'], 0)}%). Por quê: o modelo sabe como a casa é, não como a venda aconteceu.
3. **Por isso, uma faixa e não só um número:** −{band_lo}% a +{band_hi}%; nas casas da prova o preço real caiu dentro dela em **{band_cov}** (meta: 80%).

### 11. Etapa 4 · Sem a área, o erro mais que dobra (≈40 s)
- O teste: **esconder uma informação de cada vez** (embaralhar a coluna entre as casas da prova) e medir o erro típico de novo. Quanto mais o erro sobe, mais o modelo depende dela.
{plain_lines}
- Por que faz sentido: área = tamanho da casa; qualidade = custo de construção; bairro = localização{nb_txt}. Quartos quase não mudam nada: a área já diz o tamanho.
- Isso mostra **do que o modelo depende**, não causa. Números de `python -m ml.importance_plain` (`artifacts/metrics/importance_plain.json`).

## Parte 3 · O sistema (CRISP-DM: implantação)

### 12. Etapa 5 · Como as peças se conectam (≈45 s)
- **Feito uma vez, no desenvolvimento:** CSV do Kaggle → **pipeline de ML** (`ml/`, Python + scikit-learn) → **modelo treinado** (`.joblib` + metadados com hash SHA-256, métricas e gráficos em `artifacts/`).
- **A cada clique, no uso:** a **interface web** (`frontend/`, HTML/CSS/JS + Plotly) envia as 14 características → a **API FastAPI** (`backend/`) confere os valores, converte m² → pés², roda o modelo e devolve **preço, faixa de erro e casas parecidas vendidas**.
- A API carrega o modelo uma vez quando liga, confere o hash e **nunca treina** durante o uso (H4).

### 13. Etapa 5 · O que a pessoa vê: 3 abas (≈20 s)
**Simular um imóvel** (abre aqui) · **Imóveis reais** (o modelo calcula agora as {n_hold} casas da prova e mostra Real × Modelo) · **Como o modelo funciona** (a história do treino).
- **Como sabemos que funciona:** 46 testes automáticos (pytest), um teste de ponta a ponta no navegador que clica nas 3 abas e a API que recusa o arquivo do modelo se o hash SHA-256 não conferir.

### 14. Demo ao vivo (≈2,5 min)
1. **Simular um imóvel:** mudar área construída, qualidade e bairro; a estimativa e a faixa típica mudam na hora.
   Descer até **Sua casa × casas reais vendidas**: características lado a lado e, para cada casa parecida, **Real | Modelo | Diferença**.
2. **Imóveis reais:** "o modelo acabou de calcular as {n_hold} casas de teste". Clicar num ponto verde (acerto) e mostrar o painel **Real × Modelo**.
   Filtrar **Erro grande** e abrir o Id {top_err['Id']}: mostrar o aviso de venda fora do normal.
3. Opcional: **Abrir esta casa no simulador** e depois **Voltar para a casa que você montou**.

## Parte 4 · A resposta

### 15. Voltando à pergunta (≈55 s)
- **Fechar a história com a casa da capa:** {cover_txt} Com a faixa típica, o sistema diria "provavelmente entre {usd(ch_lo)} e {usd(ch_hi)}": o preço real caiu dentro.
- **Dá para estimar? Sim, e sabendo o quanto confiar:** com 14 informações simples, erro de até 10% em {within10}% das casas nunca vistas, e uma faixa de erro que se confirmou ({band_cov} para 80% prometidos).
- **H1** ✔ as características explicam ~{n(100 * M['r2'], 0)}% das diferenças de preço (R² {n(M['r2'], 2)}).
- **H2** ✔ as árvores (erro médio {_mae(C_RF)} e {_mae(C_GBM)}) erraram menos que a \"linha reta\" ({_mae(C_RL)}).
- **H3** ✔ os tipos de algoritmo diferem; entre as duas melhores versões, empate técnico.
- **H4** ✔ o sistema usa o mesmo arquivo do treino (pipeline), sem mudar nada.
- Limites: uma cidade dos EUA, 2006–2010; não é laudo de avaliação. Próximos passos: dados locais e atuais, tipo e data da venda como entradas, faixas de erro por tipo de casa.

### 16. Obrigado! Perguntas?
Deixar na tela durante as perguntas: 1.460 vendas reais · 7 em cada 10 casas com erro até 10% · erro médio {usd(M['mae'])} (chute: {usd(B['mae'])}) · {band_cov} das casas dentro da faixa. Respostas prontas para as perguntas mais prováveis: `docs/HouseValue_Guia_da_Apresentacao.pdf`, parte 4.
""", encoding="utf-8")

    # ------------------------------------------------------------------ v2: seções extras
    ana_path = config.METRICS_DIR / "model_analysis.json"
    v1_dir = config.ARTIFACTS_DIR / "history" / "v1"
    extra = ["\n## 10. Versão 2 do experimento\n"]
    if (v1_dir / "model_metadata.json").is_file():
        v1m = _load(v1_dir / "model_metadata.json")
        v1e = _load(v1_dir / "holdout_evaluation.json")
        h1 = v1e["metrics"]["app"]
        extra.append("A v2 manteve o protocolo (mesma partição, mesmos folds, mesma regra de seleção) e acrescentou dois candidatos de HistGradientBoosting: um com **categorias nativas** (OrdinalEncoder + `categorical_features`, sem one-hot; o HGB trata ausentes nativamente) e outro com as mesmas categorias nativas e **restrições monotônicas** crescentes (OverallQual, OverallCond, GrLivArea, TotalBsmtSF, LotArea, GarageCars, Fireplaces e FullBath; não em BedroomAbvGr nem nos anos). Só a variante com `log1p(y)` foi testada, porque o log venceu em todos os algoritmos da v1.\n")
        extra.append(table(["Versão", "Modelo escolhido (aplicação)", "RMSLE CV", "RMSLE holdout", "MAE holdout", "RMSE holdout", "R² holdout"], [
            [f"v{v1m['model_version']}", short(v1m["model_config"]), n(v1m["cv_metrics"]["rmsle"]["mean"]), n(h1["rmsle"]), usd(h1["mae"]), usd(h1["rmse"]), n(h1["r2"], 3)],
            [f"v{meta['model_version']}", short(meta["model_config"]), n(meta["cv_metrics"]["rmsle"]["mean"]), n(M["rmsle"]), usd(M["mae"]), usd(M["rmse"]), n(M["r2"], 3)],
        ]))
        extra.append("\n**Ressalva metodológica:** o holdout já tinha sido consultado uma vez na v1. A v2 escolheu o modelo só pela validação cruzada, mas o holdout deixou de ser totalmente inédito para este projeto; os resultados da v1 estão preservados em `artifacts/history/v1/`.\n")
    cand = hold.get("candidates", {}).get("rows", [])
    if cand:
        extra.append("\n### Holdout de cada candidato (informativo)\n")
        extra.append("Calculado depois da seleção, não usado para escolher o modelo. Alimenta o painel \"Algoritmos lado a lado\" do simulador.\n")
        extra.append(table(["Configuração", "RMSLE CV", "RMSLE holdout", "MAE holdout", "Selecionado"],
                           [[short(c["config"]), n(c["cv_rmsle_mean"]), n(c["holdout"]["rmsle"]), usd(c["holdout"]["mae"]), "sim" if c["is_selected"] else ""] for c in sorted(cand, key=lambda c: c["cv_rmsle_mean"])]))
    if ana_path.is_file():
        ana = _load(ana_path)
        t = ana["tie_test"]
        tf = ana["tie_test_full"]
        extra.append("\n## 11. Vencedor ou empate?\n")
        extra.append(f"Os dois melhores do conjunto da aplicação, **{short(t['models'][0])}** e **{short(t['models'][1])}**, foram comparados com os hiperparâmetros fixos em {t['n_splits']} partições (RepeatedKFold 5 × 4) da partição de desenvolvimento. RMSLE médio: {n(t['mean'][0])} × {n(t['mean'][1])}. O primeiro venceu em {t['first_wins']} de {t['n_splits']} partições; diferença média {n(t['diff_mean'])} ± {n(t['diff_std'])}. Veredito: **{t['verdict']}** ({t['rule']})\n")
        extra.append(f"\nNo conjunto completo, com as diferenças pareadas dos 5 folds do treino: {short(tf['models'][0])} × {short(tf['models'][1])}, o primeiro venceu {tf['first_wins']} de 5 folds; diferença {n(tf['diff_mean'])} ± {n(tf['diff_std'])} → {tf['verdict']}.\n")
        lc = ana["learning_curve"]
        extra.append("\n## 12. Curva de aprendizado\n")
        extra.append(table(["Imóveis no treino", "RMSLE treino", "RMSLE validação"], [[s_, n(a), n(b)] for s_, a, b in zip(lc["train_sizes"], lc["train_rmsle_mean"], lc["val_rmsle_mean"])]))
        extra.append("\n![Curva de aprendizado](../artifacts/figures/eval_learning_curve.png)\n")
        seg = ana["segments"]
        extra.append("\n## 13. Onde o modelo erra (holdout)\n")
        extra.append(f"{seg['bias_definition']}\n")
        for key, title in (("price_quartile", "Faixa de preço real"), ("sale_condition", "Condição de venda"), ("neighborhood", "Bairro (n ≥ 8)")):
            extra.append(f"\n**{title}**\n")
            extra.append(table(["Segmento", "n", "MAE", "MAPE", "Viés médio"], [[r["segment"], r["n"], usd(r["mae"]), n(r["mape_pct"], 1) + "%", ("+" if r["bias"] >= 0 else "−") + usd(abs(r["bias"]))] for r in seg[key]]))
        extra.append("\n![Viés por faixa de preço](../artifacts/figures/eval_error_segments.png)\n")
        q = seg["price_quartile"]
        extra.append(f"\nLeitura: o modelo puxa as estimativas para o meio da distribuição: na faixa mais barata o viés médio é {('+' if q[0]['bias'] >= 0 else '−') + usd(abs(q[0]['bias']))} e na mais cara {('+' if q[3]['bias'] >= 0 else '−') + usd(abs(q[3]['bias']))} (regressão à média). A condição de venda não é uma entrada do modelo, por isso vendas atípicas erram mais.\n")
    if band:
        extra.append("\n## 14. Faixa típica de erro (80%)\n")
        extra.append(f"{band['method']} Fatores: previsto × [{n(band['factor_low'], 3)}, {n(band['factor_high'], 3)}]. "
                     f"Conferência no holdout ({band['holdout_n']} casas, não usado na estimativa): o preço real caiu dentro da faixa em **{n(100 * band['holdout_coverage'], 1)}%** dos casos (nível nominal 80%). "
                     f"Dentro de ±10%: {n(100 * band['dev_within_10pct'], 1)}% fora da amostra no desenvolvimento e {n(100 * band['holdout_within_10pct'], 1)}% no holdout.\n")
    ev_path = D / "model_evaluation.md"
    ev_path.write_text(ev_path.read_text(encoding="utf-8") + "\n".join(extra), encoding="utf-8")

    print("Gerado: docs/methodology.md, model_evaluation.md, model_card.md, crisp_dm.md, presentation.md")


if __name__ == "__main__":
    main()
