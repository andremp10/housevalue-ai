"""Artefatos auxiliares da interface (sem treino e sem nenhuma decisão de modelagem).

Uso: python -m ml.app_assets   (depois de ml.train e ml.evaluate)

Gera:
- artifacts/models/field_distributions.json
    Distribuição de cada variável numérica do formulário na PARTIÇÃO DE
    DESENVOLVIMENTO (os dados com que o modelo foi treinado): histograma,
    para desenhar atrás dos sliders, e grade de quantis, usada pela curva de
    sensibilidade para varrer só regiões onde há dados de treino.
- artifacts/reports/holdout_samples.json
    Imóveis reais do holdout com as 14 variáveis, o preço real e a previsão
    do pipeline salvo, para a pessoa testar o modelo em casos verdadeiros.
- artifacts/models/error_band.json
    Faixa típica de erro (80%): estimada com previsões fora da amostra
    (cross_val_predict, mesmos 5 folds) na partição de desenvolvimento e
    conferida no holdout (cobertura real). Não usa o holdout para ser estimada.
- artifacts/models/form_schema.json e os nomes da API em model_metadata.json
    são regenerados a partir de ml/features.py (rótulos e unidades), sem retreino.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict

from ml import config
from ml.data import load_train
from ml.features import APP_FEATURES, FIELD_META
from ml.train import build_form_schema

N_QUANTILES = 41  # passos de 2,5%


def histogram(values: np.ndarray, integer: bool = True, max_bins: int = 28) -> dict:
    lo, hi = float(values.min()), float(values.max())
    if integer and hi - lo + 1 <= max_bins:
        edges = np.arange(lo - 0.5, hi + 1.5, 1.0)
    else:
        edges = np.linspace(lo, hi, max_bins + 1)
    counts, edges = np.histogram(values, bins=edges)
    return {"counts": counts.tolist(), "edges": [round(float(e), 2) for e in edges]}


def main() -> None:
    train = load_train()
    split = json.loads(config.SPLIT_PATH.read_text(encoding="utf-8"))
    meta = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    dev = train[train[config.ID_COLUMN].isin(split["dev_ids"])]
    # esquema do formulário e nomes da API a partir de ml/features.py (sem retreino)
    form = build_form_schema(dev, meta["numeric_features"], meta["categorical_features"])
    config.FORM_SCHEMA_PATH.write_text(json.dumps(form, indent=2, ensure_ascii=False), encoding="utf-8")
    meta["input_api_names"] = {c: FIELD_META[c]["api_name"] for c in APP_FEATURES}
    config.MODEL_METADATA_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    hold = train[train[config.ID_COLUMN].isin(split["holdout_ids"])].sort_values(config.ID_COLUMN)

    dists = {"partition": f"desenvolvimento ({len(dev)} imóveis)", "fields": {}}
    for f in form["fields"]:
        if f["type"] != "integer":
            continue
        v = dev[f["name"]].to_numpy()
        if v.max() - v.min() + 1 <= 15:  # poucos valores inteiros: todos os da faixa de treino
            grid = list(range(int(v.min()), int(v.max()) + 1))
        else:  # faixa larga: quantis, para seguir a densidade dos dados de treino
            grid = sorted({int(round(q)) for q in np.quantile(v, np.linspace(0, 1, N_QUANTILES))})
        small = v.max() - v.min() + 1 <= 15
        # Faixa comum do slider: até o percentil 97,5 (variáveis muito assimétricas, ex.: LotArea).
        common_max = int(v.max()) if (small or f.get("unit") == "ano") else int(np.ceil(np.quantile(v, 0.975)))
        common = v[v <= common_max]
        dists["fields"][f["name"]] = {
            "histogram": histogram(common, max_bins=32),
            "common_range": [int(v.min()), common_max],
            "above_common": int((v > common_max).sum()),
            "quantile_grid": grid,
            "percentiles": {str(q): float(np.quantile(v, q / 100)) for q in (5, 25, 50, 75, 95)},
        }
    (config.MODELS_DIR / "field_distributions.json").write_text(json.dumps(dists, ensure_ascii=False), encoding="utf-8")

    model = joblib.load(config.APP_MODEL_PATH)
    preds = model.predict(hold[APP_FEATURES])
    samples = []
    for (_, row), p in zip(hold.iterrows(), preds):
        samples.append(
            {
                "id": int(row[config.ID_COLUMN]),
                "features": {c: (int(row[c]) if isinstance(row[c], (int, np.integer)) else str(row[c])) for c in APP_FEATURES},
                "actual": float(row[config.TARGET]),
                "predicted": round(float(p), 2),
                "sale_condition": str(row["SaleCondition"]),
                "year_sold": int(row["YrSold"]),
            }
        )
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "holdout_samples.json").write_text(
        json.dumps({"source": "holdout (20%), nunca usado no treino", "n": len(samples), "samples": samples}, ensure_ascii=False), encoding="utf-8"
    )
    # ---------------------------------------------------------- faixa típica de erro
    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    oof = cross_val_predict(clone(model), dev[APP_FEATURES], dev[config.TARGET], cv=cv, n_jobs=1)
    y_dev = dev[config.TARGET].to_numpy()
    d = np.log(y_dev) - np.log(oof)  # >0: real acima do previsto
    lo, hi = float(np.exp(np.quantile(d, 0.10))), float(np.exp(np.quantile(d, 0.90)))
    y_h = hold[config.TARGET].to_numpy()
    inside = (y_h >= preds * lo) & (y_h <= preds * hi)
    band = {
        "level": 0.80,
        "factor_low": lo, "factor_high": hi,
        "method": "Quantis 10% e 90% de log(real) − log(previsto) em previsões fora da amostra (cross_val_predict, KFold 5, seed 42) na partição de desenvolvimento.",
        "dev_n": int(len(dev)),
        "dev_within_10pct": float(np.mean(np.abs(oof - y_dev) / y_dev <= 0.10)),
        "holdout_n": int(len(hold)),
        "holdout_coverage": float(inside.mean()),
        "holdout_within_10pct": float(np.mean(np.abs(preds - y_h) / y_h <= 0.10)),
        "holdout_within_20pct": float(np.mean(np.abs(preds - y_h) / y_h <= 0.20)),
        "explanation": "Faixa onde o preço real caiu em cerca de 80% das casas nos testes. Estimada sem olhar o holdout e conferida nele depois.",
    }
    (config.MODELS_DIR / "error_band.json").write_text(json.dumps(band, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"field_distributions.json: {len(dists['fields'])} variáveis | holdout_samples.json: {len(samples)} imóveis")
    print(f"Faixa 80%: previsto × [{lo:.3f}, {hi:.3f}] | cobertura no holdout: {band['holdout_coverage']:.1%} | "
          f"dentro de ±10%: dev {band['dev_within_10pct']:.1%}, holdout {band['holdout_within_10pct']:.1%}")


if __name__ == "__main__":
    main()
