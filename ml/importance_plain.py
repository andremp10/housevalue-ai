"""Importância por permutação em linguagem simples (para a apresentação).

Para cada uma das 14 características do formulário, embaralha a coluna nas casas de teste (holdout)
20 vezes e mede, com o modelo salvo (sem retreinar):
- o erro típico (mediana do erro percentual absoluto);
- a % de casas com erro de até 10%.
Também calcula a estimativa da casa típica em cada bairro (mesma casa, só o bairro muda).

Uso: python -m ml.importance_plain  →  artifacts/metrics/importance_plain.json
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from ml import config, data

N_REPEATS = 20
SEED = 42


def main() -> None:
    split = json.loads((config.METRICS_DIR / "split.json").read_text(encoding="utf-8"))
    train = data.load_train()
    hold = train[train[config.ID_COLUMN].isin(split["holdout_ids"])].sort_values(config.ID_COLUMN)
    pipe = joblib.load(config.MODELS_DIR / "house_price_pipeline.joblib")
    fields = json.loads((config.MODELS_DIR / "form_schema.json").read_text(encoding="utf-8"))["fields"]
    cols = [f["name"] for f in fields]
    X = hold[cols].copy()
    y = hold["SalePrice"].to_numpy()

    def stats(Xm: pd.DataFrame) -> tuple[float, float]:
        ape = np.abs(pipe.predict(Xm) - y) / y * 100
        return float(np.median(ape)), float((ape <= 10).mean() * 100)

    base_med, base_w10 = stats(X)
    rng = np.random.default_rng(SEED)
    rows = []
    for c in cols:
        med, w10 = [], []
        for _ in range(N_REPEATS):
            Xs = X.copy()
            Xs[c] = rng.permutation(Xs[c].to_numpy())
            a, b = stats(Xs)
            med.append(a)
            w10.append(b)
        rows.append({"feature": c, "median_ape_pct": float(np.mean(med)), "within_10pct": float(np.mean(w10))})
    rows.sort(key=lambda r: -r["median_ape_pct"])

    # casa típica: mediana das numéricas, categoria mais frequente no treino
    typical = {}
    for f in fields:
        typical[f["name"]] = (max(f["options"], key=lambda o: o["n_train"])["value"]
                              if f.get("options") else f["median"])
    nb = next(f for f in fields if f["name"] == "Neighborhood")
    by_nb = []
    for o in nb["options"]:
        p = float(pipe.predict(pd.DataFrame([dict(typical, Neighborhood=o["value"])]))[0])
        by_nb.append({"neighborhood": o["value"], "label": o["label"], "n_train": o["n_train"], "pred": p})
    by_nb.sort(key=lambda r: r["pred"])

    out = {
        "method": f"Cada característica embaralhada {N_REPEATS} vezes nas {len(hold)} casas de teste; médias das repetições. Modelo salvo, sem retreino.",
        "baseline": {"median_ape_pct": base_med, "within_10pct": base_w10},
        "rows": rows,
        "typical_house": typical,
        "typical_house_by_neighborhood": by_nb,
    }
    path = config.METRICS_DIR / "importance_plain.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Gerado:", path)
    for r in rows[:5]:
        print(f"  {r['feature']:14s} erro típico {r['median_ape_pct']:.1f}% · até 10%: {r['within_10pct']:.1f}%")
    print(f"  bairro: {by_nb[0]['label']} US$ {by_nb[0]['pred']:,.0f} … {by_nb[-1]['label']} US$ {by_nb[-1]['pred']:,.0f}")


if __name__ == "__main__":
    main()
