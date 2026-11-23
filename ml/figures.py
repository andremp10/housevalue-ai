"""Figuras de avaliação desenhadas a partir dos artefatos já salvos.

Uso: python -m ml.figures   (depois de ml.evaluate e ml.model_analysis)

Não treina nem avalia nada: só lê
- artifacts/metrics/holdout_predictions.csv   (real × previsto no holdout)
- artifacts/metrics/model_comparison.json     (validação cruzada de cada configuração)
- artifacts/metrics/feature_importance.json   (permutação no holdout)
- artifacts/metrics/model_analysis.json       (curva de aprendizado, viés por faixa)
e redesenha as figuras eval_*.png com a convenção visual de ml/plot_style.py.
ml.evaluate e ml.model_analysis chamam estas mesmas funções ao final.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ml import config
from ml.features import FIELD_META
from ml.plot_style import (ACCENT, BAD, GOOD, INK, MODEL, MUTED, NEUTRAL, NEUTRAL_SOFT, WARN,
                           algo_name, save, status_color, style, usd_k)

ANALYSIS_PATH = config.METRICS_DIR / "model_analysis.json"


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def holdout_figures() -> None:
    df = pd.read_csv(config.METRICS_DIR / "holdout_predictions.csv")
    y, p = df["SalePrice"].to_numpy(), df["pred_app"].to_numpy()
    pct = 100 * (p - y) / y

    # Real × modelo: um ponto por casa, cor = situação do erro
    lim = [min(y.min(), p.min()) * 0.92, max(y.max(), p.max()) * 1.04]
    fig, ax = plt.subplots(figsize=(7, 6.4))
    xs = np.array(lim)
    ax.fill_between(xs, xs * 0.9, xs * 1.1, color=GOOD, alpha=0.07, linewidth=0)
    ax.plot(lim, lim, color=MUTED, linewidth=1, linestyle="--")
    ax.scatter(y, p, s=16, c=[status_color(v) for v in pct], alpha=0.75, linewidth=0)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.xaxis.set_major_formatter(usd_k); ax.yaxis.set_major_formatter(usd_k)
    n = len(y)
    counts = [(abs(pct) <= 10).sum(), ((abs(pct) > 10) & (abs(pct) <= 20)).sum(), (abs(pct) > 20).sum()]
    handles = [Line2D([], [], marker="o", linestyle="", color=c, label=f"{lab}: {k} de {n}")
               for c, lab, k in zip((GOOD, WARN, BAD), ("Perto do real (até 10%)", "Erro moderado (10–20%)", "Erro grande (mais de 20%)"), counts)]
    handles.append(Line2D([], [], color=MUTED, linestyle="--", label="acerto perfeito"))
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=9)
    style(ax, f"Preço real × estimativa do modelo ({n} casas de teste)", "Preço real da venda (US$)", "Estimativa do modelo (US$)")
    save(fig, "eval_actual_vs_predicted.png")

    # Diferença (modelo − real) × preço real: mostra a regressão à média
    diff = p - y
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.scatter(y, diff, s=14, c=[status_color(v) for v in pct], alpha=0.75, linewidth=0)
    ax.xaxis.set_major_formatter(usd_k); ax.yaxis.set_major_formatter(usd_k)
    style(ax, "Quanto o modelo erra em cada casa (modelo − real)", "Preço real da venda (US$)", "Diferença (US$)")
    save(fig, "eval_residuals.png")

    # Distribuição do erro percentual absoluto
    ape = np.abs(pct)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.arange(0, min(ape.max(), 60) + 2.5, 2.5)
    counts_h, edges = np.histogram(np.clip(ape, 0, bins[-1] - 1e-9), bins=bins)
    ax.bar(edges[:-1], counts_h, width=np.diff(edges) * 0.92, align="edge",
           color=[status_color(e + 0.01) for e in edges[:-1]], alpha=0.85)
    med = float(np.median(ape))
    ax.axvline(med, color=INK, linewidth=1.5)
    ax.annotate(f"erro típico (mediana): {med:.1f}%".replace(".", ","), (med, counts_h.max()), xytext=(8, -4),
                textcoords="offset points", color=INK, fontsize=9.5, va="top")
    style(ax, "Tamanho do erro nas casas de teste", "Erro em relação ao preço real (%)", "Casas", grid="y")
    save(fig, "eval_abs_error_distribution.png")


def comparison_figures() -> None:
    fs = _json(config.MODEL_COMPARISON_PATH)["feature_sets"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for ax, key, title in zip(axes, ("app", "full"), ("14 características (as do formulário)", "Todas as 79 características")):
        rows = sorted(fs[key]["results"], key=lambda r: r["cv_rmsle_mean"], reverse=True)
        colors = [ACCENT if r["config"] == fs[key]["selected"] else (NEUTRAL_SOFT if r["algorithm"] == "DummyRegressor" else NEUTRAL) for r in rows]
        ax.barh([algo_name(r["config"]) for r in rows], [r["cv_rmsle_mean"] for r in rows], xerr=[r["cv_rmsle_std"] for r in rows],
                color=colors, height=0.62, error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 0})
        for i, r in enumerate(rows):
            ax.text(r["cv_rmsle_mean"] + r["cv_rmsle_std"] + 0.006, i, f"{r['cv_rmsle_mean']:.4f}".replace(".", ","), va="center", fontsize=8.5, color=MUTED)
        style(ax, title, "Erro na validação cruzada (RMSLE, menor é melhor)", "", grid="x")
    fig.suptitle("9 configurações comparadas nas mesmas 5 partições (escolhido em verde-azulado)", x=0.01, ha="left", color=INK, fontsize=11)
    save(fig, "eval_model_comparison_cv.png")

    rows = sorted((r for r in fs["app"]["results"] if r["algorithm"] != "DummyRegressor"), key=lambda r: r["cv_rmsle_mean"], reverse=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    yy = np.arange(len(rows))
    ax.barh(yy + 0.2, [r["train_rmsle_mean"] for r in rows], height=0.38, color=NEUTRAL_SOFT, label="nos dados de treino")
    ax.barh(yy - 0.2, [r["cv_rmsle_mean"] for r in rows], height=0.38, color=NEUTRAL, label="em dados não vistos (validação)")
    ax.set_yticks(yy, [algo_name(r["config"]) for r in rows])
    style(ax, "Treino × validação: o quanto cada modelo decora os dados", "RMSLE (menor é melhor)", "", grid="x")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "eval_train_vs_validation.png")


def importance_figure() -> None:
    rows = _json(config.FEATURE_IMPORTANCE_PATH)["app"]["permutation_holdout"]["rows"][::-1]
    labels = [FIELD_META.get(r["feature"], {}).get("label", r["feature"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.barh(labels, [r["importance_mean"] for r in rows], xerr=[r["importance_std"] for r in rows],
            color=MODEL, height=0.62, error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 0})
    style(ax, "O que mais pesa na estimativa (casas de teste)", "Quanto o erro piora ao embaralhar a característica (RMSLE)", "", grid="x")
    save(fig, "eval_permutation_importance_app.png")


def analysis_figures() -> None:
    a = _json(ANALYSIS_PATH)
    lc = a["learning_curve"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(lc["train_sizes"], lc["train_rmsle_mean"], "o-", color=MODEL, alpha=0.45, label="nos dados de treino")
    ax.plot(lc["train_sizes"], lc["val_rmsle_mean"], "o--", color=MODEL, label="em dados não vistos (validação)")
    lo = np.array(lc["val_rmsle_mean"]) - lc["val_rmsle_std"]
    hi = np.array(lc["val_rmsle_mean"]) + lc["val_rmsle_std"]
    ax.fill_between(lc["train_sizes"], lo, hi, color=MODEL, alpha=0.1, linewidth=0)
    style(ax, "Curva de aprendizado: mais casas, menos erro", "Casas usadas no treino", "RMSLE (menor é melhor)")
    ax.legend(frameon=False)
    save(fig, "eval_learning_curve.png")

    pq = a["segments"]["price_quartile"]
    bias = [-r["bias"] for r in pq]  # salvo como real − previsto; a figura mostra modelo − real
    labels = [r["segment"].replace(": ", "\n") for r in pq]
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.bar(labels, bias, color=NEUTRAL, width=0.6)
    ax.axhline(0, color=MUTED, linewidth=1)
    for i, b in enumerate(bias):
        ax.text(i, b + (900 if b >= 0 else -900), f"{'+' if b >= 0 else '−'}US$ {abs(b) / 1000:.1f} mil".replace(".", ","),
                ha="center", va="bottom" if b >= 0 else "top", fontsize=9.5, color=INK)
    ax.yaxis.set_major_formatter(usd_k)
    ax.set_ylim(min(bias) * 1.35, max(bias) * 1.35)
    style(ax, "Viés por faixa de preço: o modelo puxa para o meio", "Faixa de preço real (quartis das casas de teste)", "Modelo − real, média (US$)", grid="y")
    ax.text(0.99, 0.97, "acima de 0: modelo superestima · abaixo: subestima", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=MUTED)
    save(fig, "eval_error_segments.png")


def main() -> None:
    holdout_figures()
    comparison_figures()
    importance_figure()
    analysis_figures()
    print("Figuras de avaliação regeneradas em", config.FIGURES_DIR)


if __name__ == "__main__":
    main()
