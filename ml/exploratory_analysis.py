"""Fase 2 — Análise exploratória (PRD §8.2) e triagem de atributos do formulário.

Uso: python -m ml.exploratory_analysis

- Estatísticas descritivas e gráficos usam o train.csv completo (finalidade
  descritiva/dashboard; nenhum parâmetro de modelo é aprendido aqui).
- A triagem dos atributos candidatos ao formulário usa SOMENTE a partição de
  desenvolvimento (80%), para que o holdout não influencie nenhuma decisão.

Saídas: artifacts/figures/eda_*.png e artifacts/reports/eda_summary.json
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml import config
from ml.data import load_train
from ml.units import SQFT_TO_M2
from ml.plot_style import BAD, INK, MUTED, NEUTRAL, NEUTRAL_SOFT, REAL, save as _save, style, usd_k
from ml.preprocessing import NOMINAL_NUMERIC_CODES, StructuralNAFiller, dev_holdout_split

# Candidatos compreensíveis para um usuário leigo preencher (PRD §7.5).
APP_CANDIDATES = [
    "OverallQual", "GrLivArea", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GarageCars", "GarageArea",
    "YearBuilt", "YearRemodAdd", "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd",
    "LotArea", "Fireplaces", "Neighborhood", "KitchenQual", "ExterQual", "BsmtQual",
    "CentralAir", "OverallCond", "HouseStyle", "BldgType",
]


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """eta² — proporção da variância de `values` explicada pelos grupos."""
    df = pd.DataFrame({"c": categories, "v": values}).dropna()
    grand = df["v"].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for _, g in df.groupby("c")["v"])
    ss_total = ((df["v"] - grand) ** 2).sum()
    return float(ss_between / ss_total) if ss_total else 0.0


def main() -> None:
    train = load_train()
    y = train[config.TARGET]
    filled = StructuralNAFiller().fit_transform(train.drop(columns=[config.TARGET]))
    summary: dict = {"source": "train.csv (1.ª parte: descritiva, base rotulada completa)", "n": int(len(train))}

    # 1) Distribuição do alvo
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(y, bins=40, color=REAL, edgecolor="white", linewidth=1)
    axes[0].xaxis.set_major_formatter(usd_k)
    style(axes[0], "Preço de venda: assimétrico, cauda longa à direita", "Preço de venda (US$)", "Casas", grid="y")
    axes[1].hist(np.log1p(y), bins=40, color=REAL, edgecolor="white", linewidth=1)
    style(axes[1], "Log do preço: quase simétrico (o modelo aprende este)", "log(1 + preço)", "Casas", grid="y")
    _save(fig, "eda_target_distribution.png")
    counts, edges = np.histogram(y, bins=40)
    summary["target_histogram"] = {"counts": counts.tolist(), "edges": edges.round(2).tolist()}
    lcounts, ledges = np.histogram(np.log1p(y), bins=40)
    summary["target_log_histogram"] = {"counts": lcounts.tolist(), "edges": ledges.round(4).tolist()}
    summary["target_stats"] = {
        "count": int(y.count()), "min": float(y.min()), "max": float(y.max()), "mean": float(y.mean()),
        "median": float(y.median()), "std": float(y.std()), "q1": float(y.quantile(0.25)), "q3": float(y.quantile(0.75)),
        "skew": float(y.skew()), "log1p_skew": float(np.log1p(y).skew()),
    }

    # 2) Correlação (Spearman) das numéricas com o preço — descritiva
    num = [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c]) and c not in (config.ID_COLUMN, config.TARGET, *NOMINAL_NUMERIC_CODES)]
    corr = train[num].corrwith(y, method="spearman").dropna().sort_values(key=np.abs, ascending=False)
    top = corr.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top.index, top.values, color=[NEUTRAL if v >= 0 else NEUTRAL_SOFT for v in top.values], height=0.6)
    style(ax, "Características numéricas mais associadas ao preço", "Correlação de Spearman com o preço (ρ)", "", grid="x")
    _save(fig, "eda_spearman_top15.png")
    summary["spearman_with_target"] = [{"feature": k, "rho": round(float(v), 4)} for k, v in corr.items()]

    # 3) Área habitável × preço
    fig, ax = plt.subplots(figsize=(7.5, 5))
    big = train["GrLivArea"] > 4000
    m2 = train["GrLivArea"] * SQFT_TO_M2  # dados em pés²; figura em m²
    ax.scatter(m2[~big], y[~big], s=10, alpha=0.45, color=REAL, linewidth=0, label="Casas vendidas")
    ax.scatter(m2[big], y[big], s=46, facecolor="none", edgecolor=BAD, linewidth=1.6, label="Atípicas: mais de 372 m² (4.000 pés²)")
    ax.yaxis.set_major_formatter(usd_k)
    style(ax, "Área construída × preço de venda", "Área construída acima do solo (m²)", "Preço de venda (US$)")
    ax.legend(frameon=False)
    _save(fig, "eda_grlivarea_vs_price.png")
    summary["scatter_grlivarea"] = {
        "x": train["GrLivArea"].tolist(), "y": y.tolist(), "id": train[config.ID_COLUMN].tolist(),
        "overall_qual": train["OverallQual"].tolist(),
    }

    # 4) Qualidade geral × preço
    groups = [y[train["OverallQual"] == q] for q in sorted(train["OverallQual"].unique())]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(groups, tick_labels=sorted(train["OverallQual"].unique()), patch_artist=True,
               boxprops={"facecolor": "#ffeae0", "edgecolor": REAL}, medianprops={"color": REAL, "linewidth": 2},
               whiskerprops={"color": MUTED}, capprops={"color": MUTED}, flierprops={"markersize": 3, "markeredgecolor": MUTED})
    ax.yaxis.set_major_formatter(usd_k)
    style(ax, "Preço por qualidade geral do acabamento", "Qualidade geral (nota 1 a 10)", "Preço de venda (US$)", grid="y")
    _save(fig, "eda_overallqual_box.png")
    summary["by_overall_qual"] = [
        {"value": int(q), "n": int(g.size), "median": float(g.median()), "q1": float(g.quantile(0.25)), "q3": float(g.quantile(0.75)),
         "min": float(g.min()), "max": float(g.max())}
        for q, g in y.groupby(train["OverallQual"])
    ]

    # 5) Bairros
    nb = y.groupby(train["Neighborhood"]).agg(["count", "median", "mean"]).sort_values("median")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(nb.index, nb["median"], color=REAL, height=0.65)
    ax.xaxis.set_major_formatter(usd_k)
    style(ax, "Preço mediano de venda por bairro", "Mediana (US$)", "", grid="x")
    _save(fig, "eda_neighborhood_median.png")
    summary["by_neighborhood"] = [{"value": k, "n": int(r["count"]), "median": float(r["median"]), "mean": round(float(r["mean"]), 2)} for k, r in nb.iterrows()]

    # 6) Ano de construção
    by_decade = y.groupby((train["YearBuilt"] // 10) * 10).agg(["count", "median"])
    summary["by_decade_built"] = [{"decade": int(k), "n": int(r["count"]), "median": float(r["median"])} for k, r in by_decade.iterrows()]
    summary["by_year_sold"] = [{"year": int(k), "n": int(g.size), "median": float(g.median())} for k, g in y.groupby(train["YrSold"])]

    # 7) Ausências (após política de leitura) por coluna — descritiva
    miss = train.isna().mean().mul(100).round(2)
    miss = miss[miss > 0].sort_values(ascending=False)
    miss_after = filled.isna().mean().mul(100).round(2)
    summary["missing_pct"] = [{"column": c, "raw_pct": float(v), "after_structural_fill_pct": float(miss_after.get(c, 0.0))} for c, v in miss.items()]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    idx = miss.index[::-1]
    ax.barh(idx, miss[idx], color=NEUTRAL_SOFT, height=0.6, label="vazio no arquivo (ex.: \"NA\" = sem piscina)")
    ax.barh(idx, miss_after.reindex(idx).fillna(0), color=INK, height=0.6, label="ausência real (depois da regra estrutural)")
    style(ax, "Campos vazios: quase todos significam \"não tem\"", "% das casas", "", grid="x")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "eda_missing.png")

    # 8) Triagem dos candidatos ao formulário — SOMENTE partição de desenvolvimento
    dev, _ = dev_holdout_split(train)
    ly = np.log1p(dev[config.TARGET])
    screening = []
    for c in APP_CANDIDATES:
        s = dev[c]
        missing = float(s.isna().mean() * 100)
        if pd.api.types.is_numeric_dtype(s) and c not in NOMINAL_NUMERIC_CODES:
            rho = float(s.corr(ly, method="spearman"))
            screening.append({"feature": c, "kind": "numérica", "association": round(rho, 4), "measure": "ρ Spearman com log1p(SalePrice)", "abs_strength": round(abs(rho), 4), "missing_pct": round(missing, 2)})
        else:
            eta2 = correlation_ratio(StructuralNAFiller().fit_transform(dev[[c]])[c], ly)
            screening.append({"feature": c, "kind": "categórica", "association": round(eta2, 4), "measure": "η² com log1p(SalePrice)", "abs_strength": round(float(np.sqrt(eta2)), 4), "missing_pct": round(missing, 2), "n_categories": int(s.nunique())})
    screening.sort(key=lambda r: r["abs_strength"], reverse=True)
    summary["app_candidate_screening"] = {"partition": "desenvolvimento (80%)", "n": int(len(dev)), "rows": screening,
                                          "note": "abs_strength = |ρ| para numéricas e √η² (razão de correlação) para categóricas; medidas univariadas, não causais."}

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.EDA_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    print("Figuras em artifacts/figures/, resumo em artifacts/reports/eda_summary.json")
    print("Triagem (partição de desenvolvimento):")
    for r in screening:
        print(f"  {r['feature']:<14} {r['kind']:<11} força={r['abs_strength']:.3f}  ausentes={r['missing_pct']:.1f}%")


if __name__ == "__main__":
    main()
