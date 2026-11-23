"""Estilo único das figuras estáticas (matplotlib), alinhado à interface web.

Convenção (a mesma de frontend/assets/css/style.css, convertida de OKLCH para hex):
- laranja  = preço REAL da venda
- azul     = estimativa do MODELO
- verde / âmbar / vermelho = situação do erro (até 10% / 10–20% / mais de 20%)
- cinza-azulado neutro = todo o resto (distribuições, barras sem papel de real × modelo)
- verde-azulado = destaque (modelo escolhido)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from ml import config

REAL = "#e56731"
MODEL = "#2d74ca"
ACCENT = "#00766e"
NEUTRAL = "#9aa7ad"
NEUTRAL_SOFT = "#cdd6db"
GOOD, WARN, BAD = "#10703a", "#b07a1c", "#c13c36"
INK, MUTED, GRID = "#171f23", "#464f53", "#e3e8ea"

# Nomes em linguagem simples (iguais aos de frontend/assets/js/util.js → algoName).
_ALGO = [
    ("HistGradientBoostingRegressor (nativo, monotônico)", "Gradient Boosting com regras de bom senso"),
    ("HistGradientBoostingRegressor (categorias nativas)", "Gradient Boosting (bairro como categoria)"),
    ("HistGradientBoostingRegressor", "Gradient Boosting"),
    ("RandomForestRegressor", "Random Forest"),
    ("DummyRegressor", "Chute da mediana"),
    ("Ridge", "Regressão linear (Ridge)"),
]


def algo_name(cfg: str) -> str:
    log = "log1p" in cfg
    name = cfg.replace(" + log1p(y)", "")
    for old, new in _ALGO:
        if old in name:
            name = name.replace(old, new)
            break
    if log and not name.startswith("Gradient Boosting com") and "(bairro" not in name:
        name += " · preço em log"
    return name


def status_color(pct_error: float) -> str:
    a = abs(pct_error)
    return GOOD if a <= 10 else WARN if a <= 20 else BAD


usd_k = FuncFormatter(lambda v, _: f"{v / 1000:,.0f} mil".replace(",", ".").replace("-", "−"))  # eixo rotulado em US$


def style(ax, title: str, xlabel: str = "", ylabel: str = "", grid: str = "both") -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=12.5, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9.5, length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    if grid != "none":
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=150, facecolor="white")
    plt.close(fig)
