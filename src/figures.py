"""Статичные рисунки для отчёта (reports/figures). Запуск после pipeline.py: python src/figures.py"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import load_config, path  # noqa: E402

FAMILY_COLORS = {"kefrin": "#1f6f78", "leiden": "#c8553d", "spectral": "#8e6c8a", "spectral_joint": "#b08d57",
                 "kmeans": "#5b7553", "gmm": "#7a9e9f", "ward": "#3d405b", "pattern": "#d4a373"}
# те же цвета типов, что на интерактивной странице (светлая тема)
TYPE_COLORS = ["#2f6690", "#4f8a3c", "#c8553d", "#d9a03f", "#7a5ea8", "#12857f", "#9a6a44", "#c06c94",
               "#5c677d", "#a3a83b", "#6d8fc7", "#8c3b4a"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


def pareto(cfg):
    df = pd.read_csv(path(cfg, "processed", "candidates.csv"), index_col=0)
    df = df[df.admissible]
    chosen = json.load(open(path(cfg, "processed", "summary.json"), encoding="utf-8"))["chosen"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m, g in df.groupby("method"):
        ax.scatter(g.SW, g.MQ, s=16, alpha=0.75, color=FAMILY_COLORS.get(m, "grey"), label=m, lw=0)
    c = df.loc[chosen]
    ax.scatter([c.SW], [c.MQ], s=120, facecolors="none", edgecolors="black", lw=1.4)
    ax.annotate("итоговое разбиение", (c.SW, c.MQ), xytext=(8, 8), textcoords="offset points")
    ax.set_xlabel("силуэт в пространстве признаков (SW) →")
    ax.set_ylabel("модулярность на географической сети (MQ) →")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.set_title("Компромисс: сходство потребления против связности сети", loc="left")
    fig.tight_layout()
    fig.savefig(path(cfg, "figures", "pareto_sw_mq.png"))
    plt.close(fig)


def heat(df: pd.DataFrame, title: str, fname: str, cfg, fmt="{:.2f}", cmap="BuGn", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(0.9 + 0.75 * df.shape[1], 0.6 + 0.32 * df.shape[0]))
    im = ax.imshow(df.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(df.shape[1]), df.columns, rotation=30, ha="right")
    ax.set_yticks(range(df.shape[0]), df.index)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.values[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                    color="white" if (vmax is not None and v > (vmin + vmax) / 2 * 1.2) else "black")
    ax.set_title(title, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path(cfg, "figures", fname))
    plt.close(fig)


def types_map(cfg):
    from reference import load_polygons
    final = pd.read_csv(path(cfg, "processed", "final_types.csv"), index_col=0)
    # столбец не называть "type": у GeoDataFrame это свойство (тип геометрии)
    g = load_polygons(cfg, simplify_m=3000).join(final[["type"]].rename(columns={"type": "k"}), how="left").to_crs(
        "+proj=aea +lat_1=52 +lat_2=64 +lon_0=100 +datum=WGS84")
    names = cfg.get("type_names") or {}
    fig, ax = plt.subplots(figsize=(10, 5.6))
    g[g["k"].isna()].plot(ax=ax, color="#e3e4df", lw=0)
    for t, sub in g[g["k"].notna()].groupby("k"):
        sub.plot(ax=ax, color=TYPE_COLORS[int(t) % len(TYPE_COLORS)], lw=0,
                 label=f"{int(t)} · {names.get(int(t), '')}")
    ax.set_axis_off()
    ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
    ax.set_title("Типы потребления МО (серым — МО вне выборки)", loc="left")
    fig.tight_layout()
    fig.savefig(path(cfg, "figures", "types_map.png"))
    plt.close(fig)


def main():
    cfg = load_config()
    pareto(cfg)
    df = pd.read_csv(path(cfg, "processed", "candidates.csv"), index_col=0)
    idx = cfg["evaluation"]["icvi"]
    heat(df[df.admissible][idx].corr("spearman"), "Корреляции Спирмена между ICVI (допустимые кандидаты)",
         "icvi_corr.png", cfg, cmap="RdBu_r", vmin=-1, vmax=1)
    heat(pd.read_csv(path(cfg, "processed", "moran.csv"), index_col=0), "I Морана: похожи ли соседи по сети",
         "moran.png", cfg, vmin=0, vmax=1)
    heat(pd.read_csv(path(cfg, "processed", "edge_overlap.csv"), index_col=0), "Пересечение рёбер (Жаккар)",
         "edge_overlap.png", cfg, vmin=0, vmax=1)
    types_map(cfg)


if __name__ == "__main__":
    main()
