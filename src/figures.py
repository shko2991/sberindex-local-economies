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


def _polys(cfg, col_df):
    from reference import load_polygons
    return load_polygons(cfg, simplify_m=3000).join(col_df, how="left").to_crs(
        "+proj=aea +lat_1=52 +lat_2=64 +lon_0=100 +datum=WGS84")


def gap_maps(cfg):
    m = pd.read_csv(path(cfg, "processed", "labor_mismatch.csv"), index_col=0)
    g = _polys(cfg, m[["gap", "lisa", "federal_city"]])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    ax = axes[0]
    g[g.gap.isna()].plot(ax=ax, color="#e3e4df", lw=0)
    g[g.gap.notna()].plot(ax=ax, column="gap", cmap="RdBu_r", vmin=-1.2, vmax=1.2, lw=0, legend=True,
                          legend_kwds={"shrink": 0.6, "label": "log(расходы / фонд оплаты труда), к медиане"})
    ax.set_axis_off(); ax.set_title("Расходы жителей относительно фонда оплаты труда на жителя", loc="left")
    ax = axes[1]
    colors = {"HH": "#c8553d", "LL": "#2f6690", "HL": "#e9a38f", "LH": "#8fb3d9", "не значимо": "#d7d9d2",
              "нет данных": "#eeeeea"}
    g["lisa"] = g.lisa.fillna("нет данных")
    for k, c in colors.items():
        sub = g[g.lisa == k]
        if len(sub):
            sub.plot(ax=ax, color=c, lw=0, label={"HH": "тратят больше, чем объясняет рынок труда (скопление)",
                                                   "LL": "тратят меньше (скопление)", "HL": "выше среди низких",
                                                   "LH": "ниже среди высоких"}.get(k, k))
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_axis_off(); ax.set_title("Скопления остатка: локальный I Морана, поправка Бенджамини–Хохберга, q < 0,05", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path(cfg, "figures", "mismatch_maps.png"))
    plt.close(fig)


def cons_labor_heat(cfg):
    c = pd.read_csv(path(cfg, "processed", "cons_vs_labor.csv"), index_col=0)
    names, lnames = cfg.get("type_names") or {}, cfg.get("labor_type_names") or {}
    c.index = [names.get(int(i), i) for i in c.index]
    c.columns = [lnames.get(int(float(j)), j) for j in c.columns]
    heat(c.div(c.sum(axis=1), axis=0) * 100, "Типы потребления (строки) × типы рынка труда (столбцы), % строки",
         "cons_vs_labor.png", cfg, fmt="{:.0f}", vmin=0, vmax=100)


def gap_coef(cfg):
    c = pd.read_csv(path(cfg, "processed", "gap_model.csv"), index_col=0).sort_values("коэф. (станд.)")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    from matplotlib.colors import to_rgba
    col = [to_rgba("#2f6690" if v < 0 else "#c8553d", 1.0 if p < 0.05 else 0.35)
           for v, p in zip(c["коэф. (станд.)"], c.p)]
    ax.barh(c.index, c["коэф. (станд.)"], color=col)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("стандартизованный коэффициент (бледные — p ≥ 0,05)")
    ax.set_title("Что объясняет разрыв расходов и фонда оплаты труда", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path(cfg, "figures", "gap_model.png"))
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
    if path(cfg, "processed", "labor_mismatch.csv").exists():
        gap_maps(cfg)
        cons_labor_heat(cfg)
        gap_coef(cfg)


if __name__ == "__main__":
    main()
