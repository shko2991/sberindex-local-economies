"""Статичные рисунки для отчёта (reports/figures). Запуск после pipeline.py: python src/figures.py"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import labor_available, load_config, path  # noqa: E402

FAMILY_COLORS = {"kefrin": "#1f6f78", "leiden": "#c8553d", "spectral": "#8e6c8a", "spectral_joint": "#b08d57",
                 "kmeans": "#5b7553", "gmm": "#7a9e9f", "ward": "#3d405b", "pattern": "#d4a373"}
# те же цвета типов, что на интерактивной странице (светлая тема)
TYPE_COLORS = ["#2f6690", "#4f8a3c", "#c8553d", "#d9a03f", "#7a5ea8", "#12857f", "#9a6a44", "#c06c94",
               "#5c677d", "#a3a83b", "#6d8fc7", "#8c3b4a"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
SAVE = {"bbox_inches": "tight", "pad_inches": 0.15}        # ничего не обрезается по краям

# понятные подписи вместо технических имён
LABELS = {
    "share_Продовольствие": "доля: продовольствие", "share_Здоровье": "доля: здоровье",
    "share_Маркетплейсы": "доля: маркетплейсы", "share_Общественное питание": "доля: общепит",
    "share_Транспорт": "доля: транспорт", "share_Прочее": "доля: прочее",
    "level_total": "уровень расходов", "growth_rel": "относительный рост", "summer_peak": "летний пик",
    "dec_peak": "декабрьский пик", "volatility": "волатильность", "mp_shift": "сдвиг к маркетплейсам",
    "structure": "все признаки МО", "comovement": "синхронная динамика", "leadlag": "форма траектории (DTW)",
    "geography": "дороги", "multiplex": "мультиплекс",
    "доля старше трудоспособного": "доля старше трудоспособного возраста",
    "доля занятых в O–Q (госуправление, образование, здравоохранение)": "доля занятых в O–Q\n(госуправление, образование, здравоохр.)",
    "log доступности рынков": "доступность рынков (log)",
    "log расстояния до центра региона": "расстояние до центра региона (log)",
    "Север (широта ≥ 60°)": "Север, широта ≥ 60° (0/1)",
    "городской округ": "городской округ (0/1)",
    "миграционный прирост на 1000": "миграционный прирост на 1 000",
    "объектов торговли на 1000": "торговых объектов на 1 000 жителей",
    "доля современных форматов": "доля современных форматов магазинов",
}


def wrap(text: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.fill(part, width, break_long_words=False, break_on_hyphens=False)
                     for part in text.split("\n"))


def relabel(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(index=lambda x: LABELS.get(x, x), columns=lambda x: LABELS.get(x, x))


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
    fig.savefig(path(cfg, "figures", "pareto_sw_mq.png"), **SAVE)
    plt.close(fig)


def heat(df: pd.DataFrame, title: str, fname: str, cfg, fmt="{:.2f}", cmap="BuGn", vmin=None, vmax=None,
         col_wrap: int = 14, row_wrap: int = 40, col_width: float = 0.95):
    df = relabel(df)
    cols = [wrap(str(c), col_wrap) for c in df.columns]
    rows = [wrap(str(r), row_wrap) for r in df.index]
    w = max(5.5, 1.6 + col_width * df.shape[1])
    h = 1.3 + 0.36 * df.shape[0] + 0.12 * max(c.count("\n") for c in cols)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(df.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(df.shape[1]), cols, rotation=0, ha="center", fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(df.shape[0]), rows, fontsize=8)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.values[i, j]
            r, g, b, _ = im.cmap(im.norm(v))                       # цвет текста — по яркости клетки
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                    color="white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black")
    ax.set_title(wrap(title, int(w * 11)), loc="left", pad=12 + 9 * max(c.count("\n") + 1 for c in cols))
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(path(cfg, "figures", fname), **SAVE)
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
    fig.savefig(path(cfg, "figures", "types_map.png"), **SAVE)
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
    ax.set_axis_off(); ax.set_title("Расходы жителей относительно фонда оплаты труда на жителя", loc="left", fontsize=10)
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
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    ax.set_axis_off(); ax.set_title("Скопления остатка модели: локальный I Морана, поправка Бенджамини–Хохберга, q < 0,05", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path(cfg, "figures", "mismatch_maps.png"), **SAVE)
    plt.close(fig)


def cons_labor_heat(cfg):
    c = pd.read_csv(path(cfg, "processed", "cons_vs_labor_no_federal.csv"), index_col=0)   # выборка как у V и AMI
    names, lnames = cfg.get("type_names") or {}, cfg.get("labor_type_names") or {}
    n_row = c.sum(axis=1)
    c.index = [f"{names.get(int(i), i)} (N = {int(n_row[i])})" for i in c.index]
    c.columns = [lnames.get(int(float(j)), j) for j in c.columns]
    n_all = f"{int(c.values.sum()):,}".replace(",", " ")
    heat(c.div(c.sum(axis=1), axis=0) * 100,
         f"Типы потребления (строки) × типы рынка труда (столбцы): доля МО строки, %. "
         f"Без городов федерального значения, N = {n_all}; строки с малым N (Москва, Петербург) — ориентировочно",
         "cons_vs_labor.png", cfg, fmt="{:.0f}", vmin=0, vmax=100, col_wrap=11, col_width=1.25, row_wrap=60)


def gap_coef(cfg):
    c = pd.read_csv(path(cfg, "processed", "gap_model.csv"), index_col=0).sort_values("коэф. (станд.)")
    lab = [LABELS.get(i, i) for i in c.index]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    y = np.arange(len(c))
    b = c["коэф. (станд.)"].values
    has_ci = "95% ДИ, нижняя" in c
    for yi, bi, p, lo, hi in zip(y, b, c.p, c.get("95% ДИ, нижняя", c.p * np.nan), c.get("95% ДИ, верхняя", c.p * np.nan)):
        color = "#2f6690" if bi < 0 else "#c8553d"
        if has_ci:
            ax.plot([lo, hi], [yi, yi], color=color, lw=1.6, alpha=0.9 if p < 0.05 else 0.45)
        ax.scatter([bi], [yi], s=34, zorder=3, color=color if p < 0.05 else "white", edgecolors=color, lw=1.4)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y, lab, fontsize=8)
    ax.set_xlabel(wrap("Коэффициент: изменение стандартизованного log-разрыва на 1 SD непрерывного признака; "
                       "для бинарных признаков (0/1) — переход 0 → 1. Линии — 95% доверительные интервалы "
                       "(ошибки кластеризованы по регионам); незакрашенные точки — p ≥ 0,05", 110), fontsize=8)
    n = json.load(open(path(cfg, "processed", "labor_summary.json"), encoding="utf-8"))
    nn = f"{n['gap_model_n']:,}".replace(",", " ")
    r2 = f"{n['gap_model_R2']:.2f}".replace(".", ",")
    ax.set_title(wrap(f"Связи факторов с разрывом расходов и фонда оплаты труда: МНК, {nn} МО без городов "
                      f"федерального значения, R² = {r2}. Это связи, а не причинные эффекты", 100), loc="left", fontsize=9)
    fig.savefig(path(cfg, "figures", "gap_model.png"), **SAVE)
    plt.close(fig)


def main():
    cfg = load_config()
    pareto(cfg)
    df = pd.read_csv(path(cfg, "processed", "candidates.csv"), index_col=0)
    idx = cfg["evaluation"]["icvi"]
    heat(df[df.admissible][idx].corr("spearman"), "Корреляции Спирмена между шестью индексами качества (251 допустимый вариант)",
         "icvi_corr.png", cfg, cmap="RdBu_r", vmin=-1, vmax=1)
    heat(pd.read_csv(path(cfg, "processed", "moran.csv"), index_col=0), "I Морана: насколько соседи по каждой сети похожи по показателю (построчно нормированные веса)",
         "moran.png", cfg, vmin=0, vmax=1)
    heat(pd.read_csv(path(cfg, "processed", "edge_overlap.csv"), index_col=0), "Пересечение множеств рёбер разных сетей (коэффициент Жаккара)",
         "edge_overlap.png", cfg, vmin=0, vmax=1)
    types_map(cfg)
    if labor_available(cfg) and path(cfg, "processed", "labor_mismatch.csv").exists():
        gap_maps(cfg)
        cons_labor_heat(cfg)
        gap_coef(cfg)


if __name__ == "__main__":
    main()
