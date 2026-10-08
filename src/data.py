"""Загрузка и проверка данных конкурса (архив hackathonlicence.zip, лицензия CC BY-SA 4.0).

  consumption   — средние безналичные расходы жителя МО за месяц (руб.), 5 категорий + «Все категории»,
                  январь 2023 – декабрь 2024. Публикуются только МО, где оценка проходит порог качества,
                  поэтому пропуски НЕ случайны: выпавшие МО описываются отдельно.
  market_access — индекс доступности рынков 2024 г. (0–1000).
  connection    — расстояния между центрами МО на 31.12.2024 (км): автодороги (каждая пара один раз)
                  и железные дороги (каждая пара записана в ОБЕ стороны с одинаковым расстоянием —
                  вопреки описанию в PDF; здесь пары объединяются). 148 пар разных МО по автодорогам
                  имеют расстояние 0 (общий центр) — заменяются на zero_km.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import get_logger, path

log = get_logger("data")


@dataclass
class Panel:
    """Широкие таблицы: values[category] — DataFrame (territory_id × month)."""
    values: dict[str, pd.DataFrame]
    months: list[str]
    territories: np.ndarray            # МО с полным рядом (базовая выборка)
    coverage: pd.DataFrame             # по всем МО: число месяцев, в выборке ли


def load_consumption(cfg: dict) -> pd.DataFrame:
    df = pd.read_parquet(path(cfg, "raw", cfg["data"]["consumption"]))
    df = df.rename(columns={"consumption": "value"})
    dups = df.duplicated(["territory_id", "date", "category"]).sum()
    if dups:
        raise ValueError(f"Дубли «МО × месяц × категория»: {dups}")
    if (df.value <= 0).any():
        raise ValueError("Неположительные расходы")
    return df


def build_panel(cfg: dict, df: pd.DataFrame | None = None) -> Panel:
    df = load_consumption(cfg) if df is None else df
    cats = cfg["features"]["categories"] + [cfg["features"]["total"]]
    months = sorted(df.date.unique())
    values = {c: df[df.category == c].pivot(index="territory_id", columns="date", values="value")
              .reindex(columns=months) for c in cats}
    n_months = values[cfg["features"]["total"]].notna().sum(axis=1)
    full = n_months[n_months >= cfg["sample"]["months_required"]].index.values
    complete_all = np.all([values[c].loc[full].notna().all(axis=1).values for c in cats], axis=0)
    full = full[complete_all]
    coverage = pd.DataFrame({"n_months": n_months, "in_sample": n_months.index.isin(full)})
    values = {c: v.loc[full] for c, v in values.items()}
    log.info("МО всего: %d, с полным рядом: %d, месяцев: %d", len(coverage), len(full), len(months))
    return Panel(values=values, months=months, territories=full, coverage=coverage)


def excluded_profile(cfg: dict, df: pd.DataFrame, panel: Panel) -> pd.DataFrame:
    """Чем выпавшие (неполные) МО отличаются от выборки: уровень и структура расходов
    по доступным месяцам. Нужен, потому что пропуски не случайны (порог качества оценки)."""
    total = cfg["features"]["total"]
    tot = df[df.category == total].groupby("territory_id").value.median()
    shares = (df[df.category != total].pivot_table(index="territory_id", columns="category", values="value",
                                                    aggfunc="median")
              .div(tot, axis=0))
    prof = shares.assign(total_median=tot, in_sample=lambda x: x.index.isin(panel.territories))
    out = prof.groupby("in_sample").median().T
    out.columns = ["выпавшие" if not c else "в выборке" for c in out.columns]
    out.loc["число МО"] = [int((~prof.in_sample).sum()), int(prof.in_sample.sum())]
    return out


def load_market_access(cfg: dict) -> pd.Series:
    m = pd.read_parquet(path(cfg, "raw", cfg["data"]["market_access"]))
    return m.set_index("territory_id").market_access


def load_distances(cfg: dict, kind: str | None = None) -> pd.DataFrame:
    """Неориентированные пары (a < b) с расстоянием; ж/д дубли (x,y)/(y,x) объединяются."""
    kind = kind or cfg["graphs"]["geography"]["type"]
    n = pd.read_parquet(path(cfg, "raw", cfg["data"]["connection"]))
    n = n[n.type == kind]
    a = n[["territory_id_x", "territory_id_y"]].min(axis=1).values
    b = n[["territory_id_x", "territory_id_y"]].max(axis=1).values
    pairs = pd.DataFrame({"a": a, "b": b, "distance": n.distance.values})
    pairs = pairs[pairs.a != pairs.b]
    spread = pairs.groupby(["a", "b"]).distance.agg(["min", "max"])
    if (spread["max"] - spread["min"]).abs().max() > 1e-6:
        log.warning("%s: прямое и обратное расстояние различаются у части пар — берётся среднее", kind)
    pairs = pairs.groupby(["a", "b"], as_index=False).distance.mean()
    zero = (pairs.distance <= 0).sum()
    if zero:
        pairs.loc[pairs.distance <= 0, "distance"] = cfg["graphs"]["geography"]["zero_km"]
        log.info("%s: нулевых расстояний между разными МО: %d → %.1f км", kind, zero,
                 cfg["graphs"]["geography"]["zero_km"])
    return pairs


def distance_matrix(pairs: pd.DataFrame, ids: np.ndarray) -> np.ndarray:
    """Квадратная матрица расстояний для ids (np.inf — нет связи)."""
    pos = pd.Series(np.arange(len(ids)), index=ids)
    p = pairs[pairs.a.isin(pos.index) & pairs.b.isin(pos.index)]
    D = np.full((len(ids), len(ids)), np.inf)
    i, j = pos[p.a].values, pos[p.b].values
    D[i, j] = D[j, i] = p.distance.values
    np.fill_diagonal(D, 0.0)
    return D


def sample_overlap(cfg: dict, panel: Panel) -> pd.DataFrame:
    """Покрытие дополнительных данных среди МО базовой выборки."""
    full = set(panel.territories)
    ma = set(load_market_access(cfg).index)
    rows = []
    for kind in ("highway", "railway"):
        p = load_distances(cfg, kind)
        covered = set(p.a) | set(p.b)
        rows.append((f"расстояния: {kind}", len(full & covered)))
    rows.insert(0, ("индекс доступности рынков", len(full & ma)))
    hw = load_distances(cfg, "highway")
    rows.append(("доступность рынков и автодороги", len(full & ma & (set(hw.a) | set(hw.b)))))
    return pd.DataFrame(rows, columns=["данные", "МО из базовой выборки"]).assign(всего=len(full))
