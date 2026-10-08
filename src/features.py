"""Признаки узлов (МО). Три блока, каждый со своей экономической логикой:

  structure — состав расходов: доли 5 категорий + «прочее» (= 1 − сумма пяти).
              Доли — композиционные данные (сумма = 1), поэтому для евклидовых методов они
              переводятся в центрированные лог-отношения (CLR, Aitchison 1986).
              «Все категории» — знаменатель, а не шестая доля.
  level     — относительный уровень: лог-разность расходов МО и медианы всех МО в том же месяце.
              Убирает общий рост цен (~15% за 2024 г.) и декабрьский пик (~18%), но сохраняет
              различия в уровне жизни.
  dynamics  — отличия траектории МО от типичной: относительный рост 2024/2023, летний и
              декабрьский пики сверх общего, волатильность, сдвиг к маркетплейсам.

Очищенные ряды (residual) — те же лог-разности с медианой по месяцам: на них считаются
корреляции и DTW, иначе общий тренд и сезонность делают похожими почти все ряды.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data import Panel


@dataclass
class Features:
    X: pd.DataFrame                 # итоговая матрица признаков (стандартизована, блоки уравновешены)
    raw: pd.DataFrame               # те же признаки в исходных единицах (для интерпретации)
    blocks: dict[str, list[str]]    # блок → список колонок
    residual: dict[str, pd.DataFrame]   # категория → относительная траектория: лог-ряд минус медиана по МО в том же месяце
    shares: dict[str, pd.DataFrame]     # категория/прочее → доли по месяцам


def monthly_shares(panel: Panel, cfg: dict) -> dict[str, pd.DataFrame]:
    total = panel.values[cfg["features"]["total"]]
    sh = {c: panel.values[c] / total for c in cfg["features"]["categories"]}
    sh[cfg["features"]["other_label"]] = 1 - sum(sh.values())
    return sh


def residual_series(panel: Panel) -> dict[str, pd.DataFrame]:
    """log(расходы МО) − медиана log(расходов) всех МО в том же месяце, по каждой категории."""
    out = {}
    for c, v in panel.values.items():
        lv = np.log(v)
        out[c] = lv.sub(lv.median(axis=0), axis=1)
    return out


def clr(shares: pd.DataFrame) -> pd.DataFrame:
    lg = np.log(shares.clip(lower=1e-6))
    return lg.sub(lg.mean(axis=1), axis=0)


def _months(cols, months_of_year):
    return [c for c in cols if int(c[5:7]) in months_of_year]


def static_features(panel: Panel, cfg: dict) -> Features:
    shares = monthly_shares(panel, cfg)
    resid = residual_series(panel)
    total = cfg["features"]["total"]
    cols = panel.months
    y23 = [c for c in cols if c.startswith("2023")]
    y24 = [c for c in cols if c.startswith("2024")]

    mean_sh = pd.DataFrame({c: s.mean(axis=1) for c, s in shares.items()})
    struct = clr(mean_sh).add_prefix("clr_")
    rt = resid[total]
    level = pd.DataFrame({"level_total": rt.mean(axis=1)})

    def detrended_std(row):
        t = np.arange(len(row))
        b = np.polyfit(t, row.values, 1)
        return np.std(row.values - np.polyval(b, t))

    summer = _months(cols, {6, 7, 8})
    dec = _months(cols, {12})
    octnov = _months(cols, {10, 11})
    mp = shares["Маркетплейсы"]
    dyn = pd.DataFrame({
        "growth_rel": rt[y24].mean(axis=1) - rt[y23].mean(axis=1),
        "summer_peak": rt[summer].mean(axis=1) - rt.mean(axis=1),
        "dec_peak": rt[dec].mean(axis=1) - rt[octnov].mean(axis=1),
        "volatility": rt.apply(detrended_std, axis=1),
        "mp_shift": mp[y24].mean(axis=1) - mp[y23].mean(axis=1),
    })
    raw = pd.concat([mean_sh.add_prefix("share_"), level, dyn], axis=1)
    blocks = {"structure": list(struct.columns), "level": list(level.columns), "dynamics": list(dyn.columns)}
    use = [b for b in cfg["features"]["blocks"]]
    X = standardize_blocks(pd.concat([struct, level, dyn], axis=1), {b: blocks[b] for b in use})
    return Features(X=X, raw=raw, blocks={b: blocks[b] for b in use}, residual=resid, shares=shares)


def standardize_blocks(F: pd.DataFrame, blocks: dict[str, list[str]]) -> pd.DataFrame:
    """z-оценки; каждый блок делится на √(число признаков), чтобы блоки весили одинаково
    (иначе структура из 6 признаков задавила бы уровень из одного)."""
    parts = []
    for b, cols in blocks.items():
        z = (F[cols] - F[cols].mean()) / F[cols].std(ddof=0).replace(0, 1)
        parts.append(z / np.sqrt(len(cols)))
    return pd.concat(parts, axis=1)


def period_features(panel: Panel, cfg: dict, period: str | None = None) -> dict[str, pd.DataFrame]:
    """Признаки узлов по периодам (квартал): структура (CLR долей) и относительный уровень.
    Из CLR каждого периода вычитается медиана по МО того же периода: общая сезонность
    (летний рост транспорта, декабрьский — маркетплейсов) иначе «перебрасывала» бы МО между
    типами каждый квартал. Уровень уже задан относительно медианы месяца.
    Динамика внутри квартала из 3 точек ненадёжна, поэтому в периодные признаки не входит."""
    period = period or cfg["features"]["period"]
    shares = monthly_shares(panel, cfg)
    resid = residual_series(panel)
    total = cfg["features"]["total"]
    per = pd.PeriodIndex(pd.to_datetime(panel.months), freq=period).astype(str)
    out = {}
    for p in sorted(set(per)):
        m = [c for c, q in zip(panel.months, per) if q == p]
        sh = pd.DataFrame({c: s[m].mean(axis=1) for c, s in shares.items()})
        C = clr(sh)
        C = C - C.median(axis=0)
        F = pd.concat([C.add_prefix("clr_"), pd.DataFrame({"level_total": resid[total][m].mean(axis=1)})], axis=1)
        out[p] = F
    # общая стандартизация по всем периодам: признаки сопоставимы во времени
    allF = pd.concat(out.values())
    mu, sd = allF.mean(), allF.std(ddof=0).replace(0, 1)
    blocks = {"structure": [c for c in allF.columns if c.startswith("clr_")], "level": ["level_total"]}
    res = {}
    for p, F in out.items():
        z = (F - mu) / sd
        res[p] = pd.concat([z[cols] / np.sqrt(len(cols)) for cols in blocks.values()], axis=1)
    return res


def comovement_matrix(feat: Features, months: list[str] | None = None) -> np.ndarray:
    """Строки — МО: относительные траектории всех категорий, у каждого ряда вычтено среднее МО
    (сравнивается движение, а не уровень), затем строка стандартизуется.
    months — подпериод (для динамической сети по годам)."""
    res = {c: (r[months] if months else r) for c, r in feat.residual.items()}
    parts = [r.sub(r.mean(axis=1), axis=0).values for r in res.values()]
    M = np.concatenate(parts, axis=1)
    M = (M - M.mean(axis=1, keepdims=True)) / M.std(axis=1, keepdims=True).clip(min=1e-12)
    return M
