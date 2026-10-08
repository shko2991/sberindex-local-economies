"""Внешняя проверка типов: переменные, НЕ участвовавшие в кластеризации.

  непрерывные  — η² (доля межгрупповой дисперсии) и критерий Краскела–Уоллиса;
  категориальные — V Крамера и χ²;
  согласие двух разбиений — ARI, AMI (скорректированная взаимная информация), V Крамера.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def eta_squared(y: pd.Series, g: pd.Series) -> tuple[float, float]:
    d = pd.DataFrame({"y": y, "g": g}).dropna()
    grand = d.y.mean()
    ss_b = d.groupby("g").y.apply(lambda s: len(s) * (s.mean() - grand) ** 2).sum()
    ss_t = ((d.y - grand) ** 2).sum()
    kw = stats.kruskal(*[s.values for _, s in d.groupby("g").y])
    return float(ss_b / ss_t), float(kw.pvalue)


def cramers_v(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    t = pd.crosstab(a, b)
    chi2, p, _, _ = stats.chi2_contingency(t)
    n = t.values.sum()
    r, k = t.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1)))), float(p)


def partition_agreement(a: pd.Series, b: pd.Series) -> dict:
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    v, p = cramers_v(d.a, d.b)
    return {"n": len(d), "ARI": adjusted_rand_score(d.a, d.b), "AMI": adjusted_mutual_info_score(d.a, d.b),
            "V_Cramer": v, "p_chi2": p}


def external_table(labels: pd.Series, numeric: pd.DataFrame, categorical: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in numeric.columns:
        e, p = eta_squared(numeric[c].reindex(labels.index), labels)
        rows.append({"переменная": c, "мера": "η²", "значение": round(e, 3), "p": p})
    for c in categorical.columns:
        v, p = cramers_v(labels, categorical[c].reindex(labels.index))
        rows.append({"переменная": c, "мера": "V Крамера", "значение": round(v, 3), "p": p})
    return pd.DataFrame(rows)
