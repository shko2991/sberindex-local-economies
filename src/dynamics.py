"""Динамика типов: квартальная атрибутированная сеть (признаки меняются, географическая сеть
постоянна; сеть совместной динамики строится отдельно по 2023 и 2024 гг.).

Основной способ — фиксированные прототипы типов. Центры типов в пространстве квартальных
признаков вычисляются по итоговому (статическому) разбиению на всех МО-кварталах; каждый
МО-квартал относится к ближайшему прототипу по тому же критерию, что и итоговый метод
(для KEFRiN: признаки квартала + строка сети; сетевой член постоянен во времени и работает
как сглаживание). Переходы между типами тогда сравнимы во времени.

Проверка — независимая кластеризация каждого квартала и сопоставление меток венгерским
алгоритмом по коэффициенту Жаккара (Kuhn, 1955); доля совпадений с прототипным способом.

Классы траекторий МО за 8 кварталов:
  стабильные     — тип не менялся;
  устойчивый сдвиг — ровно одна смена типа, новый тип держится ≥ min_run кварталов до конца;
  колебания      — остальные (возвраты, «мигание» на границе типов).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from methods import _row_normalize


def prototypes(F: dict[str, pd.DataFrame], labels: pd.Series) -> pd.DataFrame:
    pooled = pd.concat(F.values())
    lab = pd.concat([labels.loc[f.index] for f in F.values()])
    return pooled.groupby(lab.values).mean()


def assign_periods(F: dict[str, pd.DataFrame], labels: pd.Series, P: np.ndarray | None = None,
                   xi: float = 0.0, distance: str = "cosine") -> pd.DataFrame:
    """Метки МО × период. P — матрица связей (строки в порядке labels.index), xi — её вес."""
    C = prototypes(F, labels)
    ks = C.index.values
    net = 0.0
    if P is not None and xi > 0:
        lab = labels.values
        if distance == "cosine":
            Pn = _row_normalize(P)
            L = _row_normalize(np.stack([Pn[lab == k].mean(0) for k in ks]))
            net = xi * (1 - Pn @ L.T)
        else:
            L = np.stack([P[lab == k].mean(0) for k in ks])
            net = xi * ((P[:, None, :] - L[None]) ** 2).sum(-1)
    out = {}
    for p, f in F.items():
        Y = f.loc[labels.index].values
        if distance == "cosine":
            d = 1 - _row_normalize(Y) @ _row_normalize(C.values).T
        else:
            d = ((Y[:, None, :] - C.values[None]) ** 2).sum(-1)
        out[p] = ks[(d + net).argmin(1)]
    return pd.DataFrame(out, index=labels.index)


def jaccard_match(ref: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Перенумеровать lab так, чтобы максимизировать сумму Жаккара с ref (венгерский алгоритм)."""
    a, b = np.unique(ref), np.unique(lab)
    J = np.zeros((len(b), len(a)))
    for i, y in enumerate(b):
        my = lab == y
        for j, x in enumerate(a):
            mx = ref == x
            J[i, j] = (my & mx).sum() / max((my | mx).sum(), 1)
    r, c = linear_sum_assignment(-J)
    m = {b[i]: a[j] for i, j in zip(r, c)}
    nxt = max(a.max(), b.max()) + 1
    for y in b:
        if y not in m:
            m[y] = nxt; nxt += 1
    return np.array([m[y] for y in lab])


def trajectory_classes(T: pd.DataFrame, min_run: int = 2) -> pd.Series:
    V = T.values
    sw = (V[:, 1:] != V[:, :-1]).sum(1)
    cls = np.where(sw == 0, "стабильный", "колебания").astype(object)
    for i in np.where(sw == 1)[0]:
        t = int(np.argmax(V[i, 1:] != V[i, :-1])) + 1
        if V.shape[1] - t >= min_run:
            cls[i] = "устойчивый сдвиг"
    return pd.Series(cls, index=T.index, name="trajectory")


def transition_matrix(T: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    return pd.crosstab(T[a], T[b], rownames=[a], colnames=[b])


def _mode_last(values) -> int:
    """Самый частый тип; при равенстве частот — тот из них, что встретился последним по времени."""
    vals = list(values)
    cnt = pd.Series(vals).value_counts()
    tied = set(cnt[cnt == cnt.max()].index)
    return next(v for v in reversed(vals) if v in tied)


def yearly_mode(T: pd.DataFrame) -> pd.DataFrame:
    """Тип МО за год — наиболее частый тип по кварталам года (при равенстве — последний по времени)."""
    out = {}
    for y in sorted({c[:4] for c in T.columns}):
        cols = sorted(c for c in T.columns if c.startswith(y))
        out[y] = T[cols].apply(lambda r: _mode_last(r.values), axis=1)
    return pd.DataFrame(out)
