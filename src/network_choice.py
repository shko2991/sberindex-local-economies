"""Обоснование выбора сети: что каждое семейство рёбер добавляет к признакам.

1. Пересечение рёбер (Жаккар) — насколько разные понятия близости дают одни и те же связи.
2. Сетевая автокорреляция признаков (I Морана, перестановочный тест): похожи ли соседи по
   сети по составу и уровню расходов. Для графа structure величина завышена по построению
   (граф строится из тех же признаков) — он служит верхней границей, а не аргументом.
3. Выраженность сообществ: модулярность Leiden против той же величины на случайном графе
   с теми же степенями (конфигурационная модель, сохранение степеней перестановкой рёбер).
4. Независимость от признаков: сеть, построенная из тех же данных, что и признаки, не даёт
   новой информации (циркулярность). География — внешний источник (расстояния по дорогам
   от организаторов), совместная динамика и DTW — из тех же рядов, но из остатков, а не уровней.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def morans_i(A: sp.spmatrix, x: np.ndarray, perms: int = 999, seed: int = 0) -> tuple[float, float]:
    """I Морана с построчно нормированными весами (сумма весов каждого МО = 1, I ∈ [−1, 1]);
    p — доля перестановок с I ≥ наблюдаемого. Изолированные узлы не участвуют."""
    A = sp.csr_matrix(A, dtype=float)
    deg = np.asarray(A.sum(1)).ravel()
    keep = deg > 0
    A = A[keep][:, keep]
    Wr = sp.diags(1 / deg[keep]) @ A
    z = x[keep] - x[keep].mean()

    def stat(v):
        return float(v @ (Wr @ v)) / float(v @ v)

    obs = stat(z)
    g = np.random.default_rng(seed)
    null = np.array([stat(g.permutation(z)) for _ in range(perms)])
    p = (1 + (null >= obs).sum()) / (perms + 1)
    return obs, p


def moran_table(graphs: dict[str, sp.spmatrix], raw: pd.DataFrame, perms: int = 199) -> pd.DataFrame:
    rows = {}
    for g, A in graphs.items():
        rows[g] = {c: morans_i(A, raw[c].values, perms)[0] for c in raw.columns}
    return pd.DataFrame(rows).round(3)


def degree_preserving_null(A: sp.spmatrix, seed: int = 0) -> sp.csr_matrix:
    """Случайный граф с (почти) той же последовательностью степеней: конфигурационная модель
    со случайным спариванием «полурёбер», петли и кратные рёбра удаляются (erased configuration
    model); веса рёбер берутся случайно из весов исходного графа."""
    A = sp.csr_matrix(A)
    g = np.random.default_rng(seed)
    deg = np.asarray((A > 0).sum(1)).ravel().astype(int)
    stubs = np.repeat(np.arange(len(deg)), deg)
    g.shuffle(stubs)
    if len(stubs) % 2:
        stubs = stubs[:-1]
    a, b = stubs[0::2], stubs[1::2]
    keep = a != b
    a, b = np.minimum(a[keep], b[keep]), np.maximum(a[keep], b[keep])
    pairs = np.unique(np.c_[a, b], axis=0)
    w = g.choice(sp.triu(A, 1).data, size=len(pairs), replace=True)
    M = sp.coo_matrix((w, (pairs[:, 0], pairs[:, 1])), shape=A.shape)
    return sp.csr_matrix(M + M.T)


def community_strength(graphs: dict[str, sp.spmatrix], resolution: float = 1.0, nulls: int = 5) -> pd.DataFrame:
    import leidenalg as la
    from methods import _to_igraph
    rows = []
    for name, A in graphs.items():
        def q(M, seed):
            g = _to_igraph(sp.csr_matrix(M))
            p = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                                  resolution_parameter=resolution, seed=seed)
            return p.modularity, len(set(p.membership))
        q_obs, k_obs = q(A, 1)
        qn = [q(degree_preserving_null(A, s), s)[0] for s in range(nulls)]
        rows.append({"сеть": name, "модулярность": round(q_obs, 3), "сообществ": k_obs,
                     "модулярность (нуль-модель)": round(float(np.mean(qn)), 3),
                     "превышение": round(q_obs - float(np.mean(qn)), 3)})
    return pd.DataFrame(rows)
