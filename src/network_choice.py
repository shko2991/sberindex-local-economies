"""Обоснование выбора сети: что каждое семейство рёбер добавляет к признакам.

1. Пересечение рёбер (Жаккар) — насколько разные понятия близости дают одни и те же связи.
2. Сетевая автокорреляция признаков (I Морана, перестановочный тест): похожи ли соседи по
   сети по составу и уровню расходов. Для графа structure величина завышена по построению
   (граф строится из тех же признаков) — он служит верхней границей, а не аргументом.
3. Выраженность сообществ: взвешенная модулярность разбиения Leiden против той же величины на
   случайных графах с точно теми же степенями (попарные переключения рёбер), 20 повторов.
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


def degree_preserving_null(A: sp.spmatrix, seed: int = 0, swaps_per_edge: int = 10) -> sp.csr_matrix:
    """Случайный граф с ТОЧНО той же последовательностью степеней: попарные переключения рёбер
    (a–b, c–d → a–d, c–b) без петель и кратных рёбер (igraph.Graph.rewire). Веса исходных рёбер
    случайно переставляются по новым рёбрам: распределение весов сохраняется, силы вершин — нет
    (это отдельное условие, его такая нуль-модель не обеспечивает)."""
    import random
    import igraph as ig
    from methods import _to_igraph
    g = _to_igraph(sp.csr_matrix(A))
    ig.set_random_number_generator(random.Random(seed))
    w0 = np.array(g.es["weight"], dtype=float)
    g.rewire(n=swaps_per_edge * g.ecount(), allowed_edge_types="simple")
    E = np.array(g.get_edgelist())
    w = np.random.default_rng(seed).permutation(w0)[:len(E)]
    M = sp.coo_matrix((w, (E[:, 0], E[:, 1])), shape=A.shape)
    return sp.csr_matrix(M + M.T)


def weighted_modularity(A: sp.spmatrix, membership) -> float:
    """Взвешенная модулярность Ньюмана разбиения (тот же критерий, что оптимизирует Leiden с весами)."""
    from icvi import graph_indices
    return graph_indices(A, np.asarray(membership))["MQ"]


def community_strength(graphs: dict[str, sp.spmatrix], resolution: float = 1.0, nulls: int = 20) -> pd.DataFrame:
    """Взвешенная модулярность разбиения Leiden против того же на случайных графах с теми же
    степенями: среднее, стандартное отклонение и z-оценка превышения."""
    import leidenalg as la
    from methods import _to_igraph
    rows = []
    for name, A in graphs.items():
        def q(M, seed):
            g = _to_igraph(sp.csr_matrix(M))
            p = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                                  resolution_parameter=resolution, seed=seed)
            return weighted_modularity(M, p.membership), len(set(p.membership))
        q_obs, k_obs = q(A, 1)
        qn = np.array([q(degree_preserving_null(A, s), s)[0] for s in range(nulls)])
        rows.append({"сеть": name, "модулярность": round(q_obs, 3), "сообществ": k_obs,
                     "модулярность (нуль-модель)": round(float(qn.mean()), 3),
                     "нуль-модель, ст. откл.": round(float(qn.std(ddof=1)), 4),
                     "превышение": round(q_obs - float(qn.mean()), 3),
                     "z": round((q_obs - qn.mean()) / max(qn.std(ddof=1), 1e-9), 1)})
    return pd.DataFrame(rows)
