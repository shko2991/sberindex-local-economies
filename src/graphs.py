"""Рёбра сети: четыре семейства, у каждого своя экономическая логика «близости» МО.

  structure  — похожий состав и уровень расходов (косинусная близость признаков узлов);
  comovement — синхронная динамика: корреляция ОЧИЩЕННЫХ рядов (без общего тренда и сезонности);
  leadlag    — похожая форма траектории с допуском на сдвиг: DTW по очищенному ряду общего
               уровня (окно Сакое–Чибы); отдельно — лаговые корреляции (кто кого опережает);
  geography  — физическая близость: расстояние по автодорогам, ядро exp(−d/σ).
Плотная матрица близости разрежается до k ближайших соседей и симметризуется (максимум весов).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from data import distance_matrix, load_distances
from features import Features, comovement_matrix


def knn_sparsify(S: np.ndarray, k: int, mutual: bool = False) -> sp.csr_matrix:
    """S — плотная матрица близости (чем больше, тем ближе). Остаются k ближайших у каждого узла."""
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    n = S.shape[0]
    idx = np.argpartition(-S, kth=min(k, n - 1) - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    w = S[rows, cols]
    keep = np.isfinite(w) & (w > 0)
    A = sp.csr_matrix((w[keep], (rows[keep], cols[keep])), shape=(n, n))
    if mutual:
        B = A.minimum(A.T)
        return sp.csr_matrix(B)
    return sp.csr_matrix(A.maximum(A.T))


def structure_similarity(feat: Features) -> np.ndarray:
    X = feat.X.values
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
    return (1 + Xn @ Xn.T) / 2            # косинус → [0, 1]


def comovement_similarity(feat: Features, months: list[str] | None = None) -> np.ndarray:
    M = comovement_matrix(feat, months)
    C = M @ M.T / M.shape[1]
    return np.clip(C, 0, None)            # противофазные ряды близкими не считаются


def yearly_comovement_graphs(cfg: dict, feat: Features) -> dict[str, sp.csr_matrix]:
    """Динамическая сеть: граф совместной динамики отдельно по каждому году."""
    months = list(next(iter(feat.residual.values())).columns)
    years = sorted({m[:4] for m in months})
    return {y: knn_sparsify(comovement_similarity(feat, [m for m in months if m.startswith(y)]),
                            cfg["graphs"]["knn"], cfg["graphs"].get("mutual", False)) for y in years}


def dtw_similarity(feat: Features, total: str, window: int) -> np.ndarray:
    from dtaidistance import dtw
    r = feat.residual[total]
    Z = r.sub(r.mean(axis=1), axis=0)
    Z = Z.div(Z.std(axis=1).replace(0, 1), axis=0).values.astype(np.double)
    D = dtw.distance_matrix_fast(Z, window=window, compact=False)
    D = np.asarray(D)
    np.fill_diagonal(D, 0)
    med = np.median(D[np.triu_indices_from(D, 1)])
    return np.exp(-D / med)


def lagged_correlations(feat: Features, total: str, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Для каждой пары: лаг (−L..L) с максимальной корреляцией и сама корреляция.
    Лаг > 0: ряд j повторяет ряд i с запаздыванием (i опережает j)."""
    r = feat.residual[total]
    X = r.sub(r.mean(axis=1), axis=0).values
    T = X.shape[1]
    best = np.full((len(X), len(X)), -np.inf)
    lag_of = np.zeros((len(X), len(X)), dtype=int)
    for L in range(-max_lag, max_lag + 1):
        a = X[:, :T - L] if L >= 0 else X[:, -L:]
        b = X[:, L:] if L >= 0 else X[:, :T + L]
        a = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True).clip(min=1e-12)
        b = (b - b.mean(1, keepdims=True)) / b.std(1, keepdims=True).clip(min=1e-12)
        C = a @ b.T / a.shape[1]
        upd = C > best
        best[upd], lag_of[upd] = C[upd], L
    return best, lag_of


def geography_similarity(cfg: dict, ids: np.ndarray) -> np.ndarray:
    g = cfg["graphs"]["geography"]
    D = distance_matrix(load_distances(cfg, g["type"]), ids)
    S = np.exp(-D / g["sigma_km"])
    S[~np.isfinite(D)] = 0.0
    return S


def build_graphs(cfg: dict, feat: Features, families: list[str] | None = None) -> dict[str, sp.csr_matrix]:
    gcfg = cfg["graphs"]
    total = cfg["features"]["total"]
    families = families or ["structure", "comovement", "leadlag", "geography"]
    ids = feat.X.index.values
    out = {}
    for f in families:
        if f == "structure":
            S = structure_similarity(feat); k = gcfg["knn"]
        elif f == "comovement":
            S = comovement_similarity(feat); k = gcfg["knn"]
        elif f == "leadlag":
            S = dtw_similarity(feat, total, gcfg["leadlag"]["dtw_window"]); k = gcfg["knn"]
        elif f == "geography":
            S = geography_similarity(cfg, ids); k = gcfg["geography"]["knn"]
        else:
            raise ValueError(f)
        out[f] = knn_sparsify(S, k, gcfg.get("mutual", False))
    return out


def multiplex(graphs: dict[str, sp.csr_matrix], weights: dict[str, float] | None = None) -> sp.csr_matrix:
    """Сумма слоёв, каждый нормирован на средний вес ребра (слои сопоставимы по масштабу)."""
    weights = weights or {k: 1.0 for k in graphs}
    out = None
    for k, A in graphs.items():
        if weights.get(k, 0) == 0:
            continue
        An = A / A.data.mean()
        out = An * weights[k] if out is None else out + An * weights[k]
    return sp.csr_matrix(out)


def graph_summary(graphs: dict[str, sp.csr_matrix]) -> pd.DataFrame:
    import scipy.sparse.csgraph as csg
    rows = []
    for k, A in graphs.items():
        n_comp, _ = csg.connected_components(A, directed=False)
        deg = np.asarray((A > 0).sum(axis=1)).ravel()
        rows.append({"граф": k, "рёбер": int(A.nnz / 2), "средняя степень": round(deg.mean(), 1),
                     "компонент связности": n_comp, "изолированных": int((deg == 0).sum())})
    return pd.DataFrame(rows)


def edge_overlap(graphs: dict[str, sp.csr_matrix]) -> pd.DataFrame:
    """Жаккар по множествам рёбер: насколько разные понятия «близости» дают одни и те же связи."""
    keys = list(graphs)
    E = {k: set(zip(*sp.triu(graphs[k], 1).nonzero())) for k in keys}
    M = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for a in keys:
        for b in keys:
            M.loc[a, b] = len(E[a] & E[b]) / max(len(E[a] | E[b]), 1)
    return M.round(3)
