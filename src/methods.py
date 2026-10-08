"""Методы кластеризации. Единый интерфейс: run(name, ctx, k=..., resolution=..., seed=...) → метки.

Признаковые (только матрица X):           kmeans, gmm, ward, pattern
Сетевые (только граф):                     leiden, spectral
Признаки + сеть:                           kefrin (Shalileh & Mirkin, 2022), spectral_joint

KEFRiN реализован по статье (без чужого кода):
  Shalileh S., Mirkin B. Community Partitioning over Feature-Rich Networks Using an Extended
  K-Means Method // Entropy. 2022. 24(5):626. https://doi.org/10.3390/e24050626
  Критерий: F = ρ·Σ_i d(y_i, c_k(i)) + ξ·Σ_i d(p_i, λ_k(i)),
  y_i — признаки МО, p_i — строка матрицы связей (после модулярного преобразования),
  c_k, λ_k — центры кластера в пространстве признаков и в пространстве связей.
  Шаги: MaxMin-инициализация (первое семя случайно, каждое следующее — объект с максимальной
  суммой расстояний до уже выбранных); поочерёдно «назначение — пересчёт центров» до
  неизменности разбиения. Варианты расстояния: euclidean (квадрат), cosine (1 − cos,
  векторы и центры нормируются), manhattan (L1, центр — медиана).

Порядково-инвариантная кластеризация паттернов (анализ паттернов, школа Ф.Т. Алескерова):
  Мячин А.Л. Анализ образцов в параллельных координатах на основе попарного сравнения
  параметров // Автоматика и телемеханика. 2019. №1. С. 138–152. doi:10.1134/S0005231019010100
  Для каждой пары показателей (s, j) объекта i код e_i^{sj} = 1, если x_is < x_ij; 0 — равны;
  2 — x_is > x_ij. Объекты с совпадающими кодами всех пар образуют кластер (число кластеров
  не задаётся). Результат не зависит от порядка показателей и от монотонных преобразований,
  общих для всех показателей. Сравнивать можно только сопоставимые величины, поэтому показатели
  предварительно приводятся к z-оценкам (отклонение МО от среднего по стране в станд. откл.).
  Наше расширение для сравнения с методами при фиксированном K: K самых частых паттернов —
  центры, остальные объекты присоединяются к ближайшему по Хэммингу на векторе кодов
  (число пар показателей с разным порядком — расстояние Кемени между слабыми порядками).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg


@dataclass
class Context:
    X: np.ndarray                                   # признаки (N × V), уже стандартизованы
    graphs: dict[str, sp.csr_matrix]                # семейства рёбер (N × N)
    cfg: dict
    pattern_X: np.ndarray | None = None             # показатели для анализа паттернов
    _cache: dict = field(default_factory=dict)

    def subset(self, idx: np.ndarray) -> "Context":
        sub = Context(X=self.X[idx], graphs={k: sp.csr_matrix(A[idx][:, idx]) for k, A in self.graphs.items()},
                      cfg=self.cfg, pattern_X=None if self.pattern_X is None else self.pattern_X[idx])
        if "multiplex" in self._cache:          # подграф того же мультиплекса (нормировка — по полной сети)
            M = self._cache["multiplex"]
            sub._cache["multiplex"] = sp.csr_matrix(M[idx][:, idx])
        return sub

    def graph(self, name: str) -> sp.csr_matrix:
        if name in self.graphs:
            return self.graphs[name]
        if name == "multiplex":
            from graphs import multiplex
            if "multiplex" not in self._cache:
                layers = self.cfg["graphs"].get("multiplex_layers", list(self.graphs))
                self._cache["multiplex"] = multiplex({k: self.graphs[k] for k in layers})
            return self._cache["multiplex"]
        raise KeyError(name)


# ----------------------------------------------------------------------------- признаковые

def kmeans(ctx: Context, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    return KMeans(k, n_init=ctx.cfg["methods"]["kmeans"]["n_init"], random_state=seed).fit_predict(ctx.X)


def gmm(ctx: Context, k: int, seed: int) -> np.ndarray:
    from sklearn.mixture import GaussianMixture
    c = ctx.cfg["methods"]["gmm"]
    return GaussianMixture(k, covariance_type=c["covariance_type"], n_init=c["n_init"], reg_covar=1e-4,
                           random_state=seed).fit(ctx.X).predict(ctx.X)


def ward(ctx: Context, k: int, seed: int = 0) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(k, linkage="ward").fit_predict(ctx.X)


def pattern_codes(Z: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """Коды всех пар показателей (s < j): 1 — x_s < x_j, 0 — |x_s − x_j| ≤ tol, 2 — x_s > x_j."""
    n = Z.shape[1]
    s, j = np.triu_indices(n, 1)
    diff = Z[:, s] - Z[:, j]
    codes = np.where(diff < -tol, 1, np.where(diff > tol, 2, 0))
    return codes.astype(np.int8)


def pattern_clusters(Z: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """Исходный метод: кластер = множество объектов с одинаковым вектором кодов."""
    codes = pattern_codes(Z, tol)
    _, lab = np.unique(codes, axis=0, return_inverse=True)
    return lab.ravel()


def pattern_k(ctx: Context, k: int, seed: int = 0) -> np.ndarray:
    tol = ctx.cfg["methods"]["pattern"]["tolerance"]
    codes = pattern_codes(ctx.pattern_X, tol)
    uniq, lab, cnt = np.unique(codes, axis=0, return_inverse=True, return_counts=True)
    lab = lab.ravel()
    top = np.argsort(-cnt, kind="stable")[:k]          # центры — K самых частых паттернов
    D = kemeny_distance(uniq, uniq[top])
    # при равенстве расстояний — к более частому паттерну (top уже упорядочен по частоте)
    nearest = D.argmin(axis=1)
    return nearest[lab]


def kemeny_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Расстояние между векторами кодов пар: противоположный строгий порядок пары — 1,
    строгий порядок против равенства — 0.5 (расстояние Кемени между слабыми порядками)."""
    a, b = A[:, None, :], B[None, :, :]
    return np.where(a == b, 0.0, np.where((a == 0) | (b == 0), 0.5, 1.0)).sum(axis=2)


# ----------------------------------------------------------------------------- сетевые

def _to_igraph(A: sp.csr_matrix):
    import igraph as ig
    U = sp.triu(A, 1).tocoo()
    g = ig.Graph(n=A.shape[0], edges=list(zip(U.row.tolist(), U.col.tolist())), directed=False)
    g.es["weight"] = U.data.tolist()
    return g


def leiden(ctx: Context, network: str, resolution: float, seed: int) -> np.ndarray:
    """Leiden (Traag et al., 2019), модулярность с параметром разрешения. Узлы вне главной
    компоненты (МО без дорожной связи) присоединяются по признакам, а не образуют одиночные типы."""
    import leidenalg as la
    A = ctx.graph(network)
    _, comp = csg.connected_components(A, directed=False)
    main = np.where(comp == np.bincount(comp).argmax())[0]
    g = _to_igraph(sp.csr_matrix(A[main][:, main]))
    part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                             resolution_parameter=resolution, seed=int(seed) % (2 ** 31),
                             n_iterations=ctx.cfg["methods"]["leiden"]["n_iterations"])
    return _attach_small_components(A, ctx.X, np.array(part.membership), main)


def _attach_small_components(A: sp.csr_matrix, X: np.ndarray, labels_main: np.ndarray, main: np.ndarray) -> np.ndarray:
    """Узлы вне главной компоненты связности → кластер с ближайшим центроидом признаков."""
    lab = np.full(A.shape[0], -1)
    lab[main] = labels_main
    ks = np.unique(labels_main)
    cent = np.stack([X[main][labels_main == c].mean(0) for c in ks])
    rest = np.where(lab < 0)[0]
    if len(rest):
        d = ((X[rest][:, None, :] - cent[None]) ** 2).sum(-1)
        lab[rest] = ks[d.argmin(1)]
    return lab


def spectral_graph(ctx: Context, A: sp.csr_matrix, k: int, seed: int) -> np.ndarray:
    """Спектральная кластеризация на главной компоненте связности (иначе мелкие компоненты
    забирают собственные векторы); остальные узлы присоединяются по признакам."""
    from sklearn.cluster import SpectralClustering
    _, comp = csg.connected_components(A, directed=False)
    main = np.where(comp == np.bincount(comp).argmax())[0]
    Am = A[main][:, main]
    lab = SpectralClustering(k, affinity="precomputed", assign_labels=ctx.cfg["methods"]["spectral"]["assign_labels"],
                             random_state=seed).fit_predict(Am)
    return _attach_small_components(A, ctx.X, lab, main)


def spectral(ctx: Context, network: str, k: int, seed: int) -> np.ndarray:
    return spectral_graph(ctx, ctx.graph(network), k, seed)


def spectral_joint(ctx: Context, network: str, k: int, seed: int) -> np.ndarray:
    """Признаки + сеть: спектральная кластеризация суммы графа признаковой близости (structure)
    и выбранной сети, каждый слой нормирован на средний вес ребра."""
    from graphs import multiplex
    layers = {"structure": ctx.graphs["structure"], "net": ctx.graph(network)}
    return spectral_graph(ctx, multiplex(layers), k, seed)


# ----------------------------------------------------------------------------- KEFRiN

def modularity_transform(A: sp.csr_matrix) -> np.ndarray:
    """p_ij − p_i+ · p_+j / p_++ (модулярное преобразование, как в статье)."""
    A = sp.csr_matrix(A, dtype=float)
    r = np.asarray(A.sum(1)).ravel()
    c = np.asarray(A.sum(0)).ravel()
    tot = r.sum()
    return A.toarray() - np.outer(r, c) / tot


def _row_normalize(M: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(M, axis=1, keepdims=True)
    return np.divide(M, nrm, out=np.zeros_like(M), where=nrm > 1e-12)


class KEFRiN:
    """K-means for feature-rich networks. Нулевые строки (изолированные узлы) в варианте cosine
    дают одинаковое расстояние до всех центров сети — их назначение решают признаки."""

    def __init__(self, k: int, distance: str = "cosine", rho: float = 1.0, xi: float = 1.0,
                 n_init: int = 20, max_iter: int = 100, seed: int = 0):
        if distance not in ("euclidean", "cosine", "manhattan"):
            raise ValueError(distance)
        self.k, self.distance, self.rho, self.xi = k, distance, rho, xi
        self.n_init, self.max_iter, self.seed = n_init, max_iter, seed

    # расстояния объект–центр для одного блока (N × K)
    def _d(self, M: np.ndarray, C: np.ndarray, sq: np.ndarray | None = None) -> np.ndarray:
        if self.distance == "euclidean":
            sq = (M ** 2).sum(1) if sq is None else sq
            return sq[:, None] - 2 * M @ C.T + (C ** 2).sum(1)[None, :]
        if self.distance == "cosine":
            return 1.0 - M @ C.T                       # M и C уже нормированы
        return np.abs(M[:, None, :] - C[None, :, :]).sum(-1) if M.shape[1] <= 64 else \
            np.stack([np.abs(M - c).sum(1) for c in C], axis=1)

    def _center(self, M: np.ndarray) -> np.ndarray:
        if self.distance == "manhattan":
            return np.median(M, axis=0)
        c = M.mean(0)
        if self.distance == "cosine":
            n = np.linalg.norm(c)
            c = c / n if n > 1e-12 else c
        return c

    def _dist(self, Y, P, Cy, Cp, sqY=None, sqP=None):
        return self.rho * self._d(Y, Cy, sqY) + self.xi * self._d(P, Cp, sqP)

    def _centers(self, M: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if self.distance == "manhattan":
            return np.stack([np.median(M[labels == c], axis=0) for c in range(self.k)])
        H = np.zeros((self.k, len(labels)))
        H[labels, np.arange(len(labels))] = 1.0
        C = (H @ M) / H.sum(1, keepdims=True)
        return _row_normalize(C) if self.distance == "cosine" else C

    def _maxmin_seeds(self, Y, P, rng) -> list[int]:
        seeds = [int(rng.integers(len(Y)))]
        acc = np.zeros(len(Y))
        while len(seeds) < self.k:
            s = seeds[-1]
            acc += self._dist(Y, P, Y[s:s + 1], P[s:s + 1]).ravel()
            cand = acc.copy()
            cand[seeds] = -np.inf
            seeds.append(int(np.argmax(cand)))
        return seeds

    def _single(self, Y, P, rng):
        seeds = self._maxmin_seeds(Y, P, rng)
        Cy, Cp = Y[seeds].copy(), P[seeds].copy()
        if self.distance == "cosine":
            Cy, Cp = _row_normalize(Cy), _row_normalize(Cp)
        sqY, sqP = (Y ** 2).sum(1), (P ** 2).sum(1)
        labels = None
        for it in range(self.max_iter):
            D = self._dist(Y, P, Cy, Cp, sqY, sqP)
            new = D.argmin(1)
            if labels is not None and np.array_equal(new, labels):
                break
            labels = new
            for c in range(self.k):
                if not (labels == c).any():        # пустой кластер: самый далёкий от своего центра объект
                    far = int(np.argmax(D[np.arange(len(Y)), labels]))
                    labels[far] = c
            Cy, Cp = self._centers(Y, labels), self._centers(P, labels)
        D = self._dist(Y, P, Cy, Cp, sqY, sqP)
        crit = float(D[np.arange(len(Y)), labels].sum())
        return labels, crit, it + 1

    def fit_predict(self, Y: np.ndarray, P: np.ndarray) -> np.ndarray:
        Y = np.asarray(Y, dtype=float); P = np.asarray(P, dtype=float)
        if self.distance == "cosine":
            Y, P = _row_normalize(Y), _row_normalize(P)
        rng = np.random.default_rng(self.seed)
        best = (None, np.inf, 0)
        for _ in range(self.n_init):
            res = self._single(Y, P, rng)
            if res[1] < best[1]:
                best = res
        self.labels_, self.criterion_, self.n_iter_ = best
        return self.labels_


def kefrin(ctx: Context, network: str, k: int, seed: int, distance: str | None = None,
           n_init: int | None = None, xi_mult: float = 1.0) -> np.ndarray:
    c = ctx.cfg["methods"]["kefrin"]
    key = ("P", network)
    if key not in ctx._cache:
        ctx._cache[key] = modularity_transform(ctx.graph(network))
    P = ctx._cache[key]
    Y = ctx.X
    rho, xi = c["rho_feature"], c["rho_network"] * xi_mult
    dist = distance or c["distance"]
    if dist != "cosine" and c.get("balance_scatter", True):
        # для евклидова и L1 вариантов блоки разной размерности: уравниваем полный разброс блоков
        xi = xi * (Y ** 2).sum() / max((P ** 2).sum(), 1e-12) if dist == "euclidean" else \
            xi * np.abs(Y).sum() / max(np.abs(P).sum(), 1e-12)
    m = KEFRiN(k, dist, rho, xi, n_init or c["n_init"], c["max_iter"], seed)
    return m.fit_predict(Y, P)


# ----------------------------------------------------------------------------- реестр

FEATURE_METHODS = {"kmeans": kmeans, "gmm": gmm, "ward": ward, "pattern": pattern_k}
NETWORK_METHODS = {"spectral": spectral, "kefrin": kefrin, "spectral_joint": spectral_joint}


def run(name: str, ctx: Context, k: int | None = None, network: str | None = None,
        resolution: float | None = None, seed: int = 0, **kw) -> np.ndarray:
    if name in FEATURE_METHODS:
        return FEATURE_METHODS[name](ctx, k, seed)
    if name == "leiden":
        return leiden(ctx, network, resolution, seed)
    if name in NETWORK_METHODS:
        return NETWORK_METHODS[name](ctx, network, k, seed, **kw)
    raise ValueError(name)


def relabel_by_size(lab: np.ndarray) -> np.ndarray:
    """Номера кластеров по убыванию размера (0 — самый крупный)."""
    u, cnt = np.unique(lab, return_counts=True)
    order = u[np.argsort(-cnt, kind="stable")]
    m = {c: i for i, c in enumerate(order)}
    return np.array([m[c] for c in lab])
