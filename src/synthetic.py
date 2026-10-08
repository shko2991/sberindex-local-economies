"""Синтетическая проверка: когда сеть добавляет информацию к признакам (известный ответ).

Сценарии (N = 300, 3 типа, разреженная сеть со средней степенью ≈ 10, как у kNN-графов):
  features   — признаки разделяют все типы, сеть случайна;
  network    — сеть (стохастическая блочная модель) разделяет все типы, признаки — шум;
  complement — признаки отделяют тип 0 от {1, 2}, сеть — тип 1 от типа 2;
  weak       — оба источника слабые.
Методы: k-means (признаки), Leiden (сеть), KEFRiN (cosine/euclidean, ξ ∈ {1, 4, 16}).
Мера — ARI с истинным разбиением, среднее по повторам.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari

from methods import KEFRiN, modularity_transform


def scenario(name: str, seed: int, n: int = 300, deg: float = 10.0):
    rng = np.random.default_rng(seed)
    z = np.repeat(np.arange(3), n // 3)
    Y = rng.normal(size=(n, 6))
    shift = {"features": 3.5, "network": 0.0, "complement": 0.0, "weak": 1.2}[name]
    Y[:, 0] += shift * (z == 1); Y[:, 1] += shift * (z == 2)
    if name == "complement":
        Y[z == 0, :2] += 3.0
    pin, pout = {"features": (1, 1), "network": (8, 1), "complement": (1, 1), "weak": (2.5, 1)}[name]
    P = np.where(z[:, None] == z[None, :], pin, pout).astype(float)
    if name == "complement":
        P = np.ones((n, n)); P[np.ix_(z == 1, z == 1)] = 8; P[np.ix_(z == 2, z == 2)] = 8
    P *= deg / P.sum(1, keepdims=True)
    A = np.triu((rng.random((n, n)) < P).astype(float), 1)
    return z, Y, sp.csr_matrix(A + A.T)


def leiden_labels(A, seed):
    import igraph as ig
    import leidenalg as la
    U = sp.triu(A, 1).tocoo()
    g = ig.Graph(n=A.shape[0], edges=list(zip(U.row.tolist(), U.col.tolist())))
    return np.array(la.find_partition(g, la.ModularityVertexPartition, seed=seed).membership)


def run(reps: int = 10) -> pd.DataFrame:
    rows = []
    for name in ("features", "network", "complement", "weak"):
        for r in range(reps):
            z, Y, A = scenario(name, r)
            P = modularity_transform(A)
            res = {"k-means (признаки)": KMeans(3, n_init=10, random_state=r).fit_predict(Y),
                   "Leiden (сеть)": leiden_labels(A, r)}
            base = (Y ** 2).sum() / (P ** 2).sum()
            for d in ("cosine", "euclidean"):
                for xm in (1, 4, 16):
                    xi = xm * (1.0 if d == "cosine" else base)
                    res[f"KEFRiN {d} ξ×{xm}"] = KEFRiN(3, d, 1.0, xi, n_init=10, seed=r).fit_predict(Y, P)
            for m, lab in res.items():
                rows.append({"сценарий": name, "метод": m, "ARI": ari(z, lab)})
    return pd.DataFrame(rows).groupby(["метод", "сценарий"]).ARI.mean().unstack().round(2)


if __name__ == "__main__":
    import sys
    from common import load_config, path
    t = run()
    print(t.to_string())
    t.to_csv(path(load_config(), "processed", "synthetic_benchmark.csv"))
    sys.exit(0)
