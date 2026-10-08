"""Синтетическая проверка: когда сеть добавляет информацию к признакам (известный ответ).

Сценарии (N = 300, 3 типа по 100 МО):
  features   — признаки разделяют все типы, сеть случайна (Эрдёш–Реньи);
  network    — сеть (стохастическая блочная модель) разделяет все типы, признаки — шум;
  complement — признаки отделяют тип 0 от {1, 2}, сеть — тип 1 от типа 2;
  weak       — оба источника слабые.
Матрица вероятностей рёбер симметрична по построению: блочная матрица «внутри / между» умножается
на одну общую константу, подобранную под заданную среднюю степень (5, 10, 20 — как у kNN-графов).
Методы: k-means (признаки), Leiden (сеть) и KEFRiN:
  — исходный (Shalileh & Mirkin, 2022): ρ = ξ = 1 без масштабирования;
  — с выравниванием полного разброса блоков (евклидов, ξ×1 и ξ×4) — как в основной сетке;
  — косинусный, ξ ∈ {1, 16}.
Мера — ARI с истинным разбиением: среднее и стандартное отклонение по 10 повторам; у всех KEFRiN
одинаковый бюджет — 10 инициализаций.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari

from methods import KEFRiN, modularity_transform

SIZES = (100, 100, 100)


def scenario(name: str, seed: int, deg: float = 10.0):
    rng = np.random.default_rng(seed)
    z = np.repeat(np.arange(3), SIZES)
    n = len(z)
    Y = rng.normal(size=(n, 6))
    shift = {"features": 3.5, "network": 0.0, "complement": 0.0, "weak": 1.2}[name]
    Y[:, 0] += shift * (z == 1)
    Y[:, 1] += shift * (z == 2)
    if name == "complement":
        Y[z == 0, :2] += 3.0
    same = z[:, None] == z[None, :]
    if name == "features":
        base = np.ones((n, n))
    elif name == "network":
        base = np.where(same, 8.0, 1.0)
    elif name == "weak":
        base = np.where(same, 2.5, 1.0)
    else:                                   # complement: сеть различает только типы 1 и 2
        base = np.ones((n, n))
        for a in (1, 2):
            base[np.ix_(z == a, z == a)] = 8.0
    np.fill_diagonal(base, 0)
    P = base * (deg * n / base.sum())       # одна константа: симметрия сохраняется
    U = np.triu(rng.random((n, n)) < P, 1).astype(float)
    return z, Y, sp.csr_matrix(U + U.T)


def leiden_labels(A, seed):
    import igraph as ig
    import leidenalg as la
    U = sp.triu(A, 1).tocoo()
    g = ig.Graph(n=A.shape[0], edges=list(zip(U.row.tolist(), U.col.tolist())))
    return np.array(la.find_partition(g, la.ModularityVertexPartition, seed=seed).membership)


def run(reps: int = 10, degrees=(5, 10, 20)) -> pd.DataFrame:
    rows = []
    for deg in degrees:
        for name in ("features", "network", "complement", "weak"):
            for r in range(reps):
                z, Y, A = scenario(name, r, deg)
                P = modularity_transform(A)
                base = (Y ** 2).sum() / (P ** 2).sum()
                res = {"k-means (признаки)": KMeans(3, n_init=10, random_state=r).fit_predict(Y),
                       "Leiden (сеть)": leiden_labels(A, r),
                       "KEFRiN исходный (евкл., ρ = ξ = 1)": KEFRiN(3, "euclidean", 1, 1, 10, seed=r).fit_predict(Y, P)}
                for xm in (1, 4):
                    res[f"KEFRiN евкл., выравнивание, ξ×{xm}"] = KEFRiN(3, "euclidean", 1, base * xm, 10, seed=r).fit_predict(Y, P)
                for xm in (1, 16):
                    res[f"KEFRiN cos, ξ = {xm}"] = KEFRiN(3, "cosine", 1, xm, 10, seed=r).fit_predict(Y, P)
                for m, lab in res.items():
                    rows.append({"степень": deg, "сценарий": name, "метод": m, "ARI": ari(z, lab)})
    d = pd.DataFrame(rows)
    return d.groupby(["степень", "метод", "сценарий"]).ARI.agg(["mean", "std"]).round(2).reset_index()


if __name__ == "__main__":
    import sys
    from common import load_config, path
    t = run()
    print(t.pivot_table(index=["степень", "метод"], columns="сценарий", values="mean").to_string())
    t.to_csv(path(load_config(), "processed", "synthetic_benchmark.csv"), index=False)
    sys.exit(0)
