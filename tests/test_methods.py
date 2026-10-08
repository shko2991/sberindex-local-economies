"""Проверки методов и индексов на синтетике с известным ответом. Запуск: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dynamics import jaccard_match, trajectory_classes  # noqa: E402
from fca import concepts_up_to, delta_measure, describe_clusters  # noqa: E402
from graphs import knn_sparsify  # noqa: E402
from icvi import aggregate, borda, copeland, graph_indices, rank_table, s_dbw  # noqa: E402
from methods import KEFRiN, modularity_transform, pattern_clusters, pattern_codes, kemeny_distance  # noqa: E402


def ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(a, b)


def sbm(n=240, k=3, pin=0.15, pout=0.02, seed=0):
    rng = np.random.default_rng(seed)
    z = np.repeat(np.arange(k), n // k)
    prob = np.where(z[:, None] == z[None, :], pin, pout)
    A = np.triu((rng.random((n, n)) < prob).astype(float), 1)
    return z, sp.csr_matrix(A + A.T)


def test_knn_symmetric_and_degree():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    S = -((X[:, None] - X[None]) ** 2).sum(-1) + 100
    A = knn_sparsify(S, 5)
    assert (A != A.T).nnz == 0
    assert A.diagonal().sum() == 0
    assert (np.asarray((A > 0).sum(1)).ravel() >= 5).all()


def test_kefrin_network_only_recovers_sbm():
    z, A = sbm()
    P = modularity_transform(A)
    Y = np.random.default_rng(2).normal(size=(len(z), 3))
    for d in ("cosine", "euclidean"):
        lab = KEFRiN(3, d, rho=0.0, xi=1.0, n_init=10, seed=1).fit_predict(Y, P)
        assert ari(z, lab) > 0.95, d


def test_kefrin_combines_complementary_sources():
    """Признаки отделяют тип 0 от {1, 2}, сеть — тип 1 от типа 2: вместе — все три."""
    rng = np.random.default_rng(0)
    n, k = 300, 3
    z = np.repeat(np.arange(k), n // k)
    Y = rng.normal(size=(n, 4)); Y[z == 0, :2] += 3
    prob = np.full((n, n), 0.03)
    for a in (1, 2):
        prob[np.ix_(z == a, z == a)] = 0.15
    A = np.triu((rng.random((n, n)) < prob).astype(float), 1); A = A + A.T
    P = modularity_transform(sp.csr_matrix(A))
    xi = (Y ** 2).sum() / (P ** 2).sum() * 4
    lab = KEFRiN(3, "euclidean", 1.0, xi, n_init=20, seed=2).fit_predict(Y, P)
    only_f = KEFRiN(3, "euclidean", 1.0, 0.0, n_init=20, seed=2).fit_predict(Y, P)
    assert ari(z, lab) > 0.9 and ari(z, only_f) < 0.7


def test_kefrin_criterion_not_worse_than_planted_in_easy_case():
    z, A = sbm(pin=0.3, pout=0.01)
    P = modularity_transform(A)
    Y = np.eye(3)[z] * 3 + np.random.default_rng(3).normal(size=(len(z), 3)) * 0.3
    m = KEFRiN(3, "cosine", 1, 1, n_init=5, seed=0)
    lab = m.fit_predict(Y, P)
    assert ari(z, lab) == 1.0


def test_modularity_matches_networkx():
    import networkx as nx
    z, A = sbm(seed=4)
    lab = np.random.default_rng(5).integers(0, 4, len(z))
    G = nx.from_scipy_sparse_array(A)
    ref = nx.community.modularity(G, [set(np.where(lab == c)[0]) for c in range(4)])
    assert abs(graph_indices(A, lab)["MQ"] - ref) < 1e-10


def test_avi_avu_by_hand():
    # кластер 0 = {0,1}, кластер 1 = {2,3}; S = [[4,1],[1,2]] (внутренние рёбра входят дважды)
    A = sp.csr_matrix(np.array([[0, 2, 0, 0], [2, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=float))
    r = graph_indices(A, np.array([0, 0, 1, 1]))
    assert abs(r["AVI"] - (4 / 5 + 2 / 3) / 2) < 1e-12
    assert abs(r["AVU"] - 1.0) < 1e-12


def test_s_dbw_prefers_true_partition():
    rng = np.random.default_rng(6)
    z = np.repeat([0, 1, 2], 100)
    X = np.c_[z * 4.0, np.zeros(300)] + rng.normal(size=(300, 2))
    assert s_dbw(X, z) < s_dbw(X, rng.permutation(z))


def test_pattern_invariance_and_codes():
    rng = np.random.default_rng(7)
    Z = rng.normal(size=(200, 4))
    a = pattern_clusters(Z)
    b = pattern_clusters(np.exp(Z[:, [3, 1, 0, 2]]))       # перестановка + общая монотонная функция
    assert ari(a, b) == 1.0
    c = pattern_codes(np.array([[1.0, 2.0, 2.0]]))
    assert c.tolist() == [[1, 1, 0]]                          # пары (1,2), (1,3), (2,3)
    assert kemeny_distance(np.array([[1, 1, 0]]), np.array([[2, 1, 1]]))[0, 0] == 1.5


def test_rank_aggregation():
    df = pd.DataFrame({"SW": [0.5, 0.4, 0.1], "CH": [10, 20, 5], "S_Dbw": [0.3, 0.4, 0.9],
                       "AVI": [0.6, 0.7, 0.2], "AVU": [0.2, 0.3, 0.9], "MQ": [0.4, 0.5, 0.1]}, index=list("abc"))
    idx = list(df.columns)
    R = rank_table(df, idx)
    assert borda(R).idxmin() == "c" and copeland(R)["c"] == -2
    out = aggregate(df, idx)
    assert out.index[-1] == "c"


def test_jaccard_match_and_trajectories():
    ref = np.array([0, 0, 1, 1, 2, 2])
    lab = np.array([2, 2, 0, 0, 1, 1])
    assert (jaccard_match(ref, lab) == ref).all()
    T = pd.DataFrame([[0] * 8, [0, 0, 0, 1, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1], [0] * 7 + [1]])
    assert trajectory_classes(T).tolist() == ["стабильный", "устойчивый сдвиг", "колебания", "колебания"]


def test_fca_concepts_and_delta():
    B = np.array([[1, 1, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1]], dtype=bool)
    cons = dict(concepts_up_to(B, max_len=2, min_extent=1))
    assert frozenset({0, 1}) in cons                       # понятие ({0,1}, {a,b})
    ext = cons[frozenset({0})]
    assert ext.tolist() == [True, True, True, False]
    assert delta_measure(B, frozenset({0}), ext) == 1       # добавление b теряет 1 МО
    Bdf = pd.DataFrame(B, columns=["a", "b", "c"])
    d = describe_clusters(Bdf, pd.Series([0, 0, 1, 1]), max_len=2, min_precision=0.9, min_extent=1)
    assert d[d.type == 0].description.iloc[0] in ("b", "a ∧ b")
