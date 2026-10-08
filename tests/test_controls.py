"""Контрольные примеры, посчитанные вручную, и проверки исправлений по рецензии."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dynamics import yearly_mode  # noqa: E402
from icvi import graph_indices, s_dbw  # noqa: E402
from network_choice import degree_preserving_null  # noqa: E402


def test_s_dbw_hand_example():
    # A = {0,1,2}, B = {3,4,5}: Scat = (2/3)/(17.5/6) = 0.228571; stdev = √(4/3)/2 = 0.57735;
    # плотность в середине 2,5 — 2 точки, у центров 1 и 4 — по 1; Dens_bw = (2/1 + 2/1)/2 = 2
    X = np.array([[0.], [1.], [2.], [3.], [4.], [5.]])
    lab = np.array([0, 0, 0, 1, 1, 1])
    assert abs(s_dbw(X, lab) - (0.228571428 + 2.0)) < 1e-6


def test_s_dbw_zero_density_convention():
    # A = {0,1,2}, B = {10,11,12}: в середине пусто → Dens_bw = 0; S_Dbw = Scat = (2/3)/(154/6)
    X = np.array([[0.], [1.], [2.], [10.], [11.], [12.]])
    lab = np.array([0, 0, 0, 1, 1, 1])
    assert abs(s_dbw(X, lab) - (2 / 3) / (154 / 6)) < 1e-9


def test_avi_avu_mq_three_clusters_unequal_weights():
    # S = [[6,1,0.5],[1,2,2],[0.5,2,4]] (внутренние рёбра входят в S_kk дважды)
    edges = {(0, 1): 3, (2, 3): 1, (4, 5): 2, (1, 2): 1, (3, 4): 2, (0, 5): 0.5}
    A = np.zeros((6, 6))
    for (i, j), w in edges.items():
        A[i, j] = A[j, i] = w
    r = graph_indices(sp.csr_matrix(A), np.array([0, 0, 1, 1, 2, 2]))
    assert abs(r["AVI"] - np.mean([6 / 7.5, 2 / 5, 4 / 6.5])) < 1e-12
    assert abs(r["AVU"] - 2 / 3) < 1e-12
    assert abs(r["MQ"] - (12 / 19 - 123.5 / 361)) < 1e-12


def test_yearly_mode_tie_takes_latest():
    T = pd.DataFrame([[0, 1, 1, 0, 2, 2, 2, 3], [5, 5, 6, 6, 1, 2, 3, 4]],
                     columns=["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1", "2024Q2", "2024Q3", "2024Q4"])
    Y = yearly_mode(T)
    assert Y["2023"].tolist() == [0, 6]      # A–B–B–A → A (последний среди равных), 5–5–6–6 → 6
    assert Y["2024"].tolist() == [2, 4]      # 2 встречается трижды; все разные → последний


def test_null_model_preserves_degrees_exactly():
    rng = np.random.default_rng(0)
    U = np.triu((rng.random((120, 120)) < 0.08).astype(float) * rng.random((120, 120)), 1)
    A = sp.csr_matrix(U + U.T)
    N = degree_preserving_null(A, seed=3)
    d1 = np.asarray((A > 0).sum(1)).ravel()
    d2 = np.asarray((N > 0).sum(1)).ravel()
    assert (d1 == d2).all() and N.diagonal().sum() == 0
    assert abs(np.sort(sp.triu(A, 1).data) - np.sort(sp.triu(N, 1).data)).max() < 1e-12
