"""Вспомогательные функции проверки переноса (src/extra_checks.py) на примерах с известным ответом."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from extra_checks import group_mean_pred, nearest_center, octiles, ols_pred, scale_train  # noqa: E402


def test_octiles_use_train_cutpoints_and_keep_missing_separate():
    train = pd.Series(np.arange(1, 81, dtype=float))            # 80 значений → по 10 в каждой из 8 групп
    test = pd.Series([0.5, 40.5, 80.5, np.nan])
    g_tr, g_te = octiles(train, test)
    assert np.bincount(g_tr).tolist() == [10] * 8
    assert g_te.tolist() == [0, 4, 7, -1]


def test_ols_with_one_grouping_equals_group_means():
    g_tr = np.array(["a", "a", "b", "b", "c"])
    y_tr = np.array([1.0, 3.0, 10.0, 12.0, 7.0])
    g_te = np.array(["b", "c", "a", "zz"])                     # неизвестная группа → базовый уровень
    p = ols_pred([g_tr], y_tr, [g_te])
    assert np.allclose(p[:3], [11.0, 7.0, 2.0])
    assert np.allclose(group_mean_pred(g_tr, y_tr, g_te)[:3], [11.0, 7.0, 2.0])
    assert np.isclose(group_mean_pred(g_tr, y_tr, g_te)[3], y_tr.mean())


def test_scaling_uses_train_only_and_keeps_column_weights():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3)) * np.array([1.0, 0.5, 0.2])
    tr, te = np.arange(150), np.arange(150, 200)
    Ytr, Yte = scale_train(X, tr, te, X.std(axis=0))
    assert np.allclose(Ytr.mean(axis=0), 0)
    assert np.allclose(Ytr.std(axis=0), X.std(axis=0))
    lab = (Ytr[:, 0] > 0).astype(int)
    assert set(nearest_center(Ytr, lab, Yte)) <= {0, 1}
