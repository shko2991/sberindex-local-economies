"""Проверки слоя рынка труда на синтетике."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labor import impute_demography, local_moran, section_letter  # noqa: E402


def test_section_letter_handles_cyrillic():
    assert section_letter("Раздел А Сельское, лесное хозяйство") == "A"
    assert section_letter("Раздел В Добыча полезных ископаемых") == "B"
    assert section_letter("Раздел Н Транспортировка и хранение") == "H"
    assert section_letter("Раздел C Обрабатывающие производства") == "C"
    assert section_letter("Всего по обследуемым видам экономической деятельности") is None


def ring(n=200, k=3):
    rows, cols = [], []
    for i in range(n):
        for d in range(1, k + 1):
            rows += [i, i]; cols += [(i + d) % n, (i - d) % n]
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def test_local_moran_finds_planted_clusters():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    x[20:40] += 4          # скопление высоких значений
    x[120:140] -= 4        # скопление низких
    lm = local_moran(ring(n), x, perms=199, seed=1)
    assert (lm.cluster.iloc[22:38] == "HH").mean() > 0.8
    assert (lm.cluster.iloc[122:138] == "LL").mean() > 0.8
    assert (lm.cluster.iloc[60:100] != "не значимо").mean() < 0.2


def test_local_moran_null_rate():
    x = np.random.default_rng(3).normal(size=300)
    lm = local_moran(ring(300), x, perms=199, seed=2)
    assert (lm.cluster != "не значимо").mean() < 0.12     # ~5% ложных срабатываний при α = 0,05


def test_impute_demography():
    lt = pd.DataFrame({"share_old": [0.3, np.nan, 0.2, np.nan], "share_young": [0.1, np.nan, 0.2, np.nan]},
                      index=[1, 2, 3, 4])
    ref = pd.DataFrame({"region_name": ["A", "A", "B", "C"], "mo_type": ["р", "р", "р", "о"]}, index=[1, 2, 3, 4])
    t = impute_demography(lt, ref)
    assert t.loc[2, "share_old"] == 0.3                    # медиана того же типа в регионе
    assert np.isnan(t.loc[4, "share_old"])                  # нет МО того же типа нигде
    assert t.imputed_age.tolist() == [False, True, False, True]
