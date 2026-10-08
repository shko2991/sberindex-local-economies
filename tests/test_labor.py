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


# --- контрпримеры из рецензии: агрегация таблицы рынка труда ---------------------------------
from labor import TOTAL_OKVED, labor_table  # noqa: E402

CFG = {"labor": {"years": [2023, 2024], "migration_years": [2022, 2023]}}


def _row(code, year, value, period="Январь-декабрь", **kw):
    base = {"indicator_code": code, "indicator_period": period, "indicator_unit": "", "year": year, "oktmo": "1",
            "oktmo_stable": "1", "territory_id": 1, "value": float(value), "okved2": None, "mest": None,
            "grup_2": None, "vozr": None, "migr": None, "obroz": None}
    base.update(kw)
    return base


def _frame(extra):
    rows = []
    for y in (2023, 2024):
        rows += [_row("Y48423005", y, 100, okved2=TOTAL_OKVED), _row("Y48423006", y, 1200, okved2=TOTAL_OKVED),
                 _row("Y48423007", y, 50000, okved2=TOTAL_OKVED)]
    for y, v in ((2023, 1000), (2024, 1200), (2025, 1200)):
        rows.append(_row("Y48112027", y, v, period="На 1 января", mest="Все население"))
    return pd.DataFrame(rows + extra)


def test_sector_shares_consistent_when_sections_published_in_different_years():
    sec = [_row("Y48423005", 2023, 80, okved2="Раздел A Сельское хозяйство"),
           _row("Y48423005", 2024, 20, okved2="Раздел A Сельское хозяйство"),
           _row("Y48423005", 2023, 20, okved2="Раздел C Обрабатывающие производства"),
           _row("Y48423005", 2024, 80, okved2="Раздел O Государственное управление")]
    t = labor_table(CFG, _frame(sec)).loc[1]
    shares = t[[c for c in t.index if c.startswith("emp_") and c != "emp_rate"]].astype(float)
    assert abs(shares.sum() - 1) < 1e-9
    assert abs(t["emp_первичный (A, B)"] - 0.5) < 1e-9 and abs(t["emp_промышленность (C, D, E)"] - 0.1) < 1e-9


def test_retail_total_not_double_counted():
    q = [_row("Y48002001", 2023, v, period="I квартал", obroz=o)
         for o, v in (("Магазины", 10), ("Супермаркеты", 6), ("Прочие магазины", 4), ("Киоски", 5))]
    t = labor_table(CFG, _frame(q)).loc[1]
    assert abs(t.modern_retail_share - 0.6) < 1e-9
    # торговля есть только за 2023 г.: знаменатель — среднегодовая численность 2023 г. (1000 + 1200) / 2
    assert abs(t.retail_per_1000 - 15 / 1100 * 1000) < 1e-9


def test_retail_total_not_rebuilt_from_partial_subtypes():
    q = [_row("Y48002001", 2023, 6, period="I квартал", obroz="Супермаркеты")]
    t = labor_table(CFG, _frame(q)).loc[1]
    assert np.isnan(t.modern_retail_share) and np.isnan(t.retail_per_1000)


def test_sector_balance_violation_year_is_excluded():
    sec = [_row("Y48423005", 2023, 120, okved2="Раздел A Сельское хозяйство"),     # больше итога 100: не округление
           _row("Y48423005", 2024, 30, okved2="Раздел A Сельское хозяйство")]
    diag = {}
    t = labor_table(CFG, _frame(sec), diag).loc[1]
    assert diag["МО-лет: сумма разделов больше итога сверх округления (исключены)"] == 1
    assert abs(t["emp_первичный (A, B)"] - 0.3) < 1e-9                         # только 2024 г.
    assert len(diag["таблица: нарушения баланса разделов"]) == 1


def test_population_is_mean_of_annual_means_and_urban_zero_only_when_confirmed():
    t = labor_table(CFG, _frame([])).loc[1]
    assert abs(t.population - 1150) < 1e-9                     # (1000 + 2·1200 + 1200) / 4
    assert np.isnan(t.share_urban) and t.urban_status == "нет данных"
    rural = [_row("Y48112027", y, v, period="На 1 января", mest="Сельское население")
             for y, v in ((2023, 1000), (2024, 1200), (2025, 1200))]
    t2 = labor_table(CFG, _frame(rural)).loc[1]
    assert t2.share_urban == 0 and t2.urban_status == "структурный ноль"
