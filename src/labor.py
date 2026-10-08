"""Слой рынка труда: муниципальная статистика Росстата (БД ПМО в обработке «Если быть точным»,
CC BY 4.0) → признаки МО → типология рынка труда → сопоставление с типами потребления.

Показатели (организации без субъектов малого предпринимательства; данные по месту работы):
  Y48423007  среднемесячная заработная плата, руб.         Y48423005  среднесписочная численность работников
  Y48423006  фонд заработной платы, тыс. руб.              Y48112027  оценка численности населения на 1 января
  Y48112014  население по возрастным группам на 1 января   Y48112023  миграционный прирост за год
  Y48002001  число объектов розничной торговли и общепита

Привязка к territory_id: ОКТМО строки → версия справочника СберИндекса, действовавшая в этом году;
если версии нет — любая версия с тем же ОКТМО; затем oktmo_stable. Строки с пометкой
«mun_type в исходных данных не соответствует oktmo» (части района — поселения) отбрасываются.
Окно — 2023–2024 гг., как у данных о расходах; миграция — 2022–2023 (за 2024 г. нет данных).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import get_logger, path
from reference import load_versions

log = get_logger("labor")

TOTAL_OKVED = "Всего по обследуемым видам экономической деятельности"
# укрупнённые группы разделов ОКВЭД2 (буквы в источнике бывают кириллицей: А, В, Е, Н)
SECTOR_GROUPS = {
    "первичный (A, B)": "AB",
    "промышленность (C, D, E)": "CDE",
    "строительство (F)": "F",
    "рыночные услуги (G–N, R, S)": "GHIJKLMNRS",
    "бюджетный сектор (O, P, Q)": "OPQ",
}
CYR2LAT = str.maketrans({"А": "A", "В": "B", "Е": "E", "Н": "H", "С": "C", "К": "K", "М": "M", "О": "O", "Р": "P"})


def section_letter(okved: str) -> str | None:
    if not isinstance(okved, str) or not okved.startswith("Раздел "):
        return None
    return okved.split()[1].translate(CYR2LAT)


COLUMNS = ["indicator_code", "indicator_period", "oktmo", "oktmo_stable", "year", "indicator_value", "comment",
           "okved2", "vozr", "grup_2", "migr", "mest", "obroz"]


def load_bdmo(cfg: dict) -> pd.DataFrame:
    import pyarrow.parquet as pq
    # только нужные колонки, строки — категориями (иначе 2,8 млн строк × 27 текстовых колонок не влезают в память)
    t = pq.read_table(path(cfg, "external", cfg["labor"]["file"]), columns=COLUMNS)
    d = t.to_pandas(strings_to_categorical=True)
    for c in ("oktmo", "oktmo_stable"):
        d[c] = d[c].astype(str)
    d = d[d.comment != "mun_type в исходных данных не соответствует oktmo"].copy()
    d["year"] = d.year.astype(int)
    d["value"] = pd.to_numeric(d.indicator_value, errors="coerce")
    d = d[d.value.notna()]
    v = load_versions(cfg)[["oktmo8", "territory_id", "year_from", "year_to"]].drop_duplicates()
    keys = d[["oktmo", "oktmo_stable", "year"]].drop_duplicates()
    by_ver = keys.merge(v, left_on="oktmo", right_on="oktmo8")
    by_ver = by_ver[(by_ver.year >= by_ver.year_from) & (by_ver.year < by_ver.year_to)]
    by_ver = by_ver.drop_duplicates(["oktmo", "year"])[["oktmo", "year", "territory_id"]]
    anyv = keys.merge(v.drop_duplicates("oktmo8")[["oktmo8", "territory_id"]], left_on="oktmo", right_on="oktmo8")
    stab = keys.merge(v.drop_duplicates("oktmo8")[["oktmo8", "territory_id"]], left_on="oktmo_stable", right_on="oktmo8")
    m = keys.merge(by_ver, on=["oktmo", "year"], how="left")
    for alt in (anyv, stab):
        alt = alt.drop_duplicates(["oktmo", "year"]).set_index(["oktmo", "year"]).territory_id
        miss = m.territory_id.isna()
        m.loc[miss, "territory_id"] = [alt.get((o, y), np.nan) for o, y in zip(m.oktmo[miss], m.year[miss])]
    d = d.merge(m[["oktmo", "year", "territory_id"]].drop_duplicates(["oktmo", "year"]), on=["oktmo", "year"], how="left")
    log.info("строк Росстата: %d, привязано к territory_id: %.1f%%", len(d), 100 * d.territory_id.notna().mean())
    d = d[d.territory_id.notna()]
    d["territory_id"] = d.territory_id.astype(int)
    return d


def _annual(d: pd.DataFrame, code: str, period: str, years: list[int], extra: dict | None = None,
            how: str = "sum") -> pd.Series:
    """Значение показателя по МО: сумма по ОКТМО одного МО в году (слияния), затем среднее по годам."""
    s = d[(d.indicator_code == code) & (d.indicator_period == period) & d.year.isin(years)]
    for col, val in (extra or {}).items():
        s = s[s[col] == val]
    s = s.drop_duplicates(["territory_id", "year", "oktmo"] + list(extra or {}))
    per_year = s.groupby(["territory_id", "year"]).value.agg(how)
    return per_year.groupby("territory_id").mean()


def labor_table(cfg: dict, d: pd.DataFrame | None = None) -> pd.DataFrame:
    d = load_bdmo(cfg) if d is None else d
    yrs = cfg["labor"]["years"]
    out = pd.DataFrame()
    out["employees"] = _annual(d, "Y48423005", "Январь-декабрь", yrs, {"okved2": TOTAL_OKVED})
    out["fund_thrub"] = _annual(d, "Y48423006", "Январь-декабрь", yrs, {"okved2": TOTAL_OKVED})
    out["wage"] = _annual(d, "Y48423007", "Январь-декабрь", yrs, {"okved2": TOTAL_OKVED}, how="mean")
    pop_years = sorted(set(yrs) | {max(yrs) + 1})              # среднегодовая: 1 января 2023, 2024, 2025
    out["population"] = _annual(d, "Y48112027", "На 1 января", pop_years, {"mest": "Все население"})
    age = d[(d.indicator_code == "Y48112014") & (d.grup_2 == "Всего") & (d.indicator_period == "На 1 января")
            & d.year.isin(yrs)]
    age = age.drop_duplicates(["territory_id", "year", "oktmo", "vozr"]).groupby(["territory_id", "year", "vozr"]).value.sum()
    age = age.groupby(["territory_id", "vozr"]).mean().unstack()
    out["share_old"] = age["Старше трудоспособного возраста"] / age["Всего"]
    out["share_young"] = age["Моложе трудоспособного возраста"] / age["Всего"]
    mig = _annual(d, "Y48112023", "Значение показателя за год", cfg["labor"]["migration_years"],
                  {"migr": "Миграция — всего", "vozr": "Всего", "grup_2": "Всего"})
    out["mig_rate"] = mig / out.population * 1000
    # отраслевая структура занятости
    sec = d[(d.indicator_code == "Y48423005") & (d.indicator_period == "Январь-декабрь") & d.year.isin(yrs)].copy()
    sec["letter"] = sec.okved2.map(section_letter)
    sec = sec[sec.letter.notna()].drop_duplicates(["territory_id", "year", "oktmo", "okved2"])
    for name, letters in SECTOR_GROUPS.items():
        s = sec[sec.letter.isin(list(letters))].groupby(["territory_id", "year"]).value.sum().groupby("territory_id").mean()
        out[f"emp_{name}"] = s.reindex(out.index).fillna(0) / out.employees
    known = out[[c for c in out if c.startswith("emp_")]].sum(1)
    out["emp_не распределено"] = (1 - known).clip(lower=0)     # засекреченные (малые) разделы и прочее
    # розничная инфраструктура (для объяснения расхождений, не для типологии)
    ret = d[(d.indicator_code == "Y48002001") & d.year.isin(yrs) & d.indicator_period.str.contains("квартал")]
    ret = ret.drop_duplicates(["territory_id", "year", "indicator_period", "oktmo", "obroz"])
    per_q = ret.groupby(["territory_id", "year", "indicator_period"]).value.sum()
    out["retail_per_1000"] = per_q.groupby("territory_id").mean() / out.population * 1000
    modern = ret[ret.obroz.isin(["Супермаркеты", "Гипермаркеты", "Минимаркеты"])]
    shops = ret[ret.obroz.isin(["Магазины", "Супермаркеты", "Гипермаркеты", "Минимаркеты", "Универмаги",
                                "Специализированные продовольственные магазины",
                                "Специализированные непродовольственные магазины", "Прочие магазины"])]
    q_mod = modern.groupby(["territory_id", "year", "indicator_period"]).value.sum()
    q_all = shops.groupby(["territory_id", "year", "indicator_period"]).value.sum()
    out["modern_retail_share"] = (q_mod / q_all).groupby("territory_id").mean()
    # производные
    out["emp_rate"] = out.employees / out.population
    out["fund_pc"] = out.fund_thrub * 1000 / 12 / out.population       # руб. в месяц на жителя
    return out


def labor_features(lt: pd.DataFrame, ids: pd.Index) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Признаки для типологии рынка труда: уровень (зарплата, занятость), структура занятости (CLR),
    демография (доли старше/моложе трудоспособного, миграция). Блоки уравновешены, как у потребления."""
    from features import clr, standardize_blocks
    t = lt.reindex(ids)
    level = pd.DataFrame({"wage_rel": np.log(t.wage) - np.log(t.wage).median(),
                          "emp_rate_log": np.log(t.emp_rate.clip(lower=1e-3))})
    sec_cols = [c for c in t if c.startswith("emp_") and c != "emp_rate"]
    struct = clr(t[sec_cols].clip(lower=0) + 0.005).add_prefix("clr_")
    demo = t[["share_old", "share_young", "mig_rate"]].copy()
    demo["mig_rate"] = demo.mig_rate.clip(*demo.mig_rate.quantile([0.01, 0.99]))
    F = pd.concat([level, struct, demo], axis=1)
    blocks = {"level": list(level), "structure": list(struct), "demography": list(demo)}
    ok = F.notna().all(1)
    return standardize_blocks(F[ok], blocks), blocks


def impute_demography(lt: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Доли возрастных групп там, где их нет (внутригородские территории Петербурга и несколько МО):
    медиана МО того же типа в регионе, иначе — медиана МО того же типа по стране."""
    t = lt.copy()
    r = ref.reindex(t.index)
    t["imputed_age"] = t.share_old.isna()
    for c in ("share_old", "share_young"):
        reg = t.groupby([r.region_name, r.mo_type])[c].transform("median")
        nat = t.groupby(r.mo_type)[c].transform("median")
        t[c] = t[c].fillna(reg).fillna(nat)
    return t


def distance_to_capital(cfg: dict, ref: pd.DataFrame, ids: pd.Index) -> pd.Series:
    """Расстояние по дорогам до административного центра своего субъекта (км; 0 — сам центр)."""
    from data import distance_matrix, load_distances
    caps = ref[ref.status == "административный_центр_субъекта"]
    allids = np.array(sorted(set(ids) | set(caps.index)))
    D = distance_matrix(load_distances(cfg, "highway"), allids)
    pos = pd.Series(np.arange(len(allids)), index=allids)
    out = {}
    for i in ids:
        c = caps[caps.region_name == ref.region_name.get(i)]
        if len(c) == 0:
            out[i] = np.nan
            continue
        d = min(D[pos[i], pos[j]] for j in c.index if j in pos.index)
        out[i] = d if np.isfinite(d) else np.nan
    return pd.Series(out)


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Поправка Бенджамини–Хохберга на множественные сравнения (q-значения)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def local_moran(A, x: np.ndarray, perms: int = 499, seed: int = 0, alpha: float = 0.05,
                fdr: bool = True) -> pd.DataFrame:
    """Локальный I Морана (Anselin, 1995) с условной перестановкой: значения соседей случайны,
    веса сохраняются. Квадранты: HH — высокое среди высоких, LL — низкое среди низких и т.д.
    Значимость — с поправкой Бенджамини–Хохберга (fdr=True): из ~2 000 тестов при α = 0,05
    без поправки около сотни «скоплений» были бы случайными."""
    import scipy.sparse as sp
    A = sp.csr_matrix(A, dtype=float)
    n = len(x)
    z = (x - x.mean()) / x.std()
    deg = np.asarray(A.sum(1)).ravel()
    W = sp.diags(np.where(deg > 0, 1 / np.maximum(deg, 1e-12), 0)) @ A
    lag = W @ z
    I = z * lag
    g = np.random.default_rng(seed)
    p = np.ones(n)
    for i in range(n):
        row = W.getrow(i)
        k = row.nnz
        if k == 0:
            continue
        idx = g.integers(0, n - 1, size=(perms, k))
        idx[idx >= i] += 1
        Ip = z[i] * (z[idx] * row.data).sum(1)
        p[i] = (1 + (Ip >= I[i]).sum() if I[i] >= 0 else 1 + (Ip <= I[i]).sum()) / (perms + 1)
    quad = np.where(z > 0, np.where(lag > 0, "HH", "HL"), np.where(lag > 0, "LH", "LL"))
    q = bh_qvalues(np.where(deg > 0, p, 1.0)) if fdr else p
    sig = (q < alpha) & (deg > 0)
    return pd.DataFrame({"I": I, "p": p, "q": q, "quadrant": quad, "cluster": np.where(sig, quad, "не значимо")})


def ols(y: pd.Series, X: pd.DataFrame, standardize: bool = True, groups: pd.Series | None = None):
    """МНК; ошибки — кластеризованные по группам (регионам), если groups задан: остатки соседних
    МО коррелированы, и обычные или HC3-ошибки завысили бы значимость. Иначе — HC3.
    При standardize — стандартизованные коэффициенты (бинарные признаки не стандартизуются)."""
    import statsmodels.api as sm
    d = pd.concat([y.rename("y"), X] + ([groups.rename("_g")] if groups is not None else []), axis=1).dropna()
    g = d.pop("_g") if groups is not None else None
    Xs = d.drop(columns="y")
    yy = d.y
    if standardize:
        num = [c for c in Xs if Xs[c].nunique() > 2]
        Xs[num] = (Xs[num] - Xs[num].mean()) / Xs[num].std()
        yy = (yy - yy.mean()) / yy.std()
    model = sm.OLS(yy, sm.add_constant(Xs.astype(float)))
    if g is not None:
        return model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g)[0]})
    return model.fit(cov_type="HC3")
