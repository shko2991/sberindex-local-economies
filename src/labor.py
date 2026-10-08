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
    "госуправление, образование, здравоохранение (O, P, Q)": "OPQ",   # отрасли; прокси бюджетной занятости
}
CYR2LAT = str.maketrans({"А": "A", "В": "B", "Е": "E", "Н": "H", "С": "C", "К": "K", "М": "M", "О": "O", "Р": "P"})


def section_letter(okved: str) -> str | None:
    if not isinstance(okved, str) or not okved.startswith("Раздел "):
        return None
    return okved.split()[1].translate(CYR2LAT)


COLUMNS = ["indicator_code", "indicator_period", "indicator_unit", "oktmo", "oktmo_stable", "year", "indicator_value", "comment",
           "okved2", "vozr", "grup_2", "migr", "mest", "obroz"]


def load_bdmo(cfg: dict, return_map: bool = False):
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
    m["способ привязки"] = np.where(m.territory_id.notna(), "версия справочника года", None)
    for alt, how in ((anyv, "ОКТМО из другой версии"), (stab, "oktmo_stable")):
        alt = alt.drop_duplicates(["oktmo", "year"]).set_index(["oktmo", "year"]).territory_id
        miss = m.territory_id.isna()
        m.loc[miss, "territory_id"] = [alt.get((o, y), np.nan) for o, y in zip(m.oktmo[miss], m.year[miss])]
        m.loc[miss & m.territory_id.notna(), "способ привязки"] = how
    m = m.drop_duplicates(["oktmo", "year"])
    m["способ привязки"] = m["способ привязки"].fillna("не привязано")
    om = m[["oktmo", "oktmo_stable", "year", "territory_id", "способ привязки"]]
    d = d.merge(m[["oktmo", "year", "territory_id"]], on=["oktmo", "year"], how="left")
    log.info("строк Росстата: %d, привязано к territory_id: %.1f%%", len(d), 100 * d.territory_id.notna().mean())
    d = d[d.territory_id.notna()].copy()
    d["territory_id"] = d.territory_id.astype(int)
    return (d, om) if return_map else d


def _annual(d: pd.DataFrame, code: str, period: str, years: list[int], extra: dict | None = None,
            how: str = "sum") -> pd.Series:
    """Значение показателя по МО: сумма по ОКТМО одного МО в году (слияния), затем среднее по годам."""
    s = d[(d.indicator_code == code) & (d.indicator_period == period) & d.year.isin(years)]
    for col, val in (extra or {}).items():
        s = s[s[col] == val]
    s = s.drop_duplicates(["territory_id", "year", "oktmo"] + list(extra or {}))
    per_year = s.groupby(["territory_id", "year"]).value.agg(how)
    return per_year.groupby("territory_id").mean()


def _weighted_wage(d: pd.DataFrame, years: list[int]) -> pd.Series:
    """Средняя зарплата МО: если одному МО в году соответствуют несколько ОКТМО (слияния),
    зарплаты усредняются с весами — численностью работников тех же ОКТМО; затем среднее по годам."""
    key = ["territory_id", "year", "oktmo"]
    base = (d.indicator_period == "Январь-декабрь") & d.year.isin(years) & (d.okved2 == TOTAL_OKVED)
    w = d[base & (d.indicator_code == "Y48423007")].drop_duplicates(key).set_index(key).value.rename("wage")
    e = d[base & (d.indicator_code == "Y48423005")].drop_duplicates(key).set_index(key).value.rename("emp")
    m = pd.concat([w, e], axis=1).dropna(subset=["wage"]).reset_index()
    m["emp"] = m.emp.fillna(1.0)
    per_year = m.groupby(["territory_id", "year"]).apply(lambda g: np.average(g.wage, weights=g.emp),
                                                         include_groups=False)
    return per_year.groupby("territory_id").mean()


PASSPORT = {   # код: (название, период, разрез, фильтр нужного разреза)
    "Y48423007": ("Среднемесячная заработная плата работников организаций (без СМП)", "Январь-декабрь",
                  "всего по ОКВЭД2", {"okved2": TOTAL_OKVED}),
    "Y48423005": ("Среднесписочная численность работников организаций (без СМП)", "Январь-декабрь",
                  "всего (разделы — для структуры)", {"okved2": TOTAL_OKVED}),
    "Y48423006": ("Фонд заработной платы всех работников организаций (без СМП)", "Январь-декабрь",
                  "всего по ОКВЭД2", {"okved2": TOTAL_OKVED}),
    "Y48112027": ("Оценка численности населения на 1 января", "На 1 января", "все население", {"mest": "Все население"}),
    "Y48112014": ("Численность населения по полу и возрасту на 1 января", "На 1 января",
                  "оба пола; моложе / трудоспособный / старше", {"grup_2": "Всего", "vozr": "Всего"}),
    "Y48112023": ("Миграционный прирост (убыль)", "Значение показателя за год", "миграция — всего, все возрасты, оба пола",
                  {"migr": "Миграция — всего", "vozr": "Всего", "grup_2": "Всего"}),
    "Y48002001": ("Число объектов розничной торговли и общественного питания", "квартал",
                  "магазины всего (строка 01) и взаимно исключающие категории", {"obroz": "Магазины"}),
}


def passport_years(code: str, cfg: dict) -> list[int]:
    yrs = list(cfg["labor"]["years"])
    if code == "Y48112023":
        return list(cfg["labor"]["migration_years"])
    if code == "Y48112027":            # среднегодовая численность лет окна и миграции: 1 января от первого года до года после окна
        return list(range(min(cfg["labor"]["migration_years"]), max(yrs) + 2))
    return yrs


def data_passport(d: pd.DataFrame, ids, cfg: dict) -> pd.DataFrame:
    """Паспорт показателей: покрытие нужного разреза по годам (а не наличие хотя бы одной строки)."""
    rows = []
    ids = set(ids)
    for code, (name, period, cut, flt) in PASSPORT.items():
        s = d[d.indicator_code == code]
        s = s[s.indicator_period.astype(str).str.contains(period.split()[0])]
        for col, val in flt.items():
            s = s[s[col] == val]
        yrs = passport_years(code, cfg)
        have = {y: set(s.loc[s.year == y, "territory_id"]) & ids for y in yrs}
        row = {"код": code, "показатель": name, "единица": ", ".join(map(str, s.indicator_unit.unique()[:2])),
               "период": period, "разрез": cut, "годы": "–".join(map(str, (min(yrs), max(yrs))))}
        for y in yrs:
            row[f"МО: {y}"] = len(have[y])
        row["МО: все годы"] = len(set.intersection(*have.values()))
        row["МО: хотя бы один год"] = len(set.union(*have.values()))
        rows.append(row)
    return pd.DataFrame(rows)


RETAIL_SHOPS_TOTAL = "Магазины"                    # строка 01 формы 1-ТОРГ (МО): магазины всего
RETAIL_SHOP_SUBTYPES = ["Супермаркеты", "Гипермаркеты", "Минимаркеты", "Универмаги",
                        "Специализированные продовольственные магазины",
                        "Специализированные непродовольственные магазины", "Прочие магазины"]   # составляющие строки 01
RETAIL_MODERN = ["Супермаркеты", "Гипермаркеты", "Минимаркеты"]


def _per_year(d: pd.DataFrame, code: str, period: str, years, extra: dict | None = None) -> pd.DataFrame:
    """МО × год: сумма по ОКТМО одного МО в году (слияния); пропуск остаётся пропуском."""
    s = d[(d.indicator_code == code) & (d.indicator_period == period) & d.year.isin(years)]
    for col, val in (extra or {}).items():
        s = s[s[col] == val]
    s = s.drop_duplicates(["territory_id", "year", "oktmo"] + list(extra or {}))
    return s.groupby(["territory_id", "year"]).value.sum().unstack("year").reindex(columns=list(years))


def annual_mean_population(jan1: pd.DataFrame, years) -> tuple[pd.DataFrame, pd.Series]:
    """Среднегодовая численность года y — среднее численности на 1 января годов y и y + 1 (Росстат).
    Если одной из точек нет, берётся имеющаяся (флаг «приближённо»). Возвращает МО × год и флаг."""
    out, approx = {}, pd.Series(False, index=jan1.index)
    for y in years:
        a, b = jan1.get(y), jan1.get(y + 1)
        both = a.notna() & b.notna()
        out[y] = np.where(both, (a + b) / 2, a.fillna(b))
        approx |= ~both & (a.notna() | b.notna())
    return pd.DataFrame(out, index=jan1.index), approx


def _mean_ratio(num: pd.DataFrame, den: pd.DataFrame) -> pd.Series:
    """Среднее по годам отношения показателя к знаменателю того же года (только годы, где есть оба)."""
    r = num / den.reindex(index=num.index, columns=num.columns)
    return r.mean(axis=1, skipna=True)


def labor_table(cfg: dict, d: pd.DataFrame | None = None, diagnostics: dict | None = None) -> pd.DataFrame:
    """Таблица рынка труда по МО. Все относительные показатели считаются внутри года (числитель и
    знаменатель одного года), затем усредняются по годам: при разном покрытии лет иначе возникает
    несогласованная композиция (например, отраслевые доли в сумме больше 1)."""
    d = load_bdmo(cfg) if d is None else d
    yrs = list(cfg["labor"]["years"])
    diag = diagnostics if diagnostics is not None else {}
    jan1 = _per_year(d, "Y48112027", "На 1 января", sorted(set(yrs) | {max(yrs) + 1} | {min(cfg["labor"]["migration_years"])}),
                     {"mest": "Все население"})
    jan1 = jan1.where(jan1 > 0)
    pop_y, pop_approx = annual_mean_population(jan1, yrs)
    emp_y = _per_year(d, "Y48423005", "Январь-декабрь", yrs, {"okved2": TOTAL_OKVED})
    fund_y = _per_year(d, "Y48423006", "Январь-декабрь", yrs, {"okved2": TOTAL_OKVED})
    idx = pop_y.index.union(emp_y.index).union(fund_y.index)
    pop_y, emp_y, fund_y = (x.reindex(idx) for x in (pop_y, emp_y, fund_y))
    out = pd.DataFrame(index=idx)
    out["population"] = pop_y.mean(axis=1)                       # средняя из среднегодовых 2023 и 2024
    out["population_approx"] = pop_approx.reindex(idx).fillna(False)
    out["employees"] = emp_y.mean(axis=1)
    out["fund_thrub"] = fund_y.mean(axis=1)
    out["wage"] = _weighted_wage(d, yrs)
    out["emp_rate"] = _mean_ratio(emp_y, pop_y)                  # работников на жителя, внутри года
    out["fund_pc"] = _mean_ratio(fund_y * 1000 / 12, pop_y)      # руб. в месяц на жителя, внутри года
    out["years_emp_pop"] = (emp_y.notna() & pop_y.notna()).sum(axis=1)

    # городское население: ноль — только структурный (сельское = всё население), иначе «нет данных»
    urb = _per_year(d, "Y48112027", "На 1 января", sorted(set(yrs) | {max(yrs) + 1}), {"mest": "Городское население"}).reindex(idx)
    rur = _per_year(d, "Y48112027", "На 1 января", sorted(set(yrs) | {max(yrs) + 1}), {"mest": "Сельское население"}).reindex(idx)
    tot = jan1.reindex(index=idx, columns=urb.columns)
    structural0 = urb.isna() & rur.notna() & tot.notna() & ((rur - tot).abs() <= 0.5)
    urb_f = urb.where(urb.notna(), np.where(structural0, 0.0, np.nan))
    share_u = (urb_f / tot)
    diag["доля горожан > 1 (до обрезки)"] = int((share_u > 1 + 1e-9).any(axis=1).sum())
    out["share_urban"] = share_u.mean(axis=1).clip(upper=1)
    out["urban_status"] = np.select([urb.notna().any(axis=1), structural0.any(axis=1)],
                                    ["наблюдается", "структурный ноль"], "нет данных")

    # возрастная структура: доли внутри года, затем среднее
    age = d[(d.indicator_code == "Y48112014") & (d.grup_2 == "Всего") & (d.indicator_period == "На 1 января")
            & d.year.isin(yrs)]
    age = age.drop_duplicates(["territory_id", "year", "oktmo", "vozr"]).groupby(["territory_id", "year", "vozr"]).value.sum().unstack("vozr")
    groups3 = ["Моложе трудоспособного возраста", "Трудоспособный возраст", "Старше трудоспособного возраста"]
    if all(g in age for g in groups3):
        # знаменатель — сумма трёх взаимно исключающих групп (строка «Всего» в источнике бывает несогласована);
        # год отбрасывается, если сумма групп расходится с оценкой численности на ту же дату больше чем на 5%
        asum = age[groups3].sum(axis=1, min_count=3)
        pop_same = jan1.stack().reindex(asum.index)
        bad = (asum / pop_same - 1).abs() > 0.05
        diag["МО-лет: возрастные группы расходятся с численностью > 5% (год отброшен)"] = int(bad.sum())
        if "Всего" in age:
            diag["МО-лет: строка «Всего» возраста ≠ сумме групп более чем на 1%"] = int(((age["Всего"] / asum - 1).abs() > 0.01).sum())
        ok = ~bad & asum.notna()
        for col, name in (("share_old", "Старше трудоспособного возраста"), ("share_young", "Моложе трудоспособного возраста")):
            out[col] = (age[name] / asum)[ok].groupby(level=0).mean().reindex(idx)
    else:
        out["share_old"] = out["share_young"] = np.nan

    # миграция: прирост года y к среднегодовой численности того же года
    mig_years = list(cfg["labor"]["migration_years"])
    mig_y = _per_year(d, "Y48112023", "Значение показателя за год", mig_years,
                      {"migr": "Миграция — всего", "vozr": "Всего", "grup_2": "Всего"}).reindex(idx)
    pop_m, _ = annual_mean_population(jan1.reindex(idx), mig_years)
    out["mig_rate"] = _mean_ratio(mig_y, pop_m) * 1000

    # отраслевая структура: внутри года (раздел засекречен → в «не распределено» этого года), затем
    # объединение лет: Σ_y раздел / Σ_y всего по годам, где опубликован итог
    sec = d[(d.indicator_code == "Y48423005") & (d.indicator_period == "Январь-декабрь") & d.year.isin(yrs)].copy()
    sec["letter"] = sec.okved2.map(section_letter)
    sec = sec[sec.letter.notna()].drop_duplicates(["territory_id", "year", "oktmo", "okved2"])
    grp = {l: name for name, letters in SECTOR_GROUPS.items() for l in letters}
    sec["group"] = sec.letter.map(grp)
    sec = sec[sec.group.notna()]
    sec_y = sec.groupby(["territory_id", "year", "group"]).value.sum().unstack("group").reindex(columns=list(SECTOR_GROUPS))
    tot_long = emp_y.stack().dropna().rename("total")
    sec_y = sec_y.reindex(tot_long.index)                        # только МО-годы с опубликованным итогом
    known_y = sec_y.sum(axis=1, min_count=1).fillna(0)
    # разделы и итог округлены до человека независимо: превышение итога не более чем на 0,5 человека
    # на каждый опубликованный раздел — округление; больше — нарушение баланса (выводится отдельно)
    n_sec = sec.groupby(["territory_id", "year"]).okved2.nunique().reindex(tot_long.index).fillna(0)
    excess = known_y - tot_long
    rounding = (excess > 0) & (excess <= 0.5 * n_sec)
    over = excess > 0.5 * n_sec
    diag["МО-лет: сумма разделов больше итога в пределах округления"] = int(rounding.sum())
    diag["МО-лет: сумма разделов больше итога сверх округления"] = int(over.sum())
    if over.any():
        diag["нарушения баланса разделов"] = pd.concat([known_y[over].rename("сумма разделов"), tot_long[over],
                                                      n_sec[over].rename("разделов")], axis=1)
    sec_tot = sec_y.fillna(0).groupby(level=0).sum()
    den = tot_long.groupby(level=0).sum()
    for name in SECTOR_GROUPS:
        out[f"emp_{name}"] = (sec_tot[name] / den).reindex(idx)
    known = out[[f"emp_{n}" for n in SECTOR_GROUPS]].sum(axis=1, min_count=1)
    diag["МО: сумма долей разделов больше 1 (до правила округления)"] = int((known > 1 + 1e-9).sum())
    diag["МО: максимальная сумма долей разделов"] = round(float(known.max()), 6)
    # засекреченные (малые) разделы и прочее; отрицательный остаток возможен только из-за округления
    out["emp_не распределено"] = (1 - known).clip(lower=0)

    # розничная инфраструктура: взаимно исключающие категории («Магазины» — итог строки 01; его
    # подвиды в общее число не добавляются); доля современных форматов — подвиды / магазины всего
    ret = d[(d.indicator_code == "Y48002001") & d.year.isin(yrs) & d.indicator_period.str.contains("квартал")]
    ret = ret.drop_duplicates(["territory_id", "year", "indicator_period", "oktmo", "obroz"])
    q = ret.groupby(["territory_id", "year", "indicator_period", "obroz"], observed=True).value.sum().unstack("obroz")
    if q.empty:
        out["retail_per_1000"] = out["modern_retail_share"] = np.nan
        return out
    q.columns = q.columns.astype(str)
    subs = [c for c in RETAIL_SHOP_SUBTYPES if c in q]
    shops = q[RETAIL_SHOPS_TOTAL] if RETAIL_SHOPS_TOTAL in q else pd.Series(np.nan, index=q.index)
    from_sub = q[subs].sum(axis=1, min_count=1)
    rebuilt = shops.isna() & from_sub.notna()
    diag["кварталов: магазины всего восстановлены из подвидов"] = int(rebuilt.sum())
    shops = shops.fillna(from_sub)
    exclusive = [c for c in q.columns if c not in subs and c != RETAIL_SHOPS_TOTAL]
    objects = q[exclusive].sum(axis=1, min_count=1).fillna(0) + shops.fillna(0)
    objects = objects.where(q.notna().any(axis=1))
    modern = q[[c for c in RETAIL_MODERN if c in q]].sum(axis=1, min_count=1)
    out["retail_per_1000"] = (objects.groupby(level=0).mean() / out.population * 1000).reindex(idx)
    out["modern_retail_share"] = (modern / shops.where(shops > 0)).groupby(level=0).mean().reindex(idx)
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


def bh_qvalues(p: np.ndarray, dependent: bool = False) -> np.ndarray:
    """q-значения Бенджамини–Хохберга; dependent=True — поправка Бенджамини–Иекутиели (1995/2001),
    контролирующая FDR при произвольной зависимости тестов (q умножаются на Σ 1/i)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    c = np.sum(1.0 / np.arange(1, n + 1)) if dependent else 1.0
    q = p[order] * n * c / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def _lisa_exceed(z, W, I, perms: int, rng) -> np.ndarray:
    """Число перестановок, где I_i не менее экстремален в направлении знака наблюдаемого I_i.
    Условная перестановка без возвращения: k_i разных МО из n − 1 остальных; один набор
    перестановок на все МО (как в PySAL), индекс i пропускается сдвигом."""
    n = len(z)
    kmax = int(np.diff(W.indptr).max())
    draws = np.stack([rng.choice(n - 1, size=kmax, replace=False) for _ in range(perms)])
    cnt = np.zeros(n)
    for i in range(n):
        lo, hi = W.indptr[i], W.indptr[i + 1]
        k = hi - lo
        if k == 0:
            continue
        idx = draws[:, :k].copy()
        idx[idx >= i] += 1
        Ip = z[i] * (z[idx] * W.data[lo:hi]).sum(1)
        cnt[i] = (Ip >= I[i]).sum() if I[i] >= 0 else (Ip <= I[i]).sum()
    return cnt


def local_moran(A, x: np.ndarray, perms: int = 499, seed: int = 0, alpha: float = 0.05,
                fdr: bool = True, seeds: list[int] | None = None) -> pd.DataFrame:
    """Локальный I Морана (Anselin, 1995), перестановочный тест.

    Альтернатива — односторонняя в направлении знака наблюдаемого I_i (положительный — скопление
    похожих, отрицательный — выброс). p = (1 + число перестановок не менее экстремальных) / (B + 1).
    При нескольких seed перестановки объединяются (B = perms × число seed) — это уменьшает ошибку
    Монте-Карло; дополнительно для каждого seed отдельно считается метка, и сообщается доля seed,
    в которых метка МО совпадает с итоговой (устойчивость включения).
    Семейство проверок — МО с хотя бы одним соседом. Основная поправка — Бенджамини–Хохберга,
    чувствительность — Бенджамини–Иекутиели (произвольная зависимость тестов)."""
    import scipy.sparse as sp
    A = sp.csr_matrix(A, dtype=float)
    n = len(x)
    z = (x - x.mean()) / x.std()
    deg = np.asarray(A.sum(1)).ravel()
    W = sp.csr_matrix(sp.diags(np.where(deg > 0, 1 / np.maximum(deg, 1e-12), 0)) @ A)
    W.sort_indices()
    lag = W @ z
    I = z * lag
    tested = deg > 0
    quad = np.where(z > 0, np.where(lag > 0, "HH", "HL"), np.where(lag > 0, "LH", "LL"))
    seeds = list(seeds) if seeds is not None else [seed]
    per_seed = [_lisa_exceed(z, W, I, perms, np.random.default_rng(s)) for s in seeds]

    def label(p, dependent=False):
        q = np.ones(n)
        q[tested] = bh_qvalues(p[tested], dependent) if fdr else p[tested]
        return q, np.where((q < alpha) & tested, quad, "не значимо")

    p = np.where(tested, (1 + np.sum(per_seed, axis=0)) / (perms * len(seeds) + 1), 1.0)
    q, cluster = label(p)
    q_by, cluster_by = label(p, dependent=True)
    labels_s = np.stack([label(np.where(tested, (1 + c) / (perms + 1), 1.0))[1] for c in per_seed])
    agree = (labels_s == cluster[None, :]).mean(axis=0)
    out = pd.DataFrame({"I": I, "p": p, "q": q, "q_by": q_by, "quadrant": quad, "cluster": cluster,
                        "cluster_by": cluster_by, "seed_agreement": agree})
    for j, s in enumerate(seeds):
        out[f"cluster_seed{s}"] = labels_s[j]
    return out


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
