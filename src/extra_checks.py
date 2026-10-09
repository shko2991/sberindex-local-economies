"""Дополнительные проверки полезности типов (после рецензии): перенос, 6 против 8 типов, подбор аналогов.

1. Перенос на новые регионы. Регионы делятся на 5 частей (3 разных случайных деления). На четырёх
   частях заново строятся типы: KEFRiN на мультиплексе (подграф обучающих МО) и k-means по тем же
   признакам; масштаб признаков считается по обучающей части. МО отложенных регионов относятся к
   ближайшему центру типа и в построении не участвуют. Средние показателей Росстата по типам
   (обучающая часть) прогнозируют отложенные МО; мера — R² вне выборки. Ориентиры с тем же числом
   групп и тем же протоколом: размер МО (8 групп по численности), вид МО, доступность рынков
   (8 групп), широта (8 групп), а также «размер + вид» и «размер + вид + тип» (МНК с фиктивными
   переменными) — что тип добавляет к простому описанию территории.
   Ограничения: признаки МО (отклонение от медианы всех МО в том же месяце) и графы ближайших
   соседей построены по всем МО; на обучающую часть берётся подграф.
2. Шесть, семь или восемь типов: R² вне выборки из п. 1 и η² логарифма отношения «расходы ÷ фонд
   оплаты» внутри выборки (МО без городов федерального значения).
3. Подбор территорий для сравнения. Для каждого МО — 7 аналогов по правилу страницы (тот же тип
   потребления, ближайшие по 12 признакам) и 7 «двойников» обычного подбора по зарплате и численности
   (по всей стране и внутри региона). Качество подбора — насколько аналоги похожи на МО по
   показателям, которые не использовал ни один способ: доля старше трудоспособного возраста,
   миграция, доля горожан, доля занятых в O–Q, торговые объекты на 1 000 жителей.

Выход: data/processed/transfer_check.csv, k_choice_check.csv, analog_check.csv, extra_checks.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from common import get_logger, load_config, path
from compare import build_context, cand_id, candidates, run_candidate

log = get_logger("extra_checks")

OUTCOMES = {"log зарплаты": ("wage", True), "log фонда оплаты на жителя": ("fund_pc", True),
            "log работников на жителя": ("emp_rate", True), "доля старше трудоспособного": ("share_old", False),
            "миграционный прирост": ("mig_rate", False)}
OQ = "emp_госуправление, образование, здравоохранение (O, P, Q)"
THIRD = {"доля старше трудоспособного": "share_old", "миграционный прирост": "mig_rate",
         "доля горожан": "share_urban", "доля занятых в O–Q": OQ, "торговых объектов на 1 000": "retail_per_1000"}


def octiles(train: pd.Series, test: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """8 групп по квантилям обучающей части; пропуск — отдельная группа."""
    cuts = np.unique(np.nanquantile(train, np.linspace(0, 1, 9)[1:-1]))
    f = lambda s: np.where(s.isna(), -1, np.searchsorted(cuts, s.values, side="right"))
    return f(train), f(test)


def group_mean_pred(g_tr, y_tr, g_te) -> np.ndarray:
    m = pd.Series(y_tr).groupby(np.asarray(g_tr)).mean()
    return pd.Series(np.asarray(g_te)).map(m).fillna(float(np.mean(y_tr))).values


def ols_pred(G_tr: list, y_tr, G_te: list) -> np.ndarray:
    """МНК с фиктивными переменными нескольких группировок (уровни берутся по обучающей части)."""
    def design(G, levels):
        cols = [np.ones(len(G[0]))]
        for g, lv in zip(G, levels):
            g = np.asarray(g)
            cols += [(g == v).astype(float) for v in lv[1:]]
        return np.column_stack(cols)
    levels = [list(pd.unique(np.asarray(g))) for g in G_tr]
    b = np.linalg.lstsq(design(G_tr, levels), y_tr, rcond=None)[0]
    return design(G_te, levels) @ b


def scale_train(X: np.ndarray, tr: np.ndarray, te: np.ndarray, sd_full: np.ndarray):
    """Масштаб по обучающей части с сохранением весов блоков (стандартное отклонение столбца — как во всей выборке)."""
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
    k = sd_full / np.where(sd > 0, sd, 1)
    return (X[tr] - mu) * k, (X[te] - mu) * k


def nearest_center(Ytr, lab, Yte) -> np.ndarray:
    ks = np.unique(lab)
    C = np.stack([Ytr[lab == k].mean(axis=0) for k in ks])
    return ks[((Yte[:, None, :] - C[None]) ** 2).sum(axis=2).argmin(axis=1)]


def transfer(cfg, feat, ctx, base, lt, ft) -> pd.DataFrame:
    ec = cfg["extra_checks"]
    pos = pd.Series(np.arange(len(feat.X)), index=feat.X.index)
    ids = base.index
    regions = np.array(sorted(ft.loc[ids, "region_name"].unique()))
    by_id = {cand_id(c): c for c in candidates(cfg)}
    fam = json.load(open(path(cfg, "processed", "summary.json"), encoding="utf-8"))["chosen"].rsplit("|k", 1)[0]
    sd_full = ctx.X.std(axis=0)
    attrs = pd.DataFrame({"pop": lt.population.reindex(ids), "ma": ft.market_access.reindex(ids),
                          "lat": ft.lat.reindex(ids), "kind": ft.mo_type.reindex(ids)}, index=ids)
    Y = {name: (np.log(lt[col].reindex(ids)) if lg else lt[col].reindex(ids)) for name, (col, lg) in OUTCOMES.items()}
    rows = []
    for r in range(ec["repeats"]):
        g = np.random.default_rng(int(cfg["seed"]) + 7000 + r)
        fold_of = dict(zip(regions, g.permutation(len(regions)) % ec["folds"]))
        fold = ft.loc[ids, "region_name"].map(fold_of).values
        preds = {}                                   # (группировка, показатель) → прогноз для каждого МО
        for f in range(ec["folds"]):
            te_ids, tr_ids = ids[fold == f], ids[fold != f]
            tr, te = pos[tr_ids].values, pos[te_ids].values
            Ytr, Yte = scale_train(ctx.X, tr, te, sd_full)
            groups = {}
            for k in ec["k_values"]:
                sub = ctx.subset(tr)
                sub.X = Ytr
                lab = run_candidate(sub, by_id[f"{fam}|k{k}"], seed=int(cfg["seed"]) + 7100 + 10 * r + f)
                groups[f"KEFRiN, K = {k}"] = (lab, nearest_center(Ytr, lab, Yte))
            km = KMeans(8, n_init=20, random_state=int(cfg["seed"]) + 7200 + 10 * r + f).fit(Ytr)
            groups["k-means, K = 8"] = (km.labels_, km.predict(Yte))
            for name, col in (("размер МО (8 групп)", "pop"), ("доступность рынков (8 групп)", "ma"), ("широта (8 групп)", "lat")):
                groups[name] = octiles(attrs.loc[tr_ids, col], attrs.loc[te_ids, col])
            groups["вид МО"] = (attrs.loc[tr_ids, "kind"].values, attrs.loc[te_ids, "kind"].values)
            for oname, y in Y.items():
                ytr, ok_tr = y.loc[tr_ids].values, y.loc[tr_ids].notna().values
                ok_te = y.loc[te_ids].notna().values
                ybar = float(np.mean(ytr[ok_tr]))
                def put(gname, p):
                    preds.setdefault((gname, oname), []).append(
                        pd.DataFrame({"y": y.loc[te_ids].values[ok_te], "p": p[ok_te], "ybar": ybar}))
                for gname, (gtr, gte) in groups.items():
                    put(gname, group_mean_pred(np.asarray(gtr)[ok_tr], ytr[ok_tr], np.asarray(gte)))
                base_tr = [groups["размер МО (8 групп)"][0], groups["вид МО"][0]]
                base_te = [groups["размер МО (8 групп)"][1], groups["вид МО"][1]]
                put("размер + вид МО", ols_pred([np.asarray(a)[ok_tr] for a in base_tr], ytr[ok_tr], base_te))
                for extra in ("KEFRiN, K = 8", "k-means, K = 8"):
                    put(f"размер + вид МО + {extra}", ols_pred(
                        [np.asarray(a)[ok_tr] for a in base_tr + [groups[extra][0]]], ytr[ok_tr], base_te + [groups[extra][1]]))
            log.info("перенос: деление %d, часть %d — %d МО обучения, %d отложено", r + 1, f + 1, len(tr), len(te))
        for (gname, oname), parts in preds.items():
            d = pd.concat(parts)
            r2 = 1 - ((d.y - d.p) ** 2).sum() / ((d.y - d.ybar) ** 2).sum()
            rows.append({"деление": r + 1, "группировка": gname, "показатель": oname, "R2_вне_выборки": r2, "МО": len(d)})
    df = pd.DataFrame(rows)
    out = df.groupby(["группировка", "показатель"]).R2_вне_выборки.agg(["mean", "min", "max"]).reset_index()
    out.columns = ["группировка", "показатель", "R² вне выборки (среднее)", "мин", "макс"]
    return out.round(3)


def k_choice(cfg, base, lt, L, fam) -> pd.DataFrame:
    gap = np.log(base.spend / base.fund_pc).replace([np.inf, -np.inf], np.nan)
    rows = []
    for k in cfg["extra_checks"]["k_values"]:
        lab = L.loc[base.index, f"{fam}|k{k}"]
        for name, y in [("log(расходы ÷ фонд оплаты)", gap)] + \
                       [(n, np.log(lt[c].reindex(base.index)) if lg else lt[c].reindex(base.index)) for n, (c, lg) in OUTCOMES.items()]:
            ok = y.notna()
            yy, gg = y[ok], lab[ok]
            eta = ((yy.groupby(gg).transform("mean") - yy.mean()) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
            rows.append({"K": k, "показатель": name, "η² (без ГФЗ)": round(float(eta), 3)})
    return pd.DataFrame(rows)


def analogs(cfg, feat, base, lt, ft) -> tuple[pd.DataFrame, list]:
    n_an = cfg["extra_checks"]["n_analogs"]
    ids = base.index[base.wage.notna() & lt.population.reindex(base.index).notna()]
    Z = feat.X.reindex(ids).values
    typ = ft.type.reindex(ids).values
    reg = ft.region_name.reindex(ids).values
    w = np.log(lt.wage.reindex(ids).values); p = np.log(lt.population.reindex(ids).values)
    WP = np.column_stack([(w - w.mean()) / w.std(), (p - p.mean()) / p.std()])
    third = pd.DataFrame({k: lt[c].reindex(ids).values for k, c in THIRD.items()}, index=ids)
    sd = third.std()
    err = {"аналоги по потреблению (страница)": [], "зарплата и численность, вся страна": [],
           "зарплата и численность, свой регион": [], "аналоги по потреблению, свой регион": [],
           "тот же тип потребления + зарплата и численность, свой регион": []}
    picks = {}
    for i in range(len(ids)):
        others = np.arange(len(ids)) != i
        same = np.where((typ == typ[i]) & others)[0]
        a = same[np.argsort(((Z[same] - Z[i]) ** 2).sum(axis=1))[:n_an]]
        allc = np.where(others)[0]
        b = allc[np.argsort(((WP[allc] - WP[i]) ** 2).sum(axis=1))[:n_an]]
        inreg = np.where((reg == reg[i]) & others)[0]
        rest = np.where((reg != reg[i]) & others)[0]
        cand = np.concatenate([inreg[np.argsort(((WP[inreg] - WP[i]) ** 2).sum(axis=1))],
                               rest[np.argsort(((WP[rest] - WP[i]) ** 2).sum(axis=1))]])[:n_an]
        # те же правила, но сначала свой регион (если в нём не хватает МО — добираются из других)
        reg_first = lambda pool, D: np.concatenate([pool[reg[pool] == reg[i]][np.argsort(D[reg[pool] == reg[i]])],
                                                    pool[reg[pool] != reg[i]][np.argsort(D[reg[pool] != reg[i]])]])[:n_an]
        a_reg = reg_first(same, ((Z[same] - Z[i]) ** 2).sum(axis=1))
        joint = reg_first(same, ((WP[same] - WP[i]) ** 2).sum(axis=1))
        for key, sel in zip(err, (a, b, cand, a_reg, joint)):
            d = (third.iloc[sel].mean() - third.iloc[i]).abs() / sd
            err[key].append(d.values)
        picks[ids[i]] = (a, b)
    rows = []
    for key, v in err.items():
        M = pd.DataFrame(v, columns=list(THIRD))
        for col in M:
            rows.append({"способ подбора": key, "показатель": col, "медиана |отклонения| в ст. откл.": round(float(M[col].median()), 3)})
        rows.append({"способ подбора": key, "показатель": "среднее по пяти показателям",
                     "медиана |отклонения| в ст. откл.": round(float(M.median().mean()), 3)})
    # разобранные случаи: сколько аналогов из того же типа рынка труда и каково у них отношение «расходы ÷ фонд»
    cases = []
    gap = (base.spend / base.fund_pc).reindex(ids)
    lab_t = base.labor_type.reindex(ids)
    cs = pd.read_csv(path(cfg, "processed", "cases.csv"))
    for nm, rg in cs[["МО", "регион"]].drop_duplicates().itertuples(index=False):
        hit = ft.index[(ft.name == nm) & (ft.region_name == rg)]
        if len(hit) != 1 or hit[0] not in picks:
            continue
        tid = hit[0]
        i = ids.get_loc(tid); a, b = picks[tid]
        cases.append({"territory_id": int(tid), "name": ft.loc[tid, "name"], "gap": round(float(gap.iloc[i]), 2),
                      "аналоги: тот же тип труда": int((lab_t.iloc[a] == lab_t.iloc[i]).sum()),
                      "аналоги: отношение (мин–макс)": [round(float(gap.iloc[a].min()), 2), round(float(gap.iloc[a].max()), 2)],
                      "двойники по зарплате: тот же тип потребления": int((typ[b] == typ[i]).sum()),
                      "двойники по зарплате: отношение (мин–макс)": [round(float(gap.iloc[b].min()), 2), round(float(gap.iloc[b].max()), 2)],
                      "двойники по зарплате": [ft.loc[ids[j], "name"] for j in b[:3]]})
    # по всей выборке: у скольких МО «двойники по зарплате» в основном из другого типа потребления
    share_other = float(np.mean([(typ[b] != typ[i]).sum() >= 4 for i, (a, b) in enumerate(picks.values())]))
    return pd.DataFrame(rows), cases, share_other


def main():
    cfg = load_config()
    _, feat, ctx = build_context(cfg)
    lm = pd.read_csv(path(cfg, "processed", "labor_mismatch.csv"), index_col="territory_id")
    lt = pd.read_csv(path(cfg, "processed", "labor_table.csv"), index_col="territory_id")
    ft = pd.read_csv(path(cfg, "processed", "final_types.csv"), index_col="territory_id")
    L = pd.read_parquet(path(cfg, "processed", "labels_candidates.parquet"))
    fam = json.load(open(path(cfg, "processed", "summary.json"), encoding="utf-8"))["chosen"].rsplit("|k", 1)[0]
    base = lm[lm.index.isin(feat.X.index) & ~lm.federal_city.astype(bool)]
    tr = transfer(cfg, feat, ctx, base, lt, ft)
    tr.to_csv(path(cfg, "processed", "transfer_check.csv"), index=False)
    kc = k_choice(cfg, base, lt, L, fam)
    kc.to_csv(path(cfg, "processed", "k_choice_check.csv"), index=False)
    an, cases, share_other = analogs(cfg, feat, base, lt, ft)
    an.to_csv(path(cfg, "processed", "analog_check.csv"), index=False)
    json.dump({"МО без ГФЗ": int(len(base)), "доля МО, у которых ≥4 из 7 двойников по зарплате — из другого типа потребления":
               round(share_other, 3), "случаи": cases}, open(path(cfg, "processed", "extra_checks.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log.info("готово")


if __name__ == "__main__":
    main()
