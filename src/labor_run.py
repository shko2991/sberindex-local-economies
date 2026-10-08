"""Потребление против рынка труда: типология рынка труда, согласие с типами потребления,
карта расхождений и их объяснение.

  python src/labor_run.py      (после pipeline.py)

1. Типология рынка труда — тот же конвейер выбора, что и для потребления: k-means, Уорд, GMM и
   KEFRiN на сети дорог, K = 4…12, шесть ICVI на той же сети, Борда и Коупленд, устойчивость.
   Сеть — только дороги: слои динамики построены из расходов и внесли бы потребление в типологию труда.
2. Согласие типологий: ARI, AMI, V Крамера, таблица сопряжённости.
3. Расхождение: (а) разрыв = log(безналичные расходы на жителя / фонд оплаты труда на жителя) −
   медиана; (б) остаток регрессии log расходов на признаки рынка труда; локальный I Морана
   по остатку — пространственные скопления «тратят больше / меньше, чем зарабатывают».
4. Объяснение разрыва: доля старше трудоспособного возраста (пенсии), доступность рынков,
   расстояние до регионального центра (маятниковая миграция), Север, торговая инфраструктура.
Внутригородские территории городов федерального значения считаются отдельно: занятость и фонд
оплаты учитываются по месту работы, и для центральных районов они кратно превышают население.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import get_logger, load_config, path
from compare import build_context, stability
from data import load_market_access
from icvi import aggregate, all_indices, rank_table, kendall_w
from labor import (distance_to_capital, impute_demography, labor_features, labor_table, local_moran, ols)
from methods import Context, relabel_by_size, run
from reference import load_reference
from validate import partition_agreement

log = get_logger("labor_run")


def labor_candidates(cfg):
    out = []
    for k in cfg["methods"]["k_range"]:
        for m in ("kmeans", "ward", "gmm"):
            out.append({"method": m, "k": k})
        for d, xm in (("cosine", 1), ("cosine", 4), ("euclidean", 1)):
            out.append({"method": "kefrin", "network": "geography", "k": k, "distance": d, "xi_mult": xm})
    return out


def cid(c):
    return "|".join([c["method"]] + ([c["network"], c["distance"], f"xi{c['xi_mult']}"] if c["method"] == "kefrin" else [])
                    + [f"k{c['k']}"])


def select_typology(cfg, X: np.ndarray, A, seed: int):
    ctx = Context(X=X, graphs={"geography": A}, cfg=cfg)
    rows, labels = [], {}
    for i, c in enumerate(labor_candidates(cfg)):
        kw = {"distance": c["distance"], "xi_mult": c["xi_mult"]} if c["method"] == "kefrin" else {}
        lab = relabel_by_size(run(c["method"], ctx, k=c["k"], network=c.get("network"), seed=seed + i, **kw))
        labels[cid(c)] = lab
        rows.append({"id": cid(c), **c, **all_indices(X, A, lab)})
    df = pd.DataFrame(rows).set_index("id")
    n = len(X)
    ev = cfg["evaluation"]
    df["admissible"] = (df.min_size >= ev["min_cluster_share"] * n) & (df.max_share <= ev["max_cluster_share"])
    ranked = aggregate(df[df.admissible], ev["icvi"])
    W = kendall_w(rank_table(ranked, ev["icvi"]))
    by_id = {cid(c): c for c in labor_candidates(cfg)}
    stab = {}
    for j, i in enumerate(ranked.index[:cfg["labor"]["bootstrap_top"]]):
        stab[i] = stability(cfg, ctx, by_id[i], labels[i], cfg["labor"]["bootstrap"], ev["subsample"], 2000 + j)
        log.info("труд: устойчивость %s ARI=%.3f", i, stab[i]["ARI_mean"])
    ranked = ranked.join(pd.DataFrame(stab).T)
    ok = ranked[ranked.ARI_mean >= ev["stability_min"]]
    best = (ok if len(ok) else ranked).index[0]
    return best, labels[best], ranked, W


def main():
    cfg = load_config()
    panel, feat, ctx = build_context(cfg)
    ids = feat.X.index
    ref = load_reference(cfg).loc[ids]
    fin = pd.read_csv(path(cfg, "processed", "final_types.csv"), index_col=0)
    cons_type = fin.type.reindex(ids)
    lt = impute_demography(labor_table(cfg), ref)
    lt.to_csv(path(cfg, "processed", "labor_table.csv"))
    out = {"n_with_labor": int(lt.reindex(ids).wage.notna().sum()),
           "imputed_age": int(lt.reindex(ids).imputed_age.fillna(False).sum())}

    # 1. типология рынка труда
    F, blocks = labor_features(lt, ids)
    pos = pd.Series(np.arange(len(ids)), index=ids)
    sub = pos[F.index].values
    A = ctx.graphs["geography"][sub][:, sub]
    best, lab, ranked, W = select_typology(cfg, F.values, A, int(cfg["seed"]))
    ranked.to_csv(path(cfg, "processed", "labor_ranking.csv"))
    labor_type = pd.Series(lab, index=F.index, name="labor_type")
    out.update({"labor_chosen": best, "labor_K": int(labor_type.nunique()), "labor_W": W,
                "labor_n": int(len(F)), "labor_ARI": float(ranked.loc[best, "ARI_mean"])})
    prof = lt.reindex(F.index).groupby(labor_type)[["wage", "emp_rate", "fund_pc", "share_old", "share_young",
                                                     "mig_rate"] + [c for c in lt if c.startswith("emp_") and c != "emp_rate"]].median()
    prof.insert(0, "n", labor_type.value_counts().sort_index())
    prof["share_federal_city"] = ref.federal_city.reindex(F.index).groupby(labor_type).mean()
    prof["n_regions"] = ref.region_name.reindex(F.index).groupby(labor_type).nunique()
    prof.to_csv(path(cfg, "processed", "labor_profiles.csv"))

    # 2. согласие типологий
    both = pd.DataFrame({"cons": cons_type, "labor": labor_type}).dropna()
    out["agreement_all"] = partition_agreement(both.cons, both.labor)
    nf = both[~ref.federal_city.reindex(both.index).values]
    out["agreement_no_federal"] = partition_agreement(nf.cons, nf.labor)
    pd.crosstab(both.cons, both.labor).to_csv(path(cfg, "processed", "cons_vs_labor.csv"))
    pd.crosstab(nf.cons, nf.labor).to_csv(path(cfg, "processed", "cons_vs_labor_no_federal.csv"))

    # 3. расхождение
    spend = panel.values[cfg["features"]["total"]].mean(axis=1)
    t = lt.reindex(ids)
    fed = ref.federal_city.values
    gap = np.log(spend / t.fund_pc)
    gap = gap - gap[~fed].median()
    Xl = pd.DataFrame({"log_wage": np.log(t.wage), "log_emp_rate": np.log(t.emp_rate), "share_old": t.share_old,
                       "share_young": t.share_young, "mig_rate": t.mig_rate.clip(*t.mig_rate.quantile([.01, .99]))},
                      index=ids)
    for c in [c for c in t if c.startswith("emp_") and c != "emp_rate"]:
        Xl[c] = t[c]
    Xl = Xl.drop(columns=["emp_не распределено"])          # сумма долей = 1: одна группа — база
    region = ref.region_name
    m1 = ols(np.log(spend)[~fed], Xl[~fed], standardize=True, groups=region[~fed])
    resid = pd.Series(np.nan, index=ids)
    resid.loc[m1.resid.index] = m1.resid
    out["labor_model_R2"] = float(m1.rsquared)
    out["labor_model_coef"] = {k: round(float(v), 3) for k, v in m1.params.items()}
    # локальные скопления остатка (на сети дорог, только МО с остатком)
    ok = resid.notna().values
    lm = local_moran(ctx.graphs["geography"][ok][:, ok], resid[ok].values, perms=cfg["labor"]["lisa_perms"],
                     seed=int(cfg["seed"]))
    lisa = pd.Series("нет данных", index=ids)
    lisa.loc[resid.index[ok]] = lm.cluster.values
    out["lisa_counts"] = lisa.value_counts().to_dict()
    out["lisa_counts_no_fdr"] = pd.Series(np.where(lm.p < 0.05, lm.quadrant, "не значимо")).value_counts().to_dict()

    # 4. объяснение разрыва
    ma = load_market_access(cfg).reindex(ids)
    dist = distance_to_capital(cfg, load_reference(cfg), ids)
    Xe = pd.DataFrame({
        "доля старше трудоспособного": t.share_old,
        "доля занятых в бюджетном секторе": t["emp_бюджетный сектор (O, P, Q)"],
        "log доступности рынков": np.log1p(ma),
        "log расстояния до центра региона": np.log1p(dist.reindex(ids)),
        "Север (широта ≥ 60°)": (ref.lat >= 60).astype(float),
        "городской округ": (ref.mo_type != "муниципальный район").astype(float),
        "миграционный прирост на 1000": Xl.mig_rate,
        "объектов торговли на 1000": t.retail_per_1000,
        "доля современных форматов": t.modern_retail_share}, index=ids)
    m2 = ols(gap[~fed], Xe[~fed], standardize=True, groups=region[~fed])
    out["gap_model_R2"] = float(m2.rsquared)
    out["gap_model_n"] = int(m2.nobs)
    coef = pd.DataFrame({"коэф. (станд.)": m2.params, "p": m2.pvalues}).drop(index="const")
    coef.round(4).to_csv(path(cfg, "processed", "gap_model.csv"))
    gap_by_type = pd.DataFrame({"gap": gap, "type": cons_type})[~fed].groupby("type").gap.median()
    out["gap_by_cons_type"] = gap_by_type.round(3).to_dict()

    # 5. внешняя проверка типов потребления независимыми данными Росстата (без городов фед. значения)
    from validate import eta_squared
    L = pd.read_parquet(path(cfg, "processed", "labels_candidates.parquet")).reindex(ids)
    bench = {"итоговое разбиение": cons_type,
             "k-means по признакам, K=8": L["kmeans|k8"],
             "Leiden только по дорогам": L["leiden|geography|r0.2"]}
    ext = {"log зарплаты": np.log(t.wage), "log работников на жителя": np.log(t.emp_rate),
           "доля старше трудоспособного": t.share_old, "log фонда оплаты на жителя": np.log(t.fund_pc),
           "миграционный прирост": Xl.mig_rate, "индекс доступности рынков": ma}
    rows = []
    for vname, v in ext.items():
        for pname, part in bench.items():
            e_all = eta_squared(v, part)[0]
            e_nf = eta_squared(v[~fed], part[~fed])[0]
            rows.append({"переменная": vname, "разбиение": pname, "η² все МО": round(e_all, 3),
                         "η² без городов фед. значения": round(e_nf, 3)})
    pd.DataFrame(rows).to_csv(path(cfg, "processed", "external_validation_rosstat.csv"), index=False)

    res = pd.DataFrame({"labor_type": labor_type.reindex(ids), "cons_type": cons_type, "spend": spend.round(0),
                        "wage": t.wage.round(0), "fund_pc": t.fund_pc.round(0), "emp_rate": t.emp_rate.round(3),
                        "share_old": t.share_old.round(3), "gap": gap.round(3), "resid": resid.round(3),
                        "lisa": lisa, "dist_capital_km": dist.reindex(ids).round(0), "federal_city": fed})
    res.to_csv(path(cfg, "processed", "labor_mismatch.csv"))
    with open(path(cfg, "processed", "labor_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    log.info("готово: %s", json.dumps(out, ensure_ascii=False, default=float)[:1500])


if __name__ == "__main__":
    main()
