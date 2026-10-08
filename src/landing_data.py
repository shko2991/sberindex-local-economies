"""Данные для интерактивной страницы: один JSON (landing/data.json) из результатов pipeline.py.

Состав: МО (название, регион, тип, типы по кварталам, траектория, ключевые показатели), типы
(профиль, описания АФП, доли по кварталам), сравнение методов, обоснование сети, переходы,
полигоны (TopoJSON). Числа округлены, чтобы страница оставалась лёгкой.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import get_logger, load_config, path, labor_available

log = get_logger("landing")

KEYS = ["share_Продовольствие", "share_Здоровье", "share_Маркетплейсы", "share_Общественное питание",
        "share_Транспорт", "share_Прочее", "level_total", "growth_rel", "summer_peak", "dec_peak", "volatility",
        "mp_shift"]


def r(x, n=3):
    return None if pd.isna(x) else round(float(x), n)


def build(cfg: dict, raw: pd.DataFrame, type_names: dict[int, str] | None = None) -> dict:
    P = lambda n: path(cfg, "processed", n)  # noqa: E731
    final = pd.read_csv(P("final_types.csv"), index_col=0)
    quarters = pd.read_csv(P("types_quarterly.csv"), index_col=0)
    prof = pd.read_csv(P("type_profiles.csv"), index_col=0)
    fca = pd.read_csv(P("fca_descriptions.csv"))
    cand = pd.read_csv(P("candidates.csv"), index_col=0)
    rank = pd.read_csv(P("ranking_main.csv"), index_col=0)
    summary = json.load(open(P("summary.json"), encoding="utf-8"))
    type_names = type_names or {}

    mo_ids = final.index.tolist()
    raw = raw.loc[mo_ids]
    mo = {"id": mo_ids, "name": final.name_short.tolist(), "full": final.name.tolist(),
          "region": final.region_name.tolist(), "type": final.type.astype(int).tolist(),
          "traj": final.trajectory.tolist(), "ma": [r(v, 0) for v in final.market_access],
          "q": quarters.loc[mo_ids].astype(int).values.tolist(),
          "f": [[r(v) for v in row] for row in raw[KEYS].values]}
    national = {k: r(raw[k].median()) for k in KEYS}
    q_share = quarters.apply(lambda c: c.value_counts(normalize=True)).fillna(0)
    types = []
    for t, row in prof.iterrows():
        t = int(t)
        types.append({"id": t, "name": type_names.get(t, f"Тип {t}"), "n": int(row.n),
                      "ma": r(row.market_access, 0), "city": r(row.share_city), "fed": r(row.share_federal_city),
                      "regions": int(row.n_regions), "profile": {k: r(row[k]) for k in KEYS},
                      "fca": fca[fca.type == t][["description", "precision", "coverage", "delta"]].to_dict("records"),
                      "quarters": [r(q_share.loc[t, q]) if t in q_share.index else 0 for q in quarters.columns]})
    cols = ["method", "network", "K", "SW", "CH", "S_Dbw", "AVI", "AVU", "MQ", "admissible"]
    methods = cand[cols].join(rank[["borda", "copeland", "borda_place", "copeland_place"] +
                                   [c for c in ("ARI_mean", "ARI_q10") if c in rank]], how="left")
    methods = methods.reset_index().rename(columns={"index": "id"})
    methods = json.loads(methods.round(4).to_json(orient="records", force_ascii=False))
    trans = pd.read_csv(P("transition_yearly.csv"), index_col=0)
    net = {"moran": json.loads(pd.read_csv(P("moran.csv"), index_col=0).round(3).to_json(force_ascii=False)),
           "overlap": json.loads(pd.read_csv(P("edge_overlap.csv"), index_col=0).round(3).to_json(force_ascii=False)),
           "community": json.loads(pd.read_csv(P("community_strength.csv")).to_json(orient="records", force_ascii=False))}
    topo = json.load(open(path(cfg, "interim", "mo.topo.json"), encoding="utf-8"))
    labor = None
    use_labor = labor_available(cfg)
    if use_labor and P("labor_mismatch.csv").exists():
        lm = pd.read_csv(P("labor_mismatch.csv"), index_col=0).reindex(mo_ids)
        lp = pd.read_csv(P("labor_profiles.csv"), index_col=0)
        ls = json.load(open(P("labor_summary.json"), encoding="utf-8"))
        ln = {int(k): v for k, v in (cfg.get("labor_type_names") or {}).items()}
        ct = pd.read_csv(P("cons_vs_labor_no_federal.csv"), index_col=0)   # та же выборка, что у V и AMI
        gm = pd.read_csv(P("gap_model.csv"), index_col=0)
        mo.update({"lt": [None if pd.isna(v) else int(v) for v in lm.labor_type],
                   "gap": [r(v) for v in lm.gap], "lisa": lm.lisa.fillna("нет данных").tolist(),
                   "spend": [r(v, 0) for v in lm.spend], "wage": [r(v, 0) for v in lm.wage],
                   "fund": [r(v, 0) for v in lm.fund_pc], "emp": [r(v) for v in lm.emp_rate],
                   "old": [r(v) for v in lm.share_old]})
        sec = [c for c in lp.columns if c.startswith("emp_") and c != "emp_rate"]
        labor = {"types": [{"id": int(t), "name": ln.get(int(t), f"Тип {int(t)}"), "n": int(row.n),
                            "wage": r(row.wage, 0), "emp": r(row.emp_rate), "fund": r(row.fund_pc, 0),
                            "old": r(row.share_old), "mig": r(row.mig_rate, 1), "fed": r(row.share_federal_city),
                            "regions": int(row.n_regions), "sectors": {c[4:]: r(row[c]) for c in sec}}
                           for t, row in lp.iterrows()],
                 "cross": {"rows": ct.index.astype(int).tolist(), "cols": [int(float(c)) for c in ct.columns],
                           "values": ct.values.astype(int).tolist()},
                 "gap_model": [{"name": k, "coef": r(v["коэф. (станд.)"]), "p": float(v.p)} for k, v in gm.iterrows()],
                 "summary": ls}
    # надёжность типа МО и устойчивость типов (type_stability.py), население типов (Росстат)
    if P("mo_reliability.csv").exists():
        rel = pd.read_csv(P("mo_reliability.csv"), index_col=0).reliability.reindex(mo_ids)
        mo["rel"] = [r(v, 2) for v in rel]
    if P("type_stability.csv").exists():
        ts = pd.read_csv(P("type_stability.csv"))
        ts = ts[ts["вариант"] == summary["chosen"]].set_index("тип")
        for t in types:
            if t["id"] in ts.index:
                t["jaccard"] = r(ts.loc[t["id"], "Жаккар среднее"])
                t["jaccard_q10"] = r(ts.loc[t["id"], "Жаккар 10-й проц."])
    if use_labor and P("labor_table.csv").exists():
        lt = pd.read_csv(P("labor_table.csv"), index_col=0).reindex(mo_ids)
        tser = pd.Series(mo["type"], index=mo_ids)
        for t in types:
            sel = tser[tser == t["id"]].index
            t["population"] = r(lt.population.reindex(sel).sum(), 0)
            if "share_urban" in lt:
                t["urban"] = r(lt.share_urban.reindex(sel).median())
    stability = None                 # ARI на подвыборках с тем же числом инициализаций, что в основном расчёте
    if P("finalists.csv").exists():
        fz = pd.read_csv(P("finalists.csv")).set_index("вариант")
        if summary["chosen"] in fz.index:
            stability = {"chosen": r(fz.loc[summary["chosen"], "ARI среднее"]),
                         "finalists": {int(fz.loc[i, "K"]): r(fz.loc[i, "ARI среднее"]) for i in fz.index}}
    cases = None
    if use_labor and P("cases.csv").exists():
        cases = json.loads(pd.read_csv(P("cases.csv")).round(3).to_json(orient="records", force_ascii=False))
    return {"meta": {"chosen": summary["chosen"], "n": len(mo_ids), "quarters": quarters.columns.tolist(),
                     "summary": summary, "stability": stability}, "keys": KEYS, "national": national, "types": types, "mo": mo,
            "methods": methods, "network": net,
            "transition": {"rows": trans.index.astype(int).tolist(), "cols": [int(c) for c in trans.columns],
                           "values": trans.values.astype(int).tolist()},
            "labor": labor, "cases": cases, "topo": topo}


def main():
    from compare import build_context
    from reference import export_topojson
    cfg = load_config()
    _, feat, _ = build_context(cfg)
    export_topojson(cfg)            # все МО справочника: вне выборки — серым
    names = cfg.get("type_names") or {}
    data = build(cfg, feat.raw, {int(k): v for k, v in names.items()})
    out = path(cfg, "processed", "landing_data.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    log.info("данные страницы: %s (%.1f МБ)", out, out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
