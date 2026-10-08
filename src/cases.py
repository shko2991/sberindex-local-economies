"""Разобранные случаи расхождений: по одному МО на каждый вид расхождения.

Правило отбора (воспроизводимое, без ручного выбора):
  1) «тратят больше» — МО из скопления HH с наибольшим остатком модели «расходы ← рынок труда»;
  2) «тратят меньше» — МО из скопления LL с наименьшим остатком;
  3) «Север» — МО типа «Крайний Север» или «Северные ресурсные территории» с наименьшим отношением
     расходов к фонду оплаты труда;
  4) «пригород» — МО в пределах 60 км от центра региона с наибольшим отношением расходов к фонду
     оплаты труда (кандидат на маятниковую миграцию).
Если доступна надёжность типа (type_stability.py), берутся только МО с надёжностью ≥ 0,8.
Для каждого случая — показатели МО, медианы его типа потребления и средневзвешенные значения его
соседей по дорожной сети (с теми же весами, что в графе).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import get_logger, load_config, path
from compare import build_context

log = get_logger("cases")

METRICS = {"spend": "расходы жителя, руб./мес.", "fund_pc": "фонд оплаты труда на жителя, руб./мес.",
           "ratio": "расходы ÷ фонд оплаты", "wage": "средняя зарплата, руб.", "emp_rate": "работников на жителя",
           "share_old": "доля старше трудоспособного", "resid": "остаток модели (станд.)",
           "market_access": "доступность рынков", "dist_capital_km": "до центра региона, км"}


def main():
    cfg = load_config()
    _, feat, ctx = build_context(cfg)
    ids = feat.X.index
    m = pd.read_csv(path(cfg, "processed", "labor_mismatch.csv"), index_col=0).reindex(ids)
    fin = pd.read_csv(path(cfg, "processed", "final_types.csv"), index_col=0).reindex(ids)
    m["ratio"] = m.spend / m.fund_pc
    m["market_access"] = fin.market_access
    rel_path = path(cfg, "processed", "mo_reliability.csv")
    if rel_path.exists():
        m["reliability"] = pd.read_csv(rel_path, index_col=0).reliability.reindex(ids)
    else:
        m["reliability"] = 1.0
    ok = (~m.federal_city.astype(bool)) & (m.reliability >= 0.8)
    names = {int(k): v for k, v in (cfg.get("type_names") or {}).items()}
    pick = {}
    hh = m[ok & (m.lisa == "HH")]
    ll = m[ok & (m.lisa == "LL")]
    north = m[ok & fin.type.isin([5, 7])]
    sub = m[ok & (m.dist_capital_km <= 60) & (m.dist_capital_km > 0)]
    if len(hh):
        pick["тратят больше, чем объясняет рынок труда"] = hh.resid.idxmax()
    if len(ll):
        pick["тратят меньше, чем объясняет рынок труда"] = ll.resid.idxmin()
    if len(north):
        pick["Север: расходы много ниже фонда оплаты труда"] = north.ratio.idxmin()
    if len(sub):
        pick["пригород центра региона"] = sub.ratio.idxmax()
    A = ctx.graphs["geography"]
    pos = pd.Series(np.arange(len(ids)), index=ids)
    rows = []
    for case, tid in pick.items():
        i = pos[tid]
        nb = A.getrow(i)
        w = pd.Series(nb.data, index=ids[nb.indices])
        t = int(fin.type[tid])
        same_type = m[(fin.type == t) & ~m.federal_city.astype(bool)]
        for key, label in METRICS.items():
            vals = m.loc[w.index, key].astype(float)
            okv = vals.notna().values
            v_nb = np.average(vals[okv], weights=w.values[okv]) if okv.any() else np.nan   # по соседям с данными
            rows.append({"случай": case, "МО": fin.name[tid], "регион": fin.region_name[tid],
                         "тип потребления": names.get(t, t), "показатель": label,
                         "МО": fin.name[tid], "значение": m.loc[tid, key],
                         "медиана типа": same_type[key].median(), "соседи по дорогам": v_nb,
                         "надёжность типа": m.loc[tid, "reliability"]})
    out = pd.DataFrame(rows)
    out.to_csv(path(cfg, "processed", "cases.csv"), index=False)
    log.info("случаи: %s", {k: fin.name[v] for k, v in pick.items()})


if __name__ == "__main__":
    main()
