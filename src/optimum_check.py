"""Проверка оптимизации итогового KEFRiN и связь финалистов с итоговым разбиением.

1) Критерий KEFRiN (минимизируется) у итогового разбиения и у 10 перезапусков с другими seed
   (по n_init инициализаций, как в основном расчёте) → kefrin_restarts.csv.
2) Таблицы сопряжённости итоговых типов с финалистами K = 6 и K = 7 → finalists_crosstab.csv.

  python src/optimum_check.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from common import get_logger, load_config, path
from compare import build_context
from methods import KEFRiN, modularity_transform

log = get_logger("optimum")


def kefrin_model(ctx, network: str, k: int, seed: int, xi_mult: float = 1.0):
    """Тот же расчёт, что methods.kefrin (евклидов вариант с выравниванием разброса), но с доступом к критерию."""
    c = ctx.cfg["methods"]["kefrin"]
    P = modularity_transform(ctx.graph(network))
    Y = np.asarray(ctx.X, dtype=float)
    xi = c["rho_network"] * xi_mult * (Y ** 2).sum() / max((P ** 2).sum(), 1e-12)
    return KEFRiN(k, "euclidean", c["rho_feature"], xi, c["n_init"], c["max_iter"], seed), Y, P


def criterion(m: KEFRiN, Y, P, labels) -> float:
    Cy, Cp = m._centers(Y, labels), m._centers(P, labels)
    D = m._dist(Y, P, Cy, Cp, (Y ** 2).sum(1), (P ** 2).sum(1))
    return float(D[np.arange(len(Y)), labels].sum())


def main():
    cfg = load_config()
    _, feat, ctx = build_context(cfg)
    ids = feat.X.index
    chosen = json.load(open(path(cfg, "processed", "summary.json"), encoding="utf-8"))["chosen"]
    method, network, dist, xi, k = chosen.split("|")
    assert method == "kefrin" and dist == "euclidean", chosen
    k, xi_mult = int(k[1:]), float(xi[2:])
    fin = pd.read_csv(path(cfg, "processed", "final_types.csv"), index_col=0).type.reindex(ids).values
    L = pd.read_parquet(path(cfg, "processed", "labels_candidates.parquet")).reindex(ids)
    m, Y, P = kefrin_model(ctx, network, k, 0, xi_mult)
    # итоговые метки переупорядочены по размеру — для критерия порядок не важен
    rows = [{"запуск": "итог", "seed": None, "критерий": round(criterion(m, Y, P, fin), 2), "ARI с итогом": 1.0}]
    for s in range(cfg["stability"]["init_runs"]):
        mm, _, _ = kefrin_model(ctx, network, k, 9000 + s, xi_mult)
        lab = mm.fit_predict(Y, P)
        rows.append({"запуск": f"перезапуск {s + 1}", "seed": 9000 + s, "критерий": round(mm.criterion_, 2),
                     "ARI с итогом": round(adjusted_rand_score(fin, lab), 3)})
        log.info("seed %d: критерий %.2f, ARI %.3f", 9000 + s, mm.criterion_, rows[-1]["ARI с итогом"])
    out = pd.DataFrame(rows)
    out["seed"] = out.seed.astype("Int64")
    out.to_csv(path(cfg, "processed", "kefrin_restarts.csv"), index=False)
    fam = chosen.rsplit("|k", 1)[0]
    tabs = []
    for kk in cfg["stability"]["finalist_k"]:
        fid = f"{fam}|k{kk}"
        if fid == chosen or fid not in L:
            continue
        t = pd.crosstab(pd.Series(fin, name="итоговый тип"), pd.Series(L[fid].values, name="тип финалиста"))
        t = t.stack().rename("МО").reset_index()
        t.insert(0, "финалист", fid)
        tabs.append(t[t["МО"] > 0])
    pd.concat(tabs).to_csv(path(cfg, "processed", "finalists_crosstab.csv"), index=False)
    log.info("готово")


if __name__ == "__main__":
    main()
