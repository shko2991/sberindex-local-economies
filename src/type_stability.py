"""Устойчивость отдельных типов и надёжность типа каждого МО (для финалистов).

Для финалистов (лучшие варианты итогового семейства при K = 6, 7, 8) повторяются 50 подвыборок
по 80% МО — теперь с тем же числом инициализаций KEFRiN, что и в основном расчёте (20), чтобы
изменения разбиения не объяснялись недооптимизацией. Для каждого типа — максимальный коэффициент
Жаккара с кластерами подвыборки (Hennig, 2007). Ориентиры Хеннига: среднее ≤ 0,5 — тип
«растворяется»; 0,6–0,75 — есть структура, но сомнительная; ≥ 0,75 — устойчивый; ≥ 0,85 — очень
устойчивый.

Надёжность типа МО (только итоговый вариант): метки подвыборки сопоставляются с полными
венгерским алгоритмом по Жаккару; надёжность — доля подвыборок, где МО попал в свой тип.
Отдельно — устойчивость оптимизации: 10 запусков на полных данных с другими seed.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from common import get_logger, load_config, path, rng
from compare import build_context, cand_id, candidates, run_candidate
from dynamics import jaccard_match

log = get_logger("type_stability")


def main():
    cfg = load_config()
    _, feat, ctx = build_context(cfg)
    ids = feat.X.index
    L = pd.read_parquet(path(cfg, "processed", "labels_candidates.parquet")).reindex(ids)
    chosen = json.load(open(path(cfg, "processed", "summary.json"), encoding="utf-8"))["chosen"]
    fam = chosen.rsplit("|k", 1)[0]
    finalists = [f"{fam}|k{k}" for k in cfg["stability"]["finalist_k"]]
    by_id = {cand_id(c): c for c in candidates(cfg)}
    ev = cfg["evaluation"]
    B, frac, n = cfg["stability"]["bootstrap"], ev["subsample"], len(ids)
    rows_type, rows_fin = [], []
    reliab = None
    for fid in finalists:
        c = dict(by_id[fid])
        full = L[fid].values
        g = rng(cfg, 3000)                      # одинаковые подвыборки для всех финалистов
        jac = {k: [] for k in np.unique(full)}
        aris = []
        hit = np.zeros(n); seen = np.zeros(n)
        for b in range(B):
            idx = np.sort(g.choice(n, int(frac * n), replace=False))
            lab = run_candidate(ctx.subset(idx), c, seed=5000 + b, bootstrap=False)   # 20 инициализаций
            aris.append(adjusted_rand_score(full[idx], lab))
            for k in jac:
                a = full[idx] == k
                jac[k].append(max(np.sum(a & (lab == j)) / np.sum(a | (lab == j)) for j in np.unique(lab)))
            if fid == chosen:
                m = jaccard_match(full[idx], lab)
                hit[idx] += (m == full[idx]); seen[idx] += 1
            if (b + 1) % 10 == 0:
                log.info("%s: %d/%d подвыборок", fid, b + 1, B)
        for k, v in jac.items():
            rows_type.append({"вариант": fid, "K": int(c["k"]), "тип": int(k), "МО": int((full == k).sum()),
                              "Жаккар среднее": round(float(np.mean(v)), 3),
                              "Жаккар 10-й проц.": round(float(np.quantile(v, 0.1)), 3)})
        init = [adjusted_rand_score(full, run_candidate(ctx, c, seed=9000 + s)) for s in range(cfg["stability"]["init_runs"])]
        rows_fin.append({"вариант": fid, "K": int(c["k"]), "ARI среднее": round(float(np.mean(aris)), 3),
                         "ARI 10-й проц.": round(float(np.quantile(aris, 0.1)), 3),
                         "типов с Жаккаром ≥ 0,75": int(sum(np.mean(v) >= 0.75 for v in jac.values())),
                         "типов с Жаккаром < 0,5": int(sum(np.mean(v) < 0.5 for v in jac.values())),
                         "ARI с другими seed (мин)": round(float(np.min(init)), 3),
                         "ARI с другими seed (среднее)": round(float(np.mean(init)), 3)})
        if fid == chosen:
            reliab = pd.Series(np.where(seen > 0, hit / np.maximum(seen, 1), np.nan), index=ids, name="reliability")
        log.info("%s: ARI=%.3f", fid, np.mean(aris))
    pd.DataFrame(rows_type).to_csv(path(cfg, "processed", "type_stability.csv"), index=False)
    pd.DataFrame(rows_fin).to_csv(path(cfg, "processed", "finalists.csv"), index=False)
    reliab.to_csv(path(cfg, "processed", "mo_reliability.csv"))
    log.info("готово")


if __name__ == "__main__":
    main()
