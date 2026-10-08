"""Описательные расчёты, на которые ссылается отчёт: выборка и выпавшие МО, покрытие
дополнительных данных, закон Энгеля, корреляции относительных траекторий, исходная кластеризация паттернов
и проверка гипотезы «региональная столица опережает периферию».

  python src/descriptives.py   → data/processed/descriptives.json, excluded_profile.csv, leadlag_regions.csv

Опережение: для каждой пары «столица региона — другое МО региона» ищется лаг (−3…+3 мес.) с
максимальной корреляцией первых разностей относительных траекторий общих расходов. Если столица задаёт
динамику, лаг > 0 встречается чаще, чем лаг < 0 (критерий Уилкоксона по регионам). Контроль —
случайная «псевдостолица» из того же региона.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from common import get_logger, load_config, path, rng
from data import build_panel, excluded_profile, load_consumption, sample_overlap
from features import static_features
from methods import pattern_clusters
from reference import load_reference

log = get_logger("descriptives")


def best_lag(x: np.ndarray, y: np.ndarray, max_lag: int, diff: bool = True) -> int:
    if diff:
        x, y = np.diff(x), np.diff(y)
    T, best, lag = len(x), -np.inf, 0
    for L in range(-max_lag, max_lag + 1):
        a = x[:T - L] if L >= 0 else x[-L:]
        b = y[L:] if L >= 0 else y[:T + L]
        c = np.corrcoef(a, b)[0, 1]
        if c > best:
            best, lag = c, L
    return lag


def leadlag_test(cfg, feat, ref) -> tuple[dict, pd.DataFrame]:
    R = feat.residual[cfg["features"]["total"]]
    L = cfg["graphs"]["leadlag"]["max_lag"]
    caps = ref[ref.status == "административный_центр_субъекта"]
    g = rng(cfg, 7)
    rows = []
    for cap_id, cap in caps.iterrows():
        others = ref[(ref.region_name == cap.region_name) & (ref.index != cap_id)].index
        if len(others) < 5:
            continue
        lags = np.array([best_lag(R.loc[cap_id].values, R.loc[o].values, L) for o in others])
        ps = g.choice(others)
        lags_ps = np.array([best_lag(R.loc[ps].values, R.loc[o].values, L) for o in others if o != ps])
        rows.append({"регион": cap.region_name, "МО": len(others), "столица опережает": (lags > 0).mean(),
                     "столица отстаёт": (lags < 0).mean(), "синхронно": (lags == 0).mean(),
                     "псевдостолица опережает": (lags_ps > 0).mean(), "псевдостолица отстаёт": (lags_ps < 0).mean()})
    d = pd.DataFrame(rows)
    w = stats.wilcoxon(d["столица опережает"] - d["столица отстаёт"])
    w2 = stats.wilcoxon(d["псевдостолица опережает"] - d["псевдостолица отстаёт"])
    res = {"regions": int(len(d)), "pairs": int(d["МО"].sum()),
           "lead": round(float(d["столица опережает"].mean()), 3), "lag": round(float(d["столица отстаёт"].mean()), 3),
           "sync": round(float(d["синхронно"].mean()), 3), "wilcoxon_p": round(float(w.pvalue), 3),
           "pseudo_lead": round(float(d["псевдостолица опережает"].mean()), 3),
           "pseudo_lag": round(float(d["псевдостолица отстаёт"].mean()), 3), "pseudo_wilcoxon_p": round(float(w2.pvalue), 3)}
    return res, d


def main():
    cfg = load_config()
    df = load_consumption(cfg)
    panel = build_panel(cfg, df)
    feat = static_features(panel, cfg)
    ref = load_reference(cfg).loc[feat.X.index]
    out = {}
    ex = excluded_profile(cfg, df, panel)
    ex.to_csv(path(cfg, "processed", "excluded_profile.csv"))
    out["coverage"] = sample_overlap(cfg, panel).to_dict("records")
    r = feat.raw
    out["engel"] = {"level_vs_food_share": round(float(r.level_total.corr(r["share_Продовольствие"])), 3),
                    "level_vs_catering_share": round(float(r.level_total.corr(r["share_Общественное питание"])), 3)}
    from features import comovement_matrix
    M = comovement_matrix(feat)
    C = M @ M.T / M.shape[1]
    iu = np.triu_indices_from(C, 1)
    out["residual_corr"] = {"median": round(float(np.median(C[iu])), 4), "p95": round(float(np.quantile(C[iu], 0.95)), 3)}
    from compare import pattern_indicators
    lab = pattern_clusters(pattern_indicators(feat))
    vc = pd.Series(lab).value_counts()
    out["patterns"] = {"distinct": int(len(vc)), "clusters_ge5": int((vc >= 5).sum()),
                       "mo_in_clusters_ge5": int(vc[vc >= 5].sum()), "top2_mo": int(vc.head(2).sum())}
    ll, regions = leadlag_test(cfg, feat, ref)
    regions.to_csv(path(cfg, "processed", "leadlag_regions.csv"), index=False)
    out["leadlag"] = ll
    with open(path(cfg, "processed", "descriptives.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log.info("готово: %s", json.dumps(out, ensure_ascii=False)[:800])


if __name__ == "__main__":
    main()
