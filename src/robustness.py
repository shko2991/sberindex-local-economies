"""Устойчивость выбора к правилам оценки (по результатам compare.py).

1. Набор индексов: исключение по одному индексу и агрегирование без дублей (SW и CH, AVI и MQ
   почти совпадают по рангам — пара получает вес 1/2 каждому). Для каждого варианта правила —
   победитель, место выбранного варианта и ARI между победителем и выбранным разбиением.
2. Внешняя сеть оценки: железные дороги (не входят ни в один слой мультиплекса). AVI, AVU, MQ
   считаются на kNN-графе железнодорожных расстояний для МО с ж/д сообщением.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from common import get_logger, load_config, path
from data import build_panel, distance_matrix, load_distances
from graphs import knn_sparsify
from icvi import DIRECTION, graph_indices, rank_table

log = get_logger("robustness")


def weighted_borda(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    R = rank_table(df, list(weights))
    n = len(R)
    return sum((n - R[k]) * w for k, w in weights.items())


def main():
    cfg = load_config()
    df = pd.read_csv(path(cfg, "processed", "candidates.csv"), index_col=0)
    L = pd.read_parquet(path(cfg, "processed", "labels_candidates.parquet"))
    chosen = json.load(open(path(cfg, "processed", "summary.json"), encoding="utf-8"))["chosen"]
    adm = df[df.admissible]
    idx = cfg["evaluation"]["icvi"]
    rules = {"все 6 индексов (основное правило)": {k: 1.0 for k in idx}}
    for k in idx:
        rules[f"без {k}"] = {j: 1.0 for j in idx if j != k}
    rules["без дублей (SW, CH, AVI, MQ — по 1/2)"] = {"SW": .5, "CH": .5, "S_Dbw": 1, "AVI": .5, "AVU": 1, "MQ": .5}
    rows = []
    for name, w in rules.items():
        b = weighted_borda(adm, w).sort_values(ascending=False)
        win = b.index[0]
        rows.append({"правило": name, "победитель": win, "место выбранного": int((b > b[chosen]).sum() + 1),
                     "ARI победителя с выбранным": round(adjusted_rand_score(L[chosen], L[win]), 3),
                     "K победителя": int(adm.loc[win, "K"])})
    sens = pd.DataFrame(rows)
    sens.to_csv(path(cfg, "processed", "robust_indices.csv"), index=False)

    # внешняя сеть оценки: железные дороги
    panel = build_panel(cfg)
    ids = panel.territories
    rail = load_distances(cfg, "railway")
    D = distance_matrix(rail, ids)
    has = np.isfinite(D).sum(1) > 1
    sub = np.where(has)[0]
    g = cfg["graphs"]["geography"]
    S = np.exp(-D[np.ix_(sub, sub)] / g["sigma_km"])
    S[~np.isfinite(D[np.ix_(sub, sub)])] = 0
    A = knn_sparsify(S, g["knn"])
    gi = pd.DataFrame({c: graph_indices(A, L[c].values[sub]) for c in adm.index}).T
    rail_df = adm[["SW", "CH", "S_Dbw"]].join(gi)
    rb = weighted_borda(rail_df, {k: 1.0 for k in idx}).sort_values(ascending=False)
    rail_res = {"n_mo": int(len(sub)), "winner": rb.index[0], "chosen_place": int((rb > rb[chosen]).sum() + 1),
                "n_candidates": int(len(rb)), "ari_winner_chosen": round(adjusted_rand_score(L[chosen], L[rb.index[0]]), 3),
                "top5": rb.index[:5].tolist()}
    out = {"index_sets": sens.to_dict("records"), "railway": rail_res}
    with open(path(cfg, "processed", "robustness.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    pd.set_option("display.width", 250)
    print(sens.to_string(index=False))
    print(rail_res)


if __name__ == "__main__":
    main()
