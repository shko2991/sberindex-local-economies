"""Устойчивость выбора к правилам оценки (по результатам compare.py).

1. Набор индексов: исключение по одному индексу и агрегирование без дублей (SW и CH, AVI и MQ
   почти совпадают по рангам — пара получает вес 1/2 каждому). Для каждого варианта правила —
   победитель, место выбранного варианта и ARI между победителем и выбранным разбиением.
2. Внешняя сеть оценки: железные дороги (не входят ни в один слой мультиплекса). AVI, AVU, MQ
   считаются на kNN-графе железнодорожных расстояний для МО с ж/д сообщением.
3. Что добавляет сеть: итог против k-means по тем же признакам при K = 7 и K = 8 — ARI, AVI и MQ
   на дорогах, компоненты связности каждого типа в дорожном графе и доля МО в крупнейшей
   компоненте своего типа (модулярность измеряет концентрацию связей, а не связность).
Выбранный вариант определяется тем же правилом, что в pipeline.py (compare.choose), без чтения
результатов последующих шагов.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from common import get_logger, load_config, path
from compare import choose
from data import build_panel, distance_matrix, load_distances
from graphs import knn_sparsify
from icvi import DIRECTION, graph_indices, rank_table

log = get_logger("robustness")


def weighted_borda(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    R = rank_table(df, list(weights))
    n = len(R)
    return sum((n - R[k]) * w for k, w in weights.items())


def type_components(A, labels: np.ndarray) -> tuple[list[int], float]:
    """Число компонент связности каждого типа в графе и доля МО в крупнейшей компоненте своего типа."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    A = sp.csr_matrix(A)
    comps, largest = [], 0
    for t in np.unique(labels):
        idx = np.where(labels == t)[0]
        nc, lab = connected_components(A[idx][:, idx], directed=False)
        comps.append(int(nc))
        largest += int(np.bincount(lab).max())
    return comps, largest / len(labels)


def network_value(cfg: dict, L: pd.DataFrame, chosen: str, df: pd.DataFrame) -> pd.DataFrame:
    from data import build_panel as _bp
    from graphs import geography_similarity
    panel = _bp(cfg)
    ids = panel.territories
    gc = cfg["graphs"]
    A = knn_sparsify(geography_similarity(cfg, ids), gc["geography"]["knn"], gc.get("mutual", False))   # тот же граф, что в методах
    rows = []
    for cid in [chosen, "kmeans|k7", "kmeans|k8"]:
        if cid not in L:
            continue
        lab = L[cid].reindex(ids).values
        comps, share = type_components(A, lab)
        rows.append({"вариант": cid, "K": int(len(np.unique(lab))),
                     "ARI с итогом": round(adjusted_rand_score(L[chosen].reindex(ids), lab), 3),
                     "AVI (дороги)": round(float(df.loc[cid, "AVI"]), 3), "MQ (дороги)": round(float(df.loc[cid, "MQ"]), 3),
                     "компонент по типам": " / ".join(map(str, comps)),
                     "доля МО в крупнейшей компоненте своего типа": round(share, 4)})
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    df = pd.read_csv(path(cfg, "processed", "candidates.csv"), index_col=0)
    L = pd.read_parquet(path(cfg, "processed", "labels_candidates.parquet"))
    chosen, _ = choose(cfg)
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
    nv = network_value(cfg, L, chosen, df)
    nv.to_csv(path(cfg, "processed", "network_value.csv"), index=False)
    out = {"index_sets": sens.to_dict("records"), "railway": rail_res, "chosen": chosen}
    with open(path(cfg, "processed", "robustness.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    pd.set_option("display.width", 250)
    print(sens.to_string(index=False))
    print(rail_res)


if __name__ == "__main__":
    main()
