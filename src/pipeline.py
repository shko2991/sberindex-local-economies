"""Итоговый прогон после compare.py: выбор разбиения, описания типов, динамика, проверки.

  python src/pipeline.py

Правило выбора (зафиксировано до просмотра типов): среди допустимых кандидатов — наивысшее
место по Борда, при условии устойчивости ARI_mean ≥ evaluation.stability_min на подвыборках;
место по Коупленду приводится для контроля.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import get_logger, load_config, path
from compare import build_context, cand_id, candidates
from data import load_market_access
from dynamics import assign_periods, jaccard_match, trajectory_classes, transition_matrix, yearly_mode
from fca import describe_clusters, scale
from features import period_features
from graphs import edge_overlap, graph_summary, yearly_comovement_graphs
from methods import modularity_transform, run
from network_choice import community_strength, moran_table
from reference import load_reference
from validate import external_table, partition_agreement

log = get_logger("pipeline")

RU = {"share_Продовольствие": "доля продовольствия", "share_Здоровье": "доля здоровья",
      "share_Маркетплейсы": "доля маркетплейсов", "share_Общественное питание": "доля общепита",
      "share_Транспорт": "доля транспорта", "share_Прочее": "доля прочего", "level_total": "уровень расходов",
      "growth_rel": "относительный рост", "summer_peak": "летний пик", "dec_peak": "декабрьский пик",
      "volatility": "волатильность", "mp_shift": "сдвиг к маркетплейсам"}


def choose(cfg: dict) -> tuple[str, pd.DataFrame]:
    r = pd.read_csv(path(cfg, "processed", "ranking_main.csv"), index_col=0)
    thr = cfg["evaluation"]["stability_min"]
    ok = r[r.ARI_mean >= thr] if "ARI_mean" in r else r
    if ok.empty:
        log.warning("ни один кандидат не прошёл порог устойчивости %.2f — берётся лучший по Борда", thr)
        ok = r
    return ok.index[0], r


def type_profiles(feat, labels: pd.Series, ref: pd.DataFrame, ma: pd.Series) -> pd.DataFrame:
    prof = feat.raw.groupby(labels).median()
    prof.insert(0, "n", labels.value_counts().sort_index())
    prof["market_access"] = ma.reindex(labels.index).groupby(labels).median()
    prof["share_city"] = (ref.mo_type != "муниципальный район").groupby(labels).mean()
    prof["share_federal_city"] = ref.federal_city.groupby(labels).mean()
    prof["n_regions"] = ref.region_name.groupby(labels).nunique()
    return prof


def dynamics_block(cfg: dict, panel, ctx, labels: pd.Series, cand: dict) -> dict:
    F = period_features(panel, cfg)
    P, xi, dist = None, 0.0, "euclidean"
    if cand["method"] == "kefrin":
        dist = cand["distance"]
        P = modularity_transform(ctx.graph(cand["network"]))
        xi = cfg["methods"]["kefrin"]["rho_network"] * cand["xi_mult"]
        if dist == "euclidean":
            xi *= (ctx.X ** 2).sum() / (P ** 2).sum()
    T = assign_periods(F, labels, P, xi, dist)
    traj = trajectory_classes(T, cfg["dynamics"]["min_run"])
    # проверка: без сетевого члена (он постоянен во времени и сам по себе сглаживает переходы)
    T0 = assign_periods(F, labels, None, 0.0, dist)
    traj0 = trajectory_classes(T0, cfg["dynamics"]["min_run"])
    net_only_agree = None
    if P is not None:
        from methods import _row_normalize
        Pn = _row_normalize(P) if dist == "cosine" else P
        ks = np.unique(labels.values)
        if dist == "cosine":
            Lc = _row_normalize(np.stack([Pn[labels.values == k].mean(0) for k in ks]))
            net_lab = ks[(1 - Pn @ Lc.T).argmin(1)]
        else:
            Lc = np.stack([P[labels.values == k].mean(0) for k in ks])
            net_lab = ks[((P ** 2).sum(1)[:, None] - 2 * P @ Lc.T + (Lc ** 2).sum(1)[None]).argmin(1)]
        net_only_agree = float((net_lab == labels.values).mean())
    Y = yearly_mode(T)
    trans = transition_matrix(Y, Y.columns[0], Y.columns[-1])
    # проверка: независимая кластеризация каждого квартала тем же методом + венгерское сопоставление
    agree = {}
    for q, Fq in F.items():
        sub = type(ctx)(X=Fq.loc[labels.index].values, graphs=ctx.graphs, cfg=cfg, pattern_X=ctx.pattern_X)
        sub._cache = dict(ctx._cache)
        lab = run(cand["method"], sub, k=cand.get("k"), network=cand.get("network"),
                  resolution=cand.get("resolution"), seed=cfg["seed"],
                  **({"distance": cand["distance"], "xi_mult": cand["xi_mult"]} if cand["method"] == "kefrin" else {}))
        agree[q] = float((jaccard_match(T[q].values, lab) == T[q].values).mean())
    return {"T": T, "trajectory": traj, "yearly": Y, "transition": trans, "agreement": agree,
            "trajectory_features_only": traj0.value_counts().to_dict(), "network_term_only_agreement": net_only_agree}


def main():
    cfg = load_config()
    panel, feat, ctx = build_context(cfg)
    ref = load_reference(cfg).loc[feat.X.index]
    ma = load_market_access(cfg)
    L = pd.read_parquet(path(cfg, "processed", "labels_candidates.parquet"))
    best, ranking = choose(cfg)
    cand = {cand_id(c): c for c in candidates(cfg)}[best]
    labels = L[best].rename("type")
    log.info("итоговое разбиение: %s", best)
    out = {"chosen": best}

    prof = type_profiles(feat, labels, ref, ma)
    prof.to_csv(path(cfg, "processed", "type_profiles.csv"))
    B = scale(feat.raw, RU)
    fca = describe_clusters(B, labels, max_len=cfg["fca"]["max_len"], min_precision=cfg["fca"]["min_precision"],
                            top=cfg["fca"]["top"], min_extent=cfg["fca"]["min_extent"])
    fca.to_csv(path(cfg, "processed", "fca_descriptions.csv"), index=False)

    num = pd.DataFrame({"индекс доступности рынков": ma.reindex(labels.index), "широта": ref.lat, "долгота": ref.lon})
    cat = pd.DataFrame({"регион": ref.region_name, "тип МО": ref.mo_type, "город фед. значения": ref.federal_city})
    external_table(labels, num, cat).to_csv(path(cfg, "processed", "external_validation.csv"), index=False)

    # устойчивость к исключению городов федерального значения: тот же метод на остальных МО
    keep = ~ref.federal_city.values
    sub = ctx.subset(np.where(keep)[0])
    lab_sub = run(cand["method"], sub, k=cand.get("k"), network=cand.get("network"), resolution=cand.get("resolution"),
                  seed=cfg["seed"], **({"distance": cand["distance"], "xi_mult": cand["xi_mult"]}
                                       if cand["method"] == "kefrin" else {}))
    out["without_federal_cities"] = partition_agreement(labels[keep].reset_index(drop=True), pd.Series(lab_sub))

    dyn = dynamics_block(cfg, panel, ctx, labels, cand)
    dyn["T"].to_csv(path(cfg, "processed", "types_quarterly.csv"))
    dyn["transition"].to_csv(path(cfg, "processed", "transition_yearly.csv"))
    out["trajectory_counts"] = dyn["trajectory"].value_counts().to_dict()
    out["quarterly_agreement_independent"] = dyn["agreement"]
    out["trajectory_counts_features_only"] = dyn["trajectory_features_only"]
    out["network_term_only_agreement"] = dyn["network_term_only_agreement"]

    yc = yearly_comovement_graphs(cfg, feat)
    out["comovement_edges_jaccard_2023_2024"] = float(edge_overlap(yc).iloc[0, 1])

    G = dict(ctx.graphs)
    G["multiplex"] = ctx.graph("multiplex")
    graph_summary(G).to_csv(path(cfg, "processed", "graph_summary.csv"), index=False)
    edge_overlap(G).to_csv(path(cfg, "processed", "edge_overlap.csv"))
    moran_table(G, feat.raw, perms=cfg["network_choice"]["perms"]).to_csv(path(cfg, "processed", "moran.csv"))
    community_strength(G, nulls=cfg["network_choice"]["nulls"]).to_csv(path(cfg, "processed", "community_strength.csv"),
                                                                        index=False)

    final = ref[["name", "name_short", "region_name", "mo_type", "lat", "lon"]].assign(
        type=labels, trajectory=dyn["trajectory"], market_access=ma.reindex(labels.index))
    final = final.join(dyn["yearly"].add_prefix("type_"))
    final.to_csv(path(cfg, "processed", "final_types.csv"))
    with open(path(cfg, "processed", "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    log.info("готово: %s", json.dumps(out, ensure_ascii=False, default=float)[:500])


if __name__ == "__main__":
    main()
