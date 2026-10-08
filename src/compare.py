"""Сравнение методов: сетка кандидатов → 6 ICVI → допустимость → Борда и Коупленд → устойчивость.

Запуск:  python src/compare.py            (результаты в data/processed и reports/)
Кандидат = метод × K (или разрешение Leiden) × сеть × вариант KEFRiN (расстояние, вес сети ξ).
Признаковые индексы (SW, CH, S_Dbw) считаются в одном и том же пространстве X для всех
кандидатов; сетевые (AVI, AVU, MQ) — на сети основной спецификации и, для анализа
чувствительности, на остальных сетях.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from common import get_logger, load_config, path, rng
from data import build_panel
from features import static_features
from graphs import build_graphs, multiplex
from icvi import aggregate, all_indices, graph_indices, kendall_w, rank_table
from methods import Context, relabel_by_size, run

log = get_logger("compare")


def pattern_indicators(feat) -> np.ndarray:
    """z-оценки средних долей 5 категорий: на сколько станд. откл. доля в МО выше средней по МО."""
    sh = feat.raw[[c for c in feat.raw.columns if c.startswith("share_") and not c.endswith("Прочее")]]
    return ((sh - sh.mean()) / sh.std(ddof=0)).values


def build_context(cfg: dict):
    panel = build_panel(cfg)
    feat = static_features(panel, cfg)
    graphs = build_graphs(cfg, feat)
    ctx = Context(X=feat.X.values, graphs=graphs, cfg=cfg, pattern_X=pattern_indicators(feat))
    ctx._cache["multiplex"] = multiplex({k: graphs[k] for k in cfg["graphs"]["multiplex_layers"]})
    return panel, feat, ctx


def candidates(cfg: dict) -> list[dict]:
    m = cfg["methods"]
    out = []
    for k in m["k_range"]:
        for name in ("kmeans", "gmm", "ward", "pattern"):
            out.append({"method": name, "k": k})
    for net in m["networks"]:
        for r in m["leiden"]["resolutions"]:
            out.append({"method": "leiden", "network": net, "resolution": r})
        for k in m["k_range"]:
            out.append({"method": "spectral", "network": net, "k": k})
            out.append({"method": "spectral_joint", "network": net, "k": k})
            for d in m["kefrin"]["distances"]:
                for xm in (m["kefrin"]["xi_grid"] if d == "cosine" else [1]):
                    out.append({"method": "kefrin", "network": net, "k": k, "distance": d, "xi_mult": xm})
    return out


def cand_id(c: dict) -> str:
    parts = [c["method"]]
    for key in ("network", "distance"):
        if key in c:
            parts.append(c[key])
    if "xi_mult" in c:
        parts.append(f"xi{c['xi_mult']:g}")
    parts.append(f"r{c['resolution']:g}" if "resolution" in c else f"k{c['k']}")
    return "|".join(parts)


def run_candidate(ctx: Context, c: dict, seed: int, bootstrap: bool = False) -> np.ndarray:
    kw = {}
    if c["method"] == "kefrin":
        kw = {"distance": c["distance"], "xi_mult": c["xi_mult"]}
        if bootstrap:
            kw["n_init"] = ctx.cfg["methods"]["kefrin"]["n_init_bootstrap"]
    return run(c["method"], ctx, k=c.get("k"), network=c.get("network"), resolution=c.get("resolution"),
               seed=seed, **kw)


def evaluate_all(cfg: dict, ctx: Context) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    ev = cfg["evaluation"]
    nets = [ev["network"]] + ev["sensitivity_networks"]
    rows, labels = [], {}
    cands = candidates(cfg)
    t0 = time.time()
    for i, c in enumerate(cands):
        cid = cand_id(c)
        lab = relabel_by_size(run_candidate(ctx, c, seed=int(cfg["seed"]) + i))
        labels[cid] = lab
        r = all_indices(ctx.X, ctx.graph(ev["network"]), lab)
        for net in ev["sensitivity_networks"]:
            r.update({f"{k}@{net}": v for k, v in graph_indices(ctx.graph(net), lab).items()})
        r.update({"id": cid, **c})
        rows.append(r)
        if (i + 1) % 25 == 0:
            log.info("кандидатов: %d/%d, %.0f с", i + 1, len(cands), time.time() - t0)
    df = pd.DataFrame(rows).set_index("id")
    n = len(ctx.X)
    df["admissible"] = (df.min_size >= ev["min_cluster_share"] * n) & (df.max_share <= ev["max_cluster_share"]) \
        & df.K.between(2, max(cfg["methods"]["k_range"]) * 2)
    return df, labels


def rank_candidates(df: pd.DataFrame, cfg: dict, network: str | None = None) -> pd.DataFrame:
    idx = cfg["evaluation"]["icvi"]
    d = df[df.admissible].copy()
    if network and network != cfg["evaluation"]["network"]:
        for k in ("AVI", "AVU", "MQ"):
            d[k] = d[f"{k}@{network}"]
    return aggregate(d, idx)


def stability(cfg: dict, ctx: Context, c: dict, full: np.ndarray, B: int, frac: float, seed: int) -> dict:
    """Устойчивость на подвыборках без возвращения: ARI между разбиением подвыборки и полным
    разбиением на тех же МО (Hennig, 2007, clusterboot — вариант с подвыборками)."""
    from sklearn.metrics import adjusted_rand_score
    g = rng(cfg, seed)
    n = len(ctx.X)
    aris = []
    for b in range(B):
        idx = np.sort(g.choice(n, int(frac * n), replace=False))
        sub = ctx.subset(idx)
        lab = run_candidate(sub, c, seed=seed + b, bootstrap=True)
        aris.append(adjusted_rand_score(full[idx], lab))
    a = np.array(aris)
    return {"ARI_mean": a.mean(), "ARI_q10": np.quantile(a, 0.1), "ARI_sd": a.std()}


def main():
    cfg = load_config()
    panel, feat, ctx = build_context(cfg)
    df, labels = evaluate_all(cfg, ctx)
    df.to_csv(path(cfg, "processed", "candidates.csv"))
    pd.DataFrame(labels, index=feat.X.index).to_parquet(path(cfg, "processed", "labels_candidates.parquet"))
    log.info("допустимых кандидатов: %d из %d", int(df.admissible.sum()), len(df))

    ranked = rank_candidates(df, cfg)
    W = kendall_w(rank_table(ranked, cfg["evaluation"]["icvi"]))
    log.info("согласованность индексов (W Кендалла): %.3f", W)

    ev = cfg["evaluation"]
    top = ranked.head(ev["bootstrap_top"])
    cand_by_id = {cand_id(c): c for c in candidates(cfg)}
    stab = {}
    for j, cid in enumerate(top.index):
        stab[cid] = stability(cfg, ctx, cand_by_id[cid], labels[cid], ev["bootstrap"], ev["subsample"], 1000 + j)
        log.info("устойчивость %s: ARI=%.3f", cid, stab[cid]["ARI_mean"])
    ranked = ranked.join(pd.DataFrame(stab).T)
    ranked.to_csv(path(cfg, "processed", "ranking_main.csv"))

    sens = {}
    for net in ev["sensitivity_networks"]:
        r = rank_candidates(df, cfg, net)
        sens[net] = r.index[:5].tolist()
    pd.DataFrame(sens).to_csv(path(cfg, "processed", "ranking_sensitivity_top5.csv"), index=False)
    log.info("готово")


if __name__ == "__main__":
    main()
