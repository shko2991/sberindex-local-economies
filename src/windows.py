"""Дополнительный анализ: связи, меняющиеся во времени (скользящие окна по 12 месяцев).

Основной результат (8 типов по 24 месяцам) не меняется: здесь проверяется, что происходит с типами,
если пересчитывать во времени и признаки, и связи.

Окна — 12 месяцев с концом в IV кв. 2023 г. и в I–IV кв. 2024 г. (первое и последнее не пересекаются
по месяцам). В каждом окне заново считаются:
  — признаки узлов — те же 12, что в основном расчёте; «2024 г. к 2023 г.» заменяется на «вторая
    половина окна к первой» (в 12-месячном окне каждый календарный месяц встречается ровно один раз,
    поэтому летний и декабрьский пики определены так же); стандартизация — внутри окна;
  — графы синхронной динамики и DTW — по месяцам окна; дорожный граф постоянен; мультиплекс — по
    тому же правилу, что в основном расчёте.
Метод и параметры — итоговые: KEFRiN, мультиплекс, евклидово расстояние, ξ×1 с выравниванием
разброса, K = 8, те же число инициализаций и seed. Контроль без сети — k-means, K = 8, на тех же
признаках окна.

Метки окна сопоставляются с итоговыми типами венгерским алгоритмом по таблице сопряжённости.
Устойчивый переход МО (2023 → 2024) — одновременно:
  1) тип в окне 2023 г. ≠ тип в окне 2024 г. (окна не пересекаются по месяцам);
  2) тип окна 2024 г. держится и в окне с концом в III кв. 2024 г. (закрепился, а не мигнул);
  3) k-means на тех же признаках окон показывает тот же переход A → B (не артефакт сетевого члена).
Остальные смены типа — колебания на границах типов.

  python src/windows.py   → window_labels.parquet, window_summary.csv, window_edges.csv,
                            window_transitions.csv, window_mo.csv, window_summary.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari

from common import get_logger, load_config, path
from compare import build_context
from features import Features, clr, monthly_shares, residual_series, standardize_blocks, _months
from graphs import build_graphs, multiplex
from methods import Context, kefrin

log = get_logger("windows")


def window_features(panel, cfg: dict, months: list[str]) -> Features:
    """12 признаков узла по месяцам окна (как static_features, но «год к году» → «половина к половине»)."""
    shares = {c: s[months] for c, s in monthly_shares(panel, cfg).items()}
    resid = {c: r[months] for c, r in residual_series(panel).items()}
    total = cfg["features"]["total"]
    h1, h2 = months[: len(months) // 2], months[len(months) // 2:]
    mean_sh = pd.DataFrame({c: s.mean(axis=1) for c, s in shares.items()})
    struct = clr(mean_sh).add_prefix("clr_")
    rt = resid[total]
    level = pd.DataFrame({"level_total": rt.mean(axis=1)})

    def detrended_std(row):
        t = np.arange(len(row))
        b = np.polyfit(t, row.values, 1)
        return np.std(row.values - np.polyval(b, t))

    mp = shares["Маркетплейсы"]
    dyn = pd.DataFrame({
        "growth_rel": rt[h2].mean(axis=1) - rt[h1].mean(axis=1),
        "summer_peak": rt[_months(months, {6, 7, 8})].mean(axis=1) - rt.mean(axis=1),
        "dec_peak": rt[_months(months, {12})].mean(axis=1) - rt[_months(months, {10, 11})].mean(axis=1),
        "volatility": rt.apply(detrended_std, axis=1),
        "mp_shift": mp[h2].mean(axis=1) - mp[h1].mean(axis=1),
    })
    blocks = {"structure": list(struct.columns), "level": list(level.columns), "dynamics": list(dyn.columns)}
    use = list(cfg["features"]["blocks"])
    X = standardize_blocks(pd.concat([struct, level, dyn], axis=1), {b: blocks[b] for b in use})
    raw = pd.concat([mean_sh.add_prefix("share_"), level, dyn], axis=1)
    return Features(X=X, raw=raw, blocks={b: blocks[b] for b in use}, residual=resid, shares=shares)


def match_to(ref: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Переименовать метки lab в метки ref, максимизируя совпадение (венгерский алгоритм)."""
    r, l = np.unique(ref), np.unique(lab)
    C = pd.crosstab(lab, ref).reindex(index=l, columns=r, fill_value=0).values
    rows, cols = linear_sum_assignment(-C)
    m = {l[i]: r[j] for i, j in zip(rows, cols)}
    extra = [x for x in l if x not in m]                  # если в окне больше групп, чем в ref
    for k, x in enumerate(extra):
        m[x] = 100 + k
    return np.array([m[x] for x in lab])


def edge_set(A) -> set:
    import scipy.sparse as sp
    U = sp.triu(A, 1).tocoo()
    return set(zip(U.row.tolist(), U.col.tolist()))


def main():
    cfg = load_config()
    panel, feat, ctx = build_context(cfg)
    ids = feat.X.index
    fin = pd.read_csv(path(cfg, "processed", "final_types.csv"), index_col=0).type.reindex(ids).values
    rel = pd.read_csv(path(cfg, "processed", "mo_reliability.csv"), index_col=0).reliability.reindex(ids)
    seed = int(cfg["seed"])
    k = 8
    months = list(panel.months)
    ends = [i for i in range(11, len(months)) if months[i][5:7] in ("03", "06", "09", "12")]
    windows = {f"{months[e - 11]}…{months[e]}": months[e - 11: e + 1] for e in ends}
    log.info("окна: %s", list(windows))
    L, LK, edges = {}, {}, {}
    geo = ctx.graphs["geography"]
    for name, mw in windows.items():
        fw = window_features(panel, cfg, mw)
        assert (fw.X.index == ids).all()
        g = build_graphs(cfg, fw, families=["comovement", "leadlag"])
        g["geography"] = geo
        cw = Context(X=fw.X.values, graphs=g, cfg=cfg)
        cw._cache["multiplex"] = multiplex({l: g[l] for l in cfg["graphs"]["multiplex_layers"]})
        lab = kefrin(cw, "multiplex", k, seed, distance="euclidean", xi_mult=1.0)
        km = KMeans(k, n_init=cfg["methods"]["kmeans"]["n_init"], random_state=seed).fit_predict(fw.X.values)
        L[name], LK[name] = match_to(fin, lab), match_to(fin, km)
        edges[name] = {l: edge_set(g[l]) for l in ("comovement", "leadlag")}
        log.info("%s: ARI с итогом %.3f (k-means %.3f)", name, ari(fin, lab), ari(fin, km))
    names = list(windows)
    W = pd.DataFrame(L, index=ids)
    WK = pd.DataFrame(LK, index=ids)
    pd.concat([W.add_prefix("kefrin:"), WK.add_prefix("kmeans:")], axis=1).to_parquet(
        path(cfg, "processed", "window_labels.parquet"))

    # сводка по окнам
    rows = []
    for j, n in enumerate(names):
        r = {"окно": n, "ARI с итоговыми типами": ari(fin, W[n]), "доля МО в своём итоговом типе": float((W[n] == fin).mean()),
             "k-means: ARI с итоговыми типами": ari(fin, WK[n])}
        if j:
            p = names[j - 1]
            r["ARI с предыдущим окном"] = ari(W[p], W[n])
            r["k-means: ARI с предыдущим окном"] = ari(WK[p], WK[n])
        rows.append(r)
    summ = pd.DataFrame(rows).round(3)
    summ.to_csv(path(cfg, "processed", "window_summary.csv"), index=False)

    # смена связей между окнами: Жаккар рёбер (и с полным периодом)
    full = {l: edge_set(ctx.graphs[l]) for l in ("comovement", "leadlag")}
    er = []
    for j, n in enumerate(names):
        for l in ("comovement", "leadlag"):
            e = edges[n][l]
            er.append({"окно": n, "слой": l, "Жаккар с полным периодом": len(e & full[l]) / len(e | full[l]),
                       "Жаккар с предыдущим окном": (len(e & edges[names[j - 1]][l]) / len(e | edges[names[j - 1]][l])) if j else np.nan})
    pd.DataFrame(er).round(3).to_csv(path(cfg, "processed", "window_edges.csv"), index=False)

    # МО: число смен, устойчивые переходы 2023 → 2024
    first, last, prelast = names[0], names[-1], names[-2]
    changes = (W.diff(axis=1).iloc[:, 1:] != 0).sum(axis=1)
    moved = W[first] != W[last]
    settled = W[prelast] == W[last]
    same_km = (WK[first] == W[first]) & (WK[last] == W[last])
    persistent = moved & settled & same_km
    mo = pd.DataFrame({"итоговый тип": fin, "смен типа между окнами": changes, "тип в окне 2023": W[first],
                       "тип в окне 2024": W[last], "устойчивый переход": persistent, "надёжность типа": rel.round(3)},
                      index=ids)
    mo.to_csv(path(cfg, "processed", "window_mo.csv"))
    pd.crosstab(W[first], W[last]).to_csv(path(cfg, "processed", "window_transitions.csv"))
    low = rel < 0.6
    out = {
        "windows": names,
        "same_type_all_windows": float((changes == 0).mean()),
        "changed_2023_2024": int(moved.sum()),
        "persistent_transitions": int(persistent.sum()),
        "kmeans_same_type_all_windows": float(((WK.diff(axis=1).iloc[:, 1:] != 0).sum(axis=1) == 0).mean()),
        "reliability_mean_stable": float(rel[changes == 0].mean()),
        "reliability_mean_changing": float(rel[changes > 0].mean()),
        "share_changing_among_low_reliability": float((changes[low] > 0).mean()),
        "share_changing_among_high_reliability": float((changes[~low] > 0).mean()),
        "ari_window2023_window2024": ari(W[first], W[last]),
        "kmeans_ari_window2023_window2024": ari(WK[first], WK[last]),
        "persistent_by_pair": {f"{a}→{b}": int(v) for (a, b), v in
                               mo[persistent].groupby(["тип в окне 2023", "тип в окне 2024"]).size().items()},
    }
    with open(path(cfg, "processed", "window_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(summ.to_string(index=False))
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
