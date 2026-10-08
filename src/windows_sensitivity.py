"""Чувствительность анализа скользящих окон (дополнительная проверка к windows.py).

Риск, указанный рецензентом: в 12-месячном окне «рост» и «сдвиг к маркетплейсам» — разность второй и
первой половин окна; при сдвиге окна меняется состав месяцев в половинах, и местная сезонность (не
удалённая вычитанием общей медианы) может создавать ложные «изменения». Проверка:
  — вариант «без полугодовых контрастов»: из блока динамики исключены growth_rel и mp_shift (блок
    перевзвешивается: делится на √3 вместо √5); остальное — как в windows.py;
  — для обоих вариантов — три seed KEFRiN и k-means (seed, seed + 1, seed + 2).
Сравниваются: доля МО с неизменной меткой во всех окнах, ARI между непересекающимися окнами 2023 и
2024 гг., число кандидатов на устойчивый переход и их пересечение с кандидатами основного расчёта.
Основной вариант с основным seed обязан совпасть с window_labels.parquet (самопроверка).

  python src/windows_sensitivity.py   → window_sensitivity.csv, window_sensitivity_types.csv,
                                        window_sensitivity_candidates.csv, window_sensitivity_labels.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari

from common import get_logger, load_config, path
from compare import build_context
from graphs import build_graphs, multiplex
from methods import Context, kefrin
from windows import match_to, window_features

log = get_logger("windows_sensitivity")
HALF_CONTRASTS = ("growth_rel", "mp_shift")


def run(cfg, panel, ctx, ids, fin, seed: int, drop=()):
    months = list(panel.months)
    ends = [i for i in range(11, len(months)) if months[i][5:7] in ("03", "06", "09", "12")]
    windows = {f"{months[e - 11]}…{months[e]}": months[e - 11: e + 1] for e in ends}
    W, WK = {}, {}
    for name, mw in windows.items():
        fw = window_features(panel, cfg, mw, drop=drop)
        g = build_graphs(cfg, fw, families=["comovement", "leadlag"])
        g["geography"] = ctx.graphs["geography"]
        cw = Context(X=fw.X.values, graphs=g, cfg=cfg)
        cw._cache["multiplex"] = multiplex({l: g[l] for l in cfg["graphs"]["multiplex_layers"]})
        W[name] = match_to(fin, kefrin(cw, "multiplex", 8, seed, distance="euclidean", xi_mult=1.0))
        WK[name] = match_to(fin, KMeans(8, n_init=cfg["methods"]["kmeans"]["n_init"], random_state=seed).fit_predict(fw.X.values))
    return pd.DataFrame(W, index=ids), pd.DataFrame(WK, index=ids)


def persistent(W: pd.DataFrame, WK: pd.DataFrame) -> pd.Series:
    n = list(W.columns)
    return (W[n[0]] != W[n[-1]]) & (W[n[-2]] == W[n[-1]]) & (WK[n[0]] == W[n[0]]) & (WK[n[-1]] == W[n[-1]])


def main():
    cfg = load_config()
    panel, feat, ctx = build_context(cfg)
    ids = feat.X.index
    fin = pd.read_csv(path(cfg, "processed", "final_types.csv"), index_col=0).type.reindex(ids).values
    base_seed = int(cfg["seed"])
    pub = pd.read_parquet(path(cfg, "processed", "window_labels.parquet")).reindex(ids)
    pub_W = pub[[c for c in pub if c.startswith("kefrin:")]].rename(columns=lambda c: c.split(":", 1)[1])
    pub_WK = pub[[c for c in pub if c.startswith("kmeans:")]].rename(columns=lambda c: c.split(":", 1)[1])
    base_cand = set(ids[persistent(pub_W, pub_WK).values])
    rows, cands, trows, labels = [], [], [], {}
    for variant, drop in (("все 12 признаков (как в windows.py)", ()), ("без полугодовых контрастов (growth_rel, mp_shift)", HALF_CONTRASTS)):
        for seed in (base_seed, base_seed + 1, base_seed + 2):
            W, WK = run(cfg, panel, ctx, ids, fin, seed, drop)
            if not drop and seed == base_seed:
                same = bool((W.values == pub_W.values).all() and (WK.values == pub_WK.values).all())
                log.info("самопроверка: совпадение с window_labels.parquet — %s", same)
                if not same:
                    raise SystemExit("основной вариант не совпал с window_labels.parquet")
            n = list(W.columns)
            stable = (W.diff(axis=1).iloc[:, 1:] != 0).sum(axis=1) == 0
            p = persistent(W, WK)
            cs = set(ids[p.values])
            rows.append({"вариант": variant, "seed": seed,
                         "МО с неизменной меткой во всех окнах": round(float(stable.mean()), 4),
                         "k-means: МО с неизменной меткой": round(float(((WK.diff(axis=1).iloc[:, 1:] != 0).sum(axis=1) == 0).mean()), 4),
                         "ARI окон 2023 и 2024": round(ari(W[n[0]], W[n[-1]]), 4),
                         "k-means: ARI окон 2023 и 2024": round(ari(WK[n[0]], WK[n[-1]]), 4),
                         "ARI окна 2024 с итоговыми типами": round(ari(fin, W[n[-1]]), 4),
                         "кандидатов на устойчивый переход": int(p.sum()),
                         "из них среди 33 основных": len(cs & base_cand),
                         "Жаккар с 33 основными": round(len(cs & base_cand) / max(len(cs | base_cand), 1), 3)})
            cands += [{"вариант": variant, "seed": seed, "territory_id": int(t)} for t in sorted(cs)]
            stable_k = (WK.diff(axis=1).iloc[:, 1:] != 0).sum(axis=1) == 0
            all_fin = (W.values == fin[:, None]).all(axis=1)
            for t in sorted(set(fin)):
                m = fin == t
                trows.append({"вариант": variant, "seed": seed, "итоговый тип": int(t), "МО": int(m.sum()),
                              "неизменная метка во всех окнах": round(float(stable[m].mean()), 4),
                              "во всех окнах итоговый тип": round(float(all_fin[m].mean()), 4),
                              "k-means: неизменная метка": round(float(stable_k[m].mean()), 4)})
            tag = "base" if not drop else "no_half"
            for w in n:
                labels[f"{tag}|{seed}|kefrin|{w}"] = W[w].values
                labels[f"{tag}|{seed}|kmeans|{w}"] = WK[w].values
            log.info("%s, seed %d: %s", variant, seed, rows[-1])
    out = pd.DataFrame(rows)
    out.to_csv(path(cfg, "processed", "window_sensitivity.csv"), index=False)
    pd.DataFrame(cands).to_csv(path(cfg, "processed", "window_sensitivity_candidates.csv"), index=False)
    pd.DataFrame(trows).to_csv(path(cfg, "processed", "window_sensitivity_types.csv"), index=False)
    pd.DataFrame(labels, index=ids).to_parquet(path(cfg, "processed", "window_sensitivity_labels.parquet"))
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
