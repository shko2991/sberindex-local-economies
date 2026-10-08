"""Чувствительность итогового разбиения к построению сети и геометрии признаков.

Тот же метод (KEFRiN, евклидов, ξ×1 с выравниванием разброса, K итогового варианта), меняется
одно условие:
  — исключение каждого слоя мультиплекса (дороги / синхронная динамика / DTW);
  — равные суммарные массы слоёв вместо нормировки на средний вес ребра (сейчас доли общей массы
    весов неравны);
  — ILR-координаты долей (изометрическое преобразование, Egozcue и др., 2003) вместо CLR с
    отдельной z-стандартизацией каждой координаты.
Для каждого варианта: шесть ICVI (признаковые — в пространстве итоговых признаков, поэтому для
ILR-варианта они смещены в пользу итога; сетевые — на дорожной сети), ARI с итоговым разбиением
и внешняя проверка — η² логарифма зарплаты (Росстат) без городов федерального значения.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score

from common import get_logger, load_config, path
from compare import build_context
from icvi import all_indices
from methods import KEFRiN, modularity_transform, relabel_by_size

log = get_logger("sensitivity")


def ilr(shares: pd.DataFrame) -> pd.DataFrame:
    """Опорные (pivot) ILR-координаты: z_j = √((D−j)/(D−j+1)) · ln(x_j / gmean(x_{j+1..D}))."""
    X = np.log(shares.clip(lower=1e-6).values)
    D = X.shape[1]
    Z = np.column_stack([np.sqrt((D - j - 1) / (D - j)) * (X[:, j] - X[:, j + 1:].mean(1)) for j in range(D - 1)])
    return pd.DataFrame(Z, index=shares.index, columns=[f"ilr_{j + 1}" for j in range(D - 1)])


def layer_mass_shares(graphs: dict, layers: list[str]) -> dict:
    norm = {k: graphs[k] / graphs[k].data.mean() for k in layers}
    tot = sum(m.sum() for m in norm.values())
    return {k: round(float(m.sum() / tot), 3) for k, m in norm.items()}


def kefrin_euc(X, A, k, seed, n_init):
    P = modularity_transform(A)
    xi = (X ** 2).sum() / (P ** 2).sum()
    return relabel_by_size(KEFRiN(k, "euclidean", 1.0, xi, n_init, 100, seed).fit_predict(X, P))


def main():
    cfg = load_config()
    _, feat, ctx = build_context(cfg)
    ids = feat.X.index
    chosen = json.load(open(path(cfg, "processed", "summary.json"), encoding="utf-8"))["chosen"]
    ref = pd.read_parquet(path(cfg, "processed", "labels_candidates.parquet")).reindex(ids)[chosen].values
    k = int(chosen.rsplit("|k", 1)[1])
    layers = cfg["graphs"]["multiplex_layers"]
    G = ctx.graphs
    seed, n_init = int(cfg["seed"]), cfg["methods"]["kefrin"]["n_init"]
    A_eval = G["geography"]
    X = ctx.X
    variants = {}
    from graphs import multiplex
    variants["итог (три слоя, нормировка на средний вес ребра)"] = (X, multiplex({l: G[l] for l in layers}))
    for drop in layers:
        variants[f"без слоя {drop}"] = (X, multiplex({l: G[l] for l in layers if l != drop}))
    eq = sum(G[l] / G[l].sum() for l in layers)
    variants["равные суммарные массы слоёв"] = (X, sp.csr_matrix(eq))
    # ILR вместо CLR: блок структуры — ILR-координаты, стандартизованные общим множителем
    sh = feat.raw[[c for c in feat.raw.columns if c.startswith("share_")]]
    Z = ilr(sh)
    Z = (Z - Z.mean()) / np.sqrt((Z.var(ddof=0)).sum())       # общий масштаб: геометрия сохраняется
    Xd = pd.DataFrame(X, index=ids, columns=feat.X.columns)
    struct_cols = [c for c in Xd.columns if c.startswith("clr_")]
    Xi = pd.concat([Z, Xd.drop(columns=struct_cols)], axis=1).values
    variants["ILR вместо CLR (структура)"] = (Xi, multiplex({l: G[l] for l in layers}))
    # внешняя проверка на данных Росстата (если слой рынка труда посчитан): η² log зарплаты без ГФЗ
    lt_path = path(cfg, "processed", "labor_table.csv")
    wage = fed = None
    if lt_path.exists():
        from reference import load_reference
        lt = pd.read_csv(lt_path, index_col=0).reindex(ids)
        wage = np.log(lt.wage)
        fed = load_reference(cfg).reindex(ids).federal_city.values
    rows, labels = [], {}
    for name, (Xv, A) in variants.items():
        lab = kefrin_euc(Xv, A, k, seed, n_init)
        labels[name] = lab
        r = all_indices(X, A_eval, lab)                        # индексы в одном и том же пространстве
        r.update({"вариант": name, "ARI с итогом": round(adjusted_rand_score(ref, lab), 3)})
        if wage is not None:
            from validate import eta_squared
            r["η² log зарплаты (без ГФЗ)"] = round(eta_squared(wage[~fed], pd.Series(lab, index=ids)[~fed])[0], 3)
        rows.append(r)
        log.info("%s: ARI с итогом %.3f", name, r["ARI с итогом"])
    pd.DataFrame(labels, index=ids).to_parquet(path(cfg, "processed", "labels_sensitivity.parquet"))
    cols = ["вариант", "ARI с итогом", "SW", "CH", "S_Dbw", "AVI", "AVU", "MQ", "min_size", "max_share"]
    if wage is not None:
        cols.append("η² log зарплаты (без ГФЗ)")
    df = pd.DataFrame(rows)[cols]
    df.round(3).to_csv(path(cfg, "processed", "sensitivity.csv"), index=False)
    with open(path(cfg, "processed", "layer_mass.json"), "w", encoding="utf-8") as f:
        json.dump(layer_mass_shares(G, layers), f, ensure_ascii=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
