"""Внутренние индексы качества (ICVI) и агрегирование рангов.

В пространстве признаков X:
  SW     — силуэт (Rousseeuw, 1987), ↑;
  CH     — Калински–Харабаш (1974), ↑;
  S_Dbw  — Halkidi & Vazirgiannis (2001): Scat (средний разброс внутри кластеров относительно
           общего) + Dens_bw (плотность в серединах между центрами относительно плотности
           у центров), ↓.
На графе A (S_ij — сумма весов рёбер между кластерами i и j; для неориентированного графа
каждое ребро внутри кластера входит в S_ii дважды):
  AVI    — средняя по кластерам доля веса рёбер, остающегося внутри: mean_i S_ii / Σ_j S_ij, ↑;
  AVU    — средняя по кластерам «связанность наружу»:
           (1/K) Σ_i Σ_{j≠i} S_ij / (out_i + in_j − S_ij), out_i = Σ_j S_ij − S_ii, in_j = Σ_i S_ij − S_jj, ↓;
  MQ     — модулярность Ньюмана: Σ_k [S_kk / 2m − (deg_k / 2m)²], ↑.
Определения AVI/AVU/модулярности сверены с открытой библиотекой Pattern (ВШЭ, GPL-3.0);
реализация собственная. Аббревиатура MQ в литературе неоднозначна (Modularization Quality
Манкоридиса и др. — другое определение); здесь MQ = модулярность Ньюмана.

Агрегирование: каждый индекс ранжирует кандидатов (метод × параметры); итог — по Борда
(сумма рангов) и по Коупленду (победы минус поражения в попарных сравнениях большинством
индексов). Совпадение победителей двух правил — признак устойчивого выбора.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

DIRECTION = {"SW": 1, "CH": 1, "S_Dbw": -1, "AVI": 1, "AVU": -1, "MQ": 1}


# ----------------------------------------------------------------------------- признаки

def s_dbw(X: np.ndarray, lab: np.ndarray) -> float:
    ks = np.unique(lab)
    K = len(ks)
    if K < 2:
        return np.nan
    cent = np.stack([X[lab == c].mean(0) for c in ks])
    var_all = np.linalg.norm(X.var(0))
    sig = [np.linalg.norm(X[lab == c].var(0)) for c in ks]
    scat = np.mean(sig) / var_all
    stdev = np.sqrt(np.sum(sig)) / K

    def density(P, u):
        return int((np.linalg.norm(P - u, axis=1) <= stdev).sum())

    tot = 0.0
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            P = X[(lab == ks[a]) | (lab == ks[b])]
            u = (cent[a] + cent[b]) / 2
            den = max(density(P, cent[a]), density(P, cent[b]))
            if den > 0:
                tot += density(P, u) / den
    dens_bw = tot / (K * (K - 1))
    return float(scat + dens_bw)


def feature_indices(X: np.ndarray, lab: np.ndarray, sample: int | None = None, seed: int = 0) -> dict:
    from sklearn.metrics import calinski_harabasz_score, silhouette_score
    if len(np.unique(lab)) < 2:
        return {"SW": np.nan, "CH": np.nan, "S_Dbw": np.nan}
    return {"SW": float(silhouette_score(X, lab, sample_size=sample, random_state=seed)),
            "CH": float(calinski_harabasz_score(X, lab)),
            "S_Dbw": s_dbw(X, lab)}


# ----------------------------------------------------------------------------- граф

def between_cluster_weights(A: sp.spmatrix, lab: np.ndarray) -> np.ndarray:
    ks, inv = np.unique(lab, return_inverse=True)
    H = sp.csr_matrix((np.ones(len(lab)), (np.arange(len(lab)), inv.ravel())), shape=(len(lab), len(ks)))
    return np.asarray((H.T @ A @ H).todense())


def graph_indices(A: sp.spmatrix, lab: np.ndarray) -> dict:
    S = between_cluster_weights(A, lab)
    K = len(S)
    row = S.sum(1)
    col = S.sum(0)
    diag = np.diag(S)
    with np.errstate(invalid="ignore", divide="ignore"):
        avi = np.nanmean(np.where(row > 0, diag / row, np.nan))
        out_ = row - diag
        in_ = col - diag
        avu = 0.0
        for i in range(K):
            for j in range(K):
                if i != j and S[i, j] > 0:
                    avu += S[i, j] / (out_[i] + in_[j] - S[i, j])
        avu /= K
    two_m = S.sum()
    mq = float(((diag / two_m) - (row / two_m) ** 2).sum()) if two_m > 0 else np.nan
    return {"AVI": float(avi), "AVU": float(avu), "MQ": mq}


def all_indices(X: np.ndarray, A: sp.spmatrix, lab: np.ndarray, sample: int | None = None) -> dict:
    out = feature_indices(X, lab, sample)
    out.update(graph_indices(A, lab))
    sizes = np.bincount(np.unique(lab, return_inverse=True)[1].ravel())
    out.update({"K": len(sizes), "min_size": int(sizes.min()), "max_share": float(sizes.max() / len(lab))})
    return out


# ----------------------------------------------------------------------------- агрегирование

def rank_table(df: pd.DataFrame, indices: list[str]) -> pd.DataFrame:
    """Ранг 1 — лучший по индексу (с учётом направления); равные значения — средний ранг."""
    R = pd.DataFrame(index=df.index)
    for c in indices:
        R[c] = (-DIRECTION[c] * df[c]).rank(method="average", na_option="bottom")
    return R


def borda(R: pd.DataFrame) -> pd.Series:
    n = len(R)
    return (n - R).sum(axis=1).rename("borda")          # больше — лучше


def copeland(R: pd.DataFrame) -> pd.Series:
    V = R.values
    n = len(V)
    better = (V[:, None, :] < V[None, :, :]).sum(-1)    # по скольким индексам a лучше b
    worse = (V[:, None, :] > V[None, :, :]).sum(-1)
    wins = (better > worse).sum(1)
    losses = (better < worse).sum(1)
    return pd.Series(wins - losses, index=R.index, name="copeland")


def aggregate(df: pd.DataFrame, indices: list[str]) -> pd.DataFrame:
    R = rank_table(df, indices)
    out = df.join(R.add_prefix("rank_")).join(borda(R)).join(copeland(R))
    out["borda_place"] = out.borda.rank(ascending=False, method="min").astype(int)
    out["copeland_place"] = out.copeland.rank(ascending=False, method="min").astype(int)
    return out.sort_values(["borda_place", "copeland_place"])


def kendall_w(R: pd.DataFrame) -> float:
    """Коэффициент конкордации Кендалла: согласованность индексов между собой (0..1)."""
    m, n = R.shape[1], R.shape[0]
    S = ((R.sum(1) - m * (n + 1) / 2) ** 2).sum()
    return float(12 * S / (m ** 2 * (n ** 3 - n)))
