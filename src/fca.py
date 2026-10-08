"""Описание типов средствами анализа формальных понятий (АФП / FCA).

Формальный контекст: объекты — МО, признаки — бинарные утверждения о показателях после
порядковой шкализации: значение в верхней/нижней трети распределения по МО («↑ доля
продовольствия», «↓ …») или в крайнем дециле («↑↑», «↓↓»). Формальное понятие — пара (объём, содержание), где объём —
все МО, обладающие всеми признаками содержания, а содержание — все признаки, общие для объёма
(Ganter & Wille, 1999). Описание типа k — понятие с коротким содержанием (≤ max_len признаков),
объём которого в основном состоит из МО типа k:
  точность = |объём ∩ тип k| / |объём|    (уверенность импликации «содержание → тип k»),
  покрытие = |объём ∩ тип k| / |тип k|.
Устойчивость понятия оценивается Δ-мерой: на сколько МО сократится объём при добавлении любого
ещё одного признака (Buzmakov, Kuznetsov, Napoli, 2014 — Δ-мера ограничивает индекс
устойчивости Кузнецова, 2007, и считается быстро). Малая Δ — описание держится на нескольких МО.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd




def scale(raw: pd.DataFrame, labels_ru: dict[str, str] | None = None, q: float = 1 / 3,
          q_extreme: float | None = 0.1) -> pd.DataFrame:
    """Порядковая шкализация: «↑» — значение в верхней трети МО (≥ квантиля 1−q), «↓» — в нижней;
    «↑↑»/«↓↓» — в верхнем/нижнем дециле (нужны, чтобы описать малочисленные типы)."""
    labels_ru = labels_ru or {}
    out = {}
    for c in raw.columns:
        name = labels_ru.get(c, c)
        lo, hi = raw[c].quantile(q), raw[c].quantile(1 - q)
        out[f"↑ {name}"] = raw[c] >= hi
        out[f"↓ {name}"] = raw[c] <= lo
        if q_extreme:
            lo2, hi2 = raw[c].quantile(q_extreme), raw[c].quantile(1 - q_extreme)
            out[f"↑↑ {name}"] = raw[c] >= hi2
            out[f"↓↓ {name}"] = raw[c] <= lo2
    return pd.DataFrame(out, index=raw.index)


def closure(B: np.ndarray, extent: np.ndarray) -> np.ndarray:
    """Содержание объёма: признаки, общие для всех объектов (B — булева матрица N × M)."""
    if not extent.any():
        return np.ones(B.shape[1], dtype=bool)
    return B[extent].all(0)


def concepts_up_to(B: np.ndarray, max_len: int = 3, min_extent: int = 10) -> list[tuple[frozenset, np.ndarray]]:
    """Все понятия, содержание которых порождается ≤ max_len признаками (замыкания комбинаций)."""
    M = B.shape[1]
    seen: dict[frozenset, np.ndarray] = {}
    for r in range(1, max_len + 1):
        for comb in combinations(range(M), r):
            ext = B[:, list(comb)].all(1)
            if ext.sum() < min_extent:
                continue
            intent = frozenset(np.where(closure(B, ext))[0])
            if intent not in seen:
                seen[intent] = ext
    return list(seen.items())


def delta_measure(B: np.ndarray, intent: frozenset, ext: np.ndarray) -> int:
    n = int(ext.sum())
    others = [m for m in range(B.shape[1]) if m not in intent]
    if not others:
        return n
    sub = B[ext][:, others].sum(0)
    return int(n - sub.max())


def describe_clusters(Bdf: pd.DataFrame, labels: pd.Series, max_len: int = 3, min_precision: float = 0.6,
                      top: int = 3, min_extent: int = 10) -> pd.DataFrame:
    B = Bdf.loc[labels.index].values.astype(bool)
    names = np.array(Bdf.columns)
    cons = concepts_up_to(B, max_len, min_extent)
    lab = labels.values
    rows = []
    for k in np.unique(lab):
        in_k = lab == k
        nk = in_k.sum()
        cand = []
        for intent, ext in cons:
            hit = (ext & in_k).sum()
            if hit == 0:
                continue
            prec, rec = hit / ext.sum(), hit / nk
            if prec < min_precision:
                continue
            # короткое описание: из содержания оставляем порождающие признаки минимальной длины
            cand.append((intent, ext, prec, rec))
        # лучшие по F-мере, при равенстве — более короткие
        cand.sort(key=lambda t: (-(2 * t[2] * t[3] / (t[2] + t[3])), len(t[0])))
        for intent, ext, prec, rec in cand[:top]:
            gen = minimal_generator(B, intent, ext, max_len)
            rows.append({"type": k, "description": " ∧ ".join(names[list(gen)]), "n_attrs": len(gen),
                         "extent": int(ext.sum()), "precision": round(prec, 3), "coverage": round(rec, 3),
                         "delta": delta_measure(B, intent, ext)})
    return pd.DataFrame(rows)


def minimal_generator(B: np.ndarray, intent: frozenset, ext: np.ndarray, max_len: int) -> tuple:
    """Кратчайшее подмножество содержания с тем же объёмом (для читаемости описания)."""
    items = sorted(intent)
    for r in range(1, min(max_len, len(items)) + 1):
        for comb in combinations(items, r):
            if np.array_equal(B[:, list(comb)].all(1), ext):
                return comb
    return tuple(items)
