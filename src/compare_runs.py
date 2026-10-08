"""Сверка двух каталогов результатов (например, прогона с пустого каталога и опубликованного).

  python src/compare_runs.py КАТАЛОГ_А КАТАЛОГ_Б [--out reports/cold_start_check.md]

Для каждого файла: побайтно совпадает / совпадает численно (CSV и JSON, допуск 1e-9) / различается
(с указанием первых различий) / есть только в одном каталоге.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _json_equal(a, b, tol=1e-9) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    return a == b


def _table_diff(fa: Path, fb: Path, tol=1e-9) -> str | None:
    read = pd.read_parquet if fa.suffix == ".parquet" else (lambda f: pd.read_csv(f, low_memory=False))
    a, b = read(fa), read(fb)
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return f"форма {a.shape} против {b.shape}"
    bad = []
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            ok = np.isclose(xv, yv, rtol=tol, atol=tol, equal_nan=True)
        else:
            ok = (x.astype(str) == y.astype(str)).to_numpy()
        if not ok.all():
            bad.append(f"{c}: {int((~ok).sum())} строк")
    return "; ".join(bad[:5]) if bad else None


def compare(da: Path, db: Path) -> pd.DataFrame:
    rows = []
    names = sorted({p.relative_to(da).as_posix() for p in da.rglob("*") if p.is_file()} |
                   {p.relative_to(db).as_posix() for p in db.rglob("*") if p.is_file()})
    for n in names:
        fa, fb = da / n, db / n
        if not fa.exists() or not fb.exists():
            rows.append({"файл": n, "результат": "только в " + ("Б" if not fa.exists() else "А"), "подробности": ""})
            continue
        if _sha(fa) == _sha(fb):
            rows.append({"файл": n, "результат": "побайтно совпадает", "подробности": ""})
            continue
        detail, res = "", "различается"
        try:
            if fa.suffix == ".json":
                if _json_equal(json.loads(fa.read_text(encoding="utf-8")), json.loads(fb.read_text(encoding="utf-8"))):
                    res = "совпадает численно"
            elif fa.suffix in (".csv", ".parquet"):
                d = _table_diff(fa, fb)
                res, detail = ("совпадает численно", "") if d is None else ("различается", d)
        except Exception as e:  # noqa: BLE001
            detail = f"не удалось сравнить: {e}"
        rows.append({"файл": n, "результат": res, "подробности": detail})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--title", default="Сверка каталогов результатов")
    a = ap.parse_args()
    t = compare(a.a, a.b)
    counts = t["результат"].value_counts().to_dict()
    lines = [f"# {a.title}", "", f"А: `{a.a}`  ", f"Б: `{a.b}`", "",
             "Итого: " + ", ".join(f"{k} — {v}" for k, v in counts.items()), "",
             "| Файл | Результат | Подробности |", "|---|---|---|"]
    lines += [f"| `{r.файл}` | {r.результат} | {r.подробности} |" for r in t.itertuples()]
    text = "\n".join(lines) + "\n"
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
