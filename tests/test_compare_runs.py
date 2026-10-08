import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from compare_runs import compare  # noqa: E402


def _write(d: Path, df: pd.DataFrame):
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "labels.parquet")


def test_swapped_territory_ids_are_detected(tmp_path):
    a = pd.DataFrame({"type": [0, 1]}, index=pd.Index([10, 20], name="territory_id"))
    b = pd.DataFrame({"type": [0, 1]}, index=pd.Index([20, 10], name="territory_id"))
    _write(tmp_path / "a", a)
    _write(tmp_path / "b", b)
    res = compare(tmp_path / "a", tmp_path / "b").set_index("файл").loc["labels.parquet", "результат"]
    assert res == "различается"


def test_same_values_same_ids_match(tmp_path):
    a = pd.DataFrame({"x": [0.1, 0.2]}, index=pd.Index([10, 20], name="territory_id"))
    b = a.copy()
    b["x"] = b["x"] + 1e-12
    _write(tmp_path / "a", a)
    _write(tmp_path / "b", b)
    res = compare(tmp_path / "a", tmp_path / "b").set_index("файл").loc["labels.parquet", "результат"]
    assert res in ("побайтно совпадает", "совпадает численно")
