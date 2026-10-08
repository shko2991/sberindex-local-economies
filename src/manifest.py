"""Манифест SHA-256 результатов: data/processed, reports/figures, docs/index.html → reports/manifest_sha256.txt.

  python src/manifest.py            — записать манифест
  python src/manifest.py --check    — сверить текущие файлы с манифестом (код возврата 1 при расхождении)

Формат — как у sha256sum: «хеш  путь». Каталог labor_v1/ (результаты первой версии подготовки данных,
сохранены для сравнения) входит в манифест, но новым прогоном не создаётся.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "manifest_sha256.txt"
SKIP = {"landing_data.json"}            # собирается при сборке страницы, в git не хранится


def files() -> list[Path]:
    out = [p for d in ("data/processed", "reports/figures") for p in sorted((ROOT / d).rglob("*"))
           if p.is_file() and p.name not in SKIP]
    return out + [ROOT / "docs" / "index.html"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if "--check" in sys.argv:
        want = dict(reversed(l.split("  ", 1)) for l in OUT.read_text(encoding="utf-8").splitlines() if l and not l.startswith("#"))
        bad = [f for f, h in want.items() if not (ROOT / f).exists() or sha(ROOT / f) != h]
        print(f"файлов в манифесте: {len(want)}, расхождений: {len(bad)}")
        for f in bad:
            print("  ", f)
        sys.exit(1 if bad else 0)
    lines = ["# SHA-256 результатов; проверка: python src/manifest.py --check"]
    lines += [f"{sha(p)}  {p.relative_to(ROOT).as_posix()}" for p in files()]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"манифест: {OUT.relative_to(ROOT)} ({len(lines) - 1} файлов)")


if __name__ == "__main__":
    main()
