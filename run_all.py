"""Полный прогон одной командой (данные уже скачаны: см. README).

  python run_all.py                 — всё, включая сравнение 280 вариантов (~45 мин на 2 ядрах)
  python run_all.py --skip-compare  — без сравнения (берутся готовые data/processed/candidates.csv и ranking_main.csv)
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("проверка данных", ["src/download.py", "--check"]),
    ("тесты", ["-m", "pytest", "tests", "-q"]),
    ("описательные расчёты и тест опережения", ["src/descriptives.py"]),
    ("сравнение методов", ["src/compare.py"]),
    ("устойчивость выбора к правилам оценки", ["src/robustness.py"]),
    ("итоговое разбиение, описания, динамика", ["src/pipeline.py"]),
    ("рынок труда и расхождения", ["src/labor_run.py"]),
    ("синтетическая проверка KEFRiN", ["src/synthetic.py"]),
    ("рисунки", ["src/figures.py"]),
    ("данные страницы", ["src/landing_data.py"]),
    ("сборка страницы", ["landing/build.py"]),
]

if __name__ == "__main__":
    skip = "--skip-compare" in sys.argv
    has_labor = (ROOT / "data" / "external" / "bdmo" / "bdmo_2021_2024.parquet").exists()
    if not has_labor:
        print("Нет data/external/bdmo/bdmo_2021_2024.parquet — слой рынка труда пропускается "
              "(см. src/extract_bdmo.py); страница и рисунки соберутся без него.")
    for name, args in STEPS:
        if skip and name == "сравнение методов":
            continue
        if not has_labor and name == "рынок труда и расхождения":
            continue
        print(f"\n=== {name} ===", flush=True)
        r = subprocess.run([sys.executable, *args], cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"Шаг «{name}» завершился с ошибкой ({r.returncode})")
    print("\nГотово: reports/report.md, reports/figures/, docs/index.html")
