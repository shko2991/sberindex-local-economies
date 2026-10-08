"""Выборка нужных показателей из архивов «Муниципальная статистика России с 2005 года» (tochno.st/datasets/bdmo).

Архивы «по показателям» весят ~330 ГБ без сжатия, поэтому читаются только нужные показатели, только
файлы 2021–2025 гг. и только МО верхнего уровня (без поселений). Результат — один parquet (~15 МБ).

  1) скачать архивы «по показателям» разделов 2, 31 и 32:
     https://storage.yandexcloud.net/tochno-st-catalog/Rosstat/data_bdmo_118_v20250918/by_indicator/data_section{2,31,32}_112_v20250918.zip
  2) python src/extract_bdmo.py ПАПКА_С_АРХИВАМИ   (≈ 2–30 минут в зависимости от диска)
  3) файл bdmo_2021_2024.parquet появится в data/external/bdmo/
"""
from __future__ import annotations

import pathlib
import re
import sys
import time
import zipfile

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

YEARS = ["2021", "2022", "2023", "2024", "2025"]
AGE = r"(?i)всего|все возраст|трудосп|моложе|старше"      # только итоги и крупные возрастные группы
WANT = {
    "Y48423007": ("data_section32", None),   # среднемесячная зарплата (без СМП), по ОКВЭД2
    "Y48423005": ("data_section32", None),   # среднесписочная численность работников (без СМП), по ОКВЭД2
    "Y48423006": ("data_section32", None),   # фонд заработной платы (без СМП), по ОКВЭД2
    "Y48112027": ("data_section31", None),   # оценка численности населения на 1 января
    "Y48112014": ("data_section31", AGE),    # население по полу и возрасту → итоги и группы
    "Y48002001": ("data_section2", None),    # число объектов розничной торговли и общепита
    "Y48112023": ("data_section31", AGE),    # миграционный прирост
}


def read_filtered(zf, name, age_rx):
    with zf.open(name) as f:
        header = f.readline().decode("utf-8-sig").strip().split(";")
    opts = dict(read_options=pacsv.ReadOptions(column_names=header, skip_rows=1, block_size=1 << 26),
                parse_options=pacsv.ParseOptions(delimiter=";"),
                convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in header}))
    kept, n_in = [], 0
    with zf.open(name) as f:
        for batch in pacsv.open_csv(f, **opts):
            n_in += batch.num_rows
            m = pc.and_(pc.match_substring(batch["mun_level"], "верхнего"), pc.is_in(batch["year"], pa.array(YEARS)))
            if age_rx and "vozr" in header:
                m = pc.and_(m, pc.match_substring_regex(batch["vozr"], age_rx))
            b = batch.filter(m)
            if b.num_rows:
                kept.append(pa.Table.from_batches([b]))
    return kept, n_in


def main(folder: pathlib.Path):
    root = pathlib.Path(__file__).resolve().parents[1]
    out_dir = root / "data" / "external" / "bdmo" / "parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for code, (arch, age_rx) in WANT.items():
        target = out_dir / f"{code}.parquet"
        if target.exists():
            print(code, "уже извлечён"); continue
        z = next(folder.glob(f"{arch}_*.zip"))
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if n.startswith(f"data_{code}_parts/") and n.endswith(".csv")]
            by_year = [n for n in names if re.search(r"_year\d{4}_", n)]
            if by_year:
                names = [n for n in by_year if re.search(r"_year(" + "|".join(YEARS) + r"|9999)_", n)]
            tables = []
            for k, name in enumerate(names, 1):
                kept, n_in = read_filtered(zf, name, age_rx)
                tables += kept
                print(f"  {code} [{k}/{len(names)}] {name.split('/')[-1]}: {n_in:,} строк, {time.time() - t0:.0f} с", flush=True)
        if tables:
            pq.write_table(pa.concat_tables(tables), target)
    allt = pd.concat([pq.read_table(p).to_pandas() for p in sorted(out_dir.glob("Y*.parquet"))], ignore_index=True)
    allt.to_parquet(out_dir.parent / "bdmo_2021_2024.parquet", index=False)
    print("итог:", allt.shape)
    print(allt.groupby(["indicator_code", "year"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
