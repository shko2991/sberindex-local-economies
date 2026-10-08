"""Получение и проверка исходных данных. Исходные данные в репозиторий не входят: архив конкурса СберИндекса
скачивается с официального адреса, справочник МО и муниципальная статистика — вручную; скрипт сверяет
контрольные суммы с версией, на которой получены результаты (data/raw/hackathon/README.md).

  python src/download.py                 — архив конкурса (скачивается с официального адреса)
  python src/download.py --dict PATH     — справочник МО СберИндекса (архив .rar/.zip, скачанный вручную)
  python src/download.py --check         — проверить наличие и контрольные суммы всех файлов

Справочник МО и муниципальная статистика скачиваются вручную (страницы с выбором файлов):
  справочник:  https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities
  статистика:  https://tochno.st/datasets/bdmo — разделы «Занятость и заработная плата»,
               «Население», «Розничная торговля и общественное питание» → data/external/bdmo/
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from common import get_logger, load_config, path

log = get_logger("download")

SHA256 = {
    "consumption.parquet": "9833ddaaee7b2a182ed4cceeed16469031700ea87d508bd976150c6770ef8a61",
    "connection.parquet": "20cbd5213d3ac1d0a867f097b485431811614b283d7366eb5cba9f65dc80493d",
    "market_access.parquet": "434258afe322b7e6e6610b2552d129ae47094613de72dae3d13f6965a28d1dc1",
    "t_dict_municipal_districts.xlsx": "4150658c3298fbc87ed79838f503f3a5b9a28da33257ff803c7d9231ddb775d6",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_hackathon(cfg: dict) -> None:
    raw = path(cfg, "raw", "x").parent
    zpath = raw / "hackathonlicence.zip"
    if not zpath.exists():
        log.info("скачиваю %s", cfg["data"]["source_url"])
        with urllib.request.urlopen(cfg["data"]["source_url"], timeout=120) as r, open(zpath, "wb") as f:
            shutil.copyfileobj(r, f)
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            base = Path(name).name
            if base.endswith(".parquet"):
                with z.open(name) as src, open(raw / base, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    log.info("архив конкурса распакован в %s", raw)


def unpack_dict(cfg: dict, archive: Path) -> None:
    archive = Path(archive).resolve()          # до смены каталога: относительный путь иначе сломается
    out = path(cfg, "external", cfg["reference"]["table"]).parent
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
    else:
        import libarchive              # pip install libarchive-c (нужна системная libarchive)
        import os
        cwd = os.getcwd()
        os.chdir(out)
        try:
            libarchive.extract_file(str(archive))
        finally:
            os.chdir(cwd)
    log.info("справочник распакован в %s", out)


def check(cfg: dict, allow_new: bool = False) -> bool:
    """Наличие файлов и совпадение контрольных сумм с версией, на которой получены результаты.
    Несовпадение — ошибка (другая версия данных даст другие числа), если не задан --allow-new-data."""
    ok = True
    files = {n: path(cfg, "raw", n) for n in ("consumption.parquet", "connection.parquet", "market_access.parquet")}
    files["t_dict_municipal_districts.xlsx"] = path(cfg, "external", cfg["reference"]["table"])
    files["t_dict_municipal_districts_poly.gpkg"] = path(cfg, "external", cfg["reference"]["polygons"])
    for n, p in files.items():
        if not p.exists():
            log.error("нет файла: %s", p); ok = False; continue
        if n in SHA256 and sha256(p) != SHA256[n]:
            if allow_new:
                log.warning("контрольная сумма отличается (новая версия данных, разрешено флагом): %s", p)
            else:
                log.error("контрольная сумма отличается: %s — это другая версия данных; "
                          "запустите с --allow-new-data, если так и задумано", p)
                ok = False
        else:
            log.info("ок: %s", p.name)
    bdmo = path(cfg, "external", "bdmo", "x").parent
    log.info("муниципальная статистика: %s", "есть файлы" if any(bdmo.iterdir()) else "нет (слой рынка труда пропускается)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", type=Path, help="архив справочника МО СберИндекса")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--allow-new-data", action="store_true", help="не считать ошибкой другие контрольные суммы")
    a = ap.parse_args()
    cfg = load_config()
    if a.check:
        sys.exit(0 if check(cfg, a.allow_new_data) else 1)
    if a.dict:
        unpack_dict(cfg, a.dict)
    else:
        fetch_hackathon(cfg)
    check(cfg, a.allow_new_data)


if __name__ == "__main__":
    main()
