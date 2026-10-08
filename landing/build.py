"""Сборка страницы: данные из data/processed/landing_data.json встраиваются в шаблон.
Получается один самодостаточный файл landing/index.html (без внешних запросов к данным) —
его можно открыть локально, выложить на GitHub Pages или опубликовать как страницу.

  python landing/build.py
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import load_config, path  # noqa: E402


def main():
    cfg = load_config()
    data = json.loads(path(cfg, "processed", "landing_data.json").read_text(encoding="utf-8"))
    tpl = (ROOT / "landing" / "template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    body = tpl.replace("/*__DATA__*/null", payload)
    # библиотеки встраиваются в страницу: она работает без интернета (лицензии: d3 — ISC, topojson — BSD-3)
    for url, local in (("https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js", "d3.min.js"),
                       ("https://cdnjs.cloudflare.com/ajax/libs/topojson/3.0.2/topojson.min.js", "topojson.min.js")):
        code = (ROOT / "landing" / "vendor" / local).read_text(encoding="utf-8").replace("</script", "<\\/script")
        body = body.replace(f'<script src="{url}"></script>', f"<script>{code}</script>")
    (ROOT / "landing" / "index.html").write_text(body, encoding="utf-8")
    # для GitHub Pages — полноценный документ со своим заголовком
    page = ("<!doctype html>\n<html lang=\"ru\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            "<style>html{color-scheme:light dark}body{margin:0}[hidden]{display:none!important}</style>\n</head>\n<body>\n"
            + body + "\n</body>\n</html>\n")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "index.html").write_text(page, encoding="utf-8")
    print("landing/index.html и docs/index.html:", round(len(body.encode()) / 1e6, 2), "МБ")


if __name__ == "__main__":
    main()
