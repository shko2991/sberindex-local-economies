"""Сборка reports/report.pdf из reports/report.md.

Markdown → HTML (библиотека markdown) → PDF (Chromium через Playwright). Ссылки «Автор, год» в тексте
становятся внутренними ссылками на записи списка литературы (раздел 10), DOI и адреса — внешними
ссылками, рисунки получают подписи «Рис. N». Обложка — звёздная карта МО со страницы проекта
(docs/index.html), поэтому страницу нужно собрать раньше.

  pip install markdown playwright      # Chromium: python -m playwright install chromium
  python reports/build_pdf.py          → reports/report.pdf (основной отчёт) и копия в docs/ для GitHub Pages

Шрифт: HSE Sans (НИУ ВШЭ, https://it.hse.ru/en/hsesans/) — встраивается в PDF только для просмотра и
печати, в репозиторий не входит; он должен быть установлен в системе. Без него — PT Sans.
"""
from __future__ import annotations

import base64
import html
import re
import shutil
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
DOCS = [  # исходник, PDF, подпись серии на обложке и в колонтитуле
    ("report.md", "report.pdf", "Методологический отчёт"),
]
TITLE = "Типы локальных экономик муниципалитетов"
SUBTITLE = "Безналичное потребление и рынок труда"
RUBRICS = ["Данные", "Сеть", "Методы", "Типы", "Рынок труда", "Динамика", "Литература"]
ALIASES = {"SNA": ("european", 2008), "Калински": ("caliński", 1974)}
# ссылки без «автор, год»: фраза в тексте → фрагмент записи списка литературы
PHRASES = {"формы 1-ТОРГ (МО)": "1-ТОРГ", "1-ТОРГ (МО)": "1-ТОРГ",
           "методологическим пояснениям Росстата": "Методологические пояснения",
           "«Муниципальная статистика России с 2005 года»": "Муниципальная статистика России"}


def split_bibliography(md: str):
    """Раздел 10 → HTML-список с якорями ref-N; ключ записи — (фамилия первого автора, год)."""
    m0 = re.search(r"^## (?:\d+\. )?Литература[^\n]*", md, flags=re.M)
    i0, heading = m0.start(), m0.group(0)[3:]
    nxt = md.find("\n## ", i0 + 5)
    i1 = nxt if nxt >= 0 else len(md)
    refs, out, para, n = {}, [], [], 0

    def flush():
        if para:
            out.append(markdown.markdown(" ".join(para)))
            para.clear()
    for ln in md[i0:i1].splitlines()[1:]:
        if ln.startswith("- "):
            flush()
            n += 1
            entry = ln[2:]
            first = "von Luxburg" if entry.startswith("von ") else entry.split()[0].strip(",.«»\"")
            year = re.search(r"\b(1[89]\d\d|20\d\d)\b", entry)
            key = (first.lower(), int(year.group(1))) if year else None
            if key and key not in refs:
                refs[key] = f"ref-{n}"
            refs.setdefault("_text", {})[f"ref-{n}"] = entry
            entry = re.sub(r"doi:(\S+?)(?=[\s,;]|$)", lambda m: f"[doi:{m.group(1)}](https://doi.org/{m.group(1)})", entry)
            entry = re.sub(r"(?<![(<])(https?://[^\s)]+)", r"<\1>", entry)
            inner = markdown.markdown(entry)[3:-4]
            out.append(f'<p class="bibitem" id="ref-{n}"><span class="n">{n}.</span>{inner}</p>')
        elif ln.startswith("**") and ln.rstrip().endswith("**"):
            flush()
            out.append(f'<p class="bibgroup">{html.escape(ln.strip().strip("*"))}</p>')
        elif ln.strip():
            para.append(ln.strip())
        else:
            flush()
    flush()
    bib_html = f'<section class="bib"><h2 id="literatura">{html.escape(heading)}</h2>' + "\n".join(out) + "</section>"
    return md[:i0], bib_html, md[i1:], refs


def normalize_lists(md: str) -> str:
    """Python-Markdown строже GitHub: список должен отделяться пустой строкой, вложенный — 4 пробелами."""
    out, prev = [], ""
    item = re.compile(r"^(\s*)([-*]|\d+\.) ")
    in_code = False
    for ln in md.splitlines():
        if ln.startswith("```"):
            in_code = not in_code
        if not in_code:
            m = item.match(ln)
            if m and len(m.group(1)) in (2, 3):
                ln = "    " + ln.lstrip()
            elif m is None and re.match(r"^  {2}\S", ln) and out and item.match(out[-1].lstrip() and out[-1]) and out[-1].startswith("    "):
                ln = "      " + ln.lstrip()
            if m and prev.strip() and not item.match(prev) and not prev.startswith((" ", ">", "|")):
                out.append("")
        out.append(ln)
        prev = ln
    return "\n".join(out)


def link_citations(text: str, refs: dict) -> str:
    """«Фамилия … год» → ссылка на запись списка. Защищаем код, ссылки и HTML."""
    keep = []

    def stash(m):
        keep.append(m.group(0)); return f"\x00{len(keep) - 1}\x00"
    text = re.sub(r"```.*?```|`[^`\n]*`|\]\([^)\n]*\)|<[^>\n]+>", stash, text, flags=re.S)
    names = {}
    texts = refs.get("_text", {})
    for phrase, frag in PHRASES.items():
        rid = next((k for k, v in texts.items() if frag in v), None)
        if rid:
            def repl_p(m, rid=rid):
                keep.append(f'<a class="cite" href="#{rid}">{m.group(0)}</a>')
                return f"\x00{len(keep) - 1}\x00"
            text = re.sub(re.escape(phrase), repl_p, text)
    for key, rid in refs.items():
        if key == "_text":
            continue
        name, year = key
        names.setdefault(name, []).append((year, rid))
    for alias, key in ALIASES.items():
        if key in refs:
            names.setdefault(alias.lower(), []).append((key[1], refs[key]))
    for name in sorted(names, key=len, reverse=True):
        disp = re.escape(name)
        for year, rid in names[name]:
            pat = re.compile(rf"(?<![\w\x00])({disp}(?:и др\.|et al\.|[^.;()\[\]\x00]){{0,60}}?\(?{year}\)?)(?!\d)", re.I)

            def repl(m):
                keep.append(f'<a class="cite" href="#{rid}">{m.group(1)}</a>')
                return f"\x00{len(keep) - 1}\x00"
            text = pat.sub(repl, text)
    # повторный год того же автора сразу после ссылки: «Aitchison, 1982, 1986»
    def follow(m):
        tok = keep[int(m.group(1))]
        nm = re.search(r'>([^<\s,]+)', tok)
        key = (nm.group(1).lower(), int(m.group(3))) if nm else None
        if key in refs:
            keep.append(f'<a class="cite" href="#{refs[key]}">{m.group(3)}</a>')
            return f"\x00{m.group(1)}\x00{m.group(2)}\x00{len(keep) - 1}\x00"
        return m.group(0)
    text = re.sub(r"\x00(\d+)\x00([,;] ?)(1[89]\d\d|20\d\d)\b", follow, text)
    while "\x00" in text:
        text = re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], text)
    return text


def figures(h: str) -> str:
    n = 0

    def repl(m):
        nonlocal n
        n += 1
        alt, src = html.unescape(m.group(1)), m.group(2)
        data = base64.b64encode((ROOT / "reports" / src).read_bytes()).decode()
        return (f'<figure><figcaption><b>Рис. {n}.</b> {html.escape(alt)}</figcaption>'
                f'<img src="data:image/png;base64,{data}" alt="{html.escape(alt)}"></figure>')
    return re.sub(r'<p><img alt="([^"]*)" src="([^"]+)" ?/?></p>', repl, h)


def cover_image(page) -> str:
    page.goto((ROOT / "docs" / "index.html").as_uri())
    page.wait_for_timeout(3500)
    page.evaluate("document.querySelector('.g-top').style.visibility='hidden'; document.getElementById('gtip').hidden=true")
    page.wait_for_timeout(300)
    shot = page.locator("#sky").screenshot()
    return base64.b64encode(shot).decode()


CSS = r"""
@page { size: A4; margin: 16mm 15mm 17mm 15mm; }
@page :first { margin-top: 0; }
:root { --ink: #1b1733; --ink2: #4a4566; --vio: #5b2fc9; --vio2: #7c4dff; --lav: #f1ecff; --rule: #d9d2f0; --night: #120d38; }
* { box-sizing: border-box; }
html { font-family: "HSE Sans", "PT Sans", "Segoe UI", Arial, sans-serif; font-size: 10.6pt; color: var(--ink); line-height: 1.4; }
body { margin: 0; }
a { color: var(--vio); text-decoration: none; }
a.cite { color: var(--vio); border-bottom: 0.5pt dotted var(--vio2); }
.cover { margin: 0 -15mm 6mm; padding: 14mm 15mm 9mm; background: radial-gradient(ellipse 60% 80% at 80% 40%, #3a2290 0%, #1a1150 45%, var(--night) 75%);
  color: #fff; position: relative; overflow: hidden; min-height: 92mm; }
.cover img { position: absolute; right: -12mm; top: 2mm; width: 130mm; opacity: 1; mix-blend-mode: screen;
  -webkit-mask-image: radial-gradient(ellipse 62% 62% at 50% 50%, #000 50%, transparent 78%); mask-image: radial-gradient(ellipse 62% 62% at 50% 50%, #000 50%, transparent 78%); }
.cover .series { font-size: 9pt; color: #d8ccff; letter-spacing: 0.02em; }
.cover h1 { font-family: "HSE Sans", "PT Sans", Arial, sans-serif; font-weight: 700; font-size: 25pt; line-height: 1.05; text-transform: uppercase; margin: 6mm 0 2mm; max-width: 98mm;
  text-shadow: 0 0 10px rgba(180, 140, 255, 0.6); }
.cover .sub { font-family: "HSE Sans", "PT Sans", sans-serif; font-style: italic; font-size: 13pt; color: #e9e1ff; max-width: 100mm; }
.chips { display: flex; flex-wrap: wrap; gap: 2mm; margin-top: 6mm; max-width: 110mm; }
.chips span { border: 0.6pt solid rgba(255,255,255,0.7); border-radius: 4mm; padding: 0.8mm 3mm; font-size: 8.4pt; text-transform: uppercase; letter-spacing: 0.03em; }
.chips span:nth-child(4) { background: #fff; color: var(--night); font-weight: 700; }
.badge { display: inline-block; margin-top: 6mm; background: #fff; color: var(--night); font-weight: 700; padding: 1.4mm 4mm; border-radius: 1.5mm; font-size: 10pt; }
blockquote { margin: 0 0 4mm; padding: 3mm 4mm; background: var(--lav); border-left: 1.2mm solid var(--vio2); font-size: 10pt; text-align: justify; }
blockquote p { margin: 0; }
blockquote p + p { margin-top: 2mm; color: var(--ink2); font-size: 9pt; }
h1.doc { display: none; }
h2 { font-size: 13pt; color: var(--vio); text-transform: uppercase; margin: 7mm 0 2.5mm; letter-spacing: 0.01em; break-after: avoid; }
h3 { font-size: 11pt; color: var(--ink); margin: 5mm 0 2mm; break-after: avoid; }
h4 { font-size: 10.6pt; color: var(--vio); margin: 4mm 0 1.5mm; break-after: avoid; }
p { margin: 0 0 2.2mm; text-align: justify; hyphens: auto; }
ul, ol { margin: 0 0 2.5mm; padding-left: 5.5mm; }
li { margin-bottom: 1mm; text-align: justify; hyphens: auto; }
li::marker { color: var(--vio2); }
hr { border: 0; border-top: 0.6pt solid var(--rule); margin: 5mm 0; }
code { font-family: "PT Mono", Consolas, monospace; font-size: 8.6pt; background: #f4f2fa; padding: 0 0.8mm; border-radius: 0.6mm; word-break: break-word; }
pre { background: #f4f2fa; border: 0.5pt solid var(--rule); padding: 3mm; font-size: 8pt; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 4mm; font-size: 8.8pt; line-height: 1.25; break-inside: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th { background: var(--lav); color: var(--night); text-align: left; padding: 1.4mm 1.6mm; border-bottom: 0.8pt solid var(--vio2); vertical-align: bottom; }
td { padding: 1.2mm 1.6mm; border-bottom: 0.4pt solid var(--rule); vertical-align: top; }
td code { font-size: 7.8pt; word-break: normal; overflow-wrap: anywhere; }
td:first-child code { white-space: nowrap; overflow-wrap: normal; }
figure { margin: 3mm 0 5mm; break-inside: avoid; }
figure img { width: 100%; max-height: 105mm; object-fit: contain; display: block; }
figcaption { font-size: 9.2pt; color: var(--vio); margin-bottom: 1.5mm; }
figcaption b { color: var(--vio); }
.toc { columns: 2; column-gap: 8mm; font-size: 9pt; margin: 0 0 4mm; padding: 3mm 4mm; border: 0.6pt solid var(--rule); border-radius: 1.5mm; break-inside: avoid; }
.toc div { break-inside: avoid; margin-bottom: 0.8mm; }
.toc .t { font-weight: 700; color: var(--vio); column-span: all; margin-bottom: 1.5mm; text-transform: uppercase; font-size: 9.4pt; }
.bib ol, .bib p { font-size: 9pt; }
p.bibgroup { font-weight: 700; color: var(--vio); margin: 3mm 0 1mm; text-align: left; }
.bibitem { margin: 0 0 1.2mm; padding-left: 8mm; text-indent: -8mm; text-align: left; font-size: 9pt; }
.bibitem .n { display: inline-block; width: 8mm; text-indent: 0; color: var(--vio); font-weight: 700; }
.bibitem:target { background: var(--lav); }
"""


def render(b, md_name: str, pdf_name: str, series: str, img: str):
    md = normalize_lists((ROOT / "reports" / md_name).read_text(encoding="utf-8"))
    head, bib_html, tail, refs = split_bibliography(md)
    exts = ["tables", "fenced_code", "sane_lists", "toc", "attr_list"]
    counter = iter(range(1, 10_000))
    slug = lambda v, sep: f"s{next(counter)}"          # короткие якоря: длинные кириллические имена ломают PDF
    conv = lambda t: markdown.markdown(t, extensions=exts, extension_configs={"toc": {"slugify": slug}})
    body_html = conv(link_citations(head, refs)) + bib_html + conv(link_citations(tail, refs))
    # убрать заголовок документа: он на обложке
    body_html = re.sub(r"<h1[^>]*>.*?</h1>\s*<p>.*?</p>", "", body_html, count=1, flags=re.S)
    body_html = figures(body_html)
    toc = ['<div class="toc"><div class="t">Содержание</div>']
    for m in re.finditer(r'<h2 id="([^"]+)">(.*?)</h2>', body_html):
        toc.append(f'<div><a href="#{m.group(1)}">{m.group(2)}</a></div>')
    toc.append("</div>")
    cover = (f'<header class="cover"><img src="data:image/png;base64,{img}" alt="">'
             f'<div class="series">{series} · Конкурс СберИндекса, направление «Кластеризация»</div>'
             f'<h1>{TITLE}</h1><div class="sub">{SUBTITLE}, 2023–2024 гг.</div>'
             f'<div class="chips">{"".join(f"<span>{r}</span>" for r in RUBRICS)}</div>'
             f'<div class="badge">Версия от 8 октября 2026 г.</div></header>')
    # аннотация (первые цитаты) идут сразу после обложки, затем содержание
    quotes = re.findall(r"<blockquote>.*?</blockquote>", body_html, flags=re.S)[:2]
    for q in quotes:
        body_html = body_html.replace(q, "", 1)
    doc = (f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>{TITLE}. {series}</title><style>{CSS}</style></head>'
           f"<body>{cover}{''.join(quotes)}{''.join(toc)}{body_html}</body></html>")
    tmp = ROOT / "reports" / ".report_print.html"
    tmp.write_text(doc, encoding="utf-8")
    page = b.new_page()
    page.goto(tmp.as_uri())
    page.wait_for_timeout(500)
    footer = ('<div style="width:100%;font-family:HSE Sans,PT Sans,Arial,sans-serif;font-size:7.5pt;color:#6b6489;padding:0 15mm;'
              f'display:flex;justify-content:space-between;"><span>{TITLE}. {series}. Конкурс СберИндекса, 2026</span>'
              '<span class="pageNumber"></span></div>')
    out = ROOT / "reports" / pdf_name
    page.pdf(path=str(out), prefer_css_page_size=True, print_background=True, display_header_footer=True,
             header_template="<span></span>", footer_template=footer)
    tmp.unlink()
    (ROOT / "docs").mkdir(exist_ok=True)
    shutil.copyfile(out, ROOT / "docs" / pdf_name)          # для ссылок со страницы на GitHub Pages
    print("PDF:", out.relative_to(ROOT), round(out.stat().st_size / 1e6, 2), "МБ")


def build():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        img = cover_image(b.new_page(viewport={"width": 1440, "height": 900}))
        for md_name, pdf_name, series in DOCS:
            render(b, md_name, pdf_name, series, img)
        b.close()


if __name__ == "__main__":
    sys.exit(build())
