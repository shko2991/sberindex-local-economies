# Встроенные библиотеки страницы

| Файл | Библиотека | Версия | Лицензия | Текст лицензии |
|---|---|---|---|---|
| `d3.min.js` | D3 (https://d3js.org) | 7.9.0 | ISC | `LICENSE-d3.txt` |
| `topojson.min.js` | TopoJSON (https://github.com/topojson/topojson), включает topojson-client, topojson-server, topojson-simplify | 3.0.2 | BSD-3-Clause | `LICENSE-topojson.txt` |

Файлы встраиваются в `docs/index.html` при сборке (`landing/build.py`), поэтому страница работает без
обращения к CDN.
