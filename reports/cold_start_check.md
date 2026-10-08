# Проверка запуска с пустого каталога результатов

**Дата:** 5 октября 2026 г. **Коммит:** `6396771`.

**Порядок проверки:**

1. Репозиторий клонирован в отдельный каталог (`git clone`). Каталоги `data/processed`,
   `data/interim`, `reports/figures` и файл `docs/index.html` очищены. Исходные данные —
   архив конкурса, справочник МО и выгрузка Росстата — подключены как есть.
2. Выполнен `python run_all.py` целиком: проверка данных, 27 тестов, сравнение 280 вариантов и все
   последующие шаги. Время — 69 минут на 2 ядрах. Ошибок нет; шаг `robustness.py` выполнен до
   появления `summary.json`.
3. Результаты сверены с опубликованными скриптом `src/compare_runs.py`. Он сравнивает таблицы вместе
   с индексом (`territory_id`). Исторический каталог `labor_v1/` новый расчёт не создаёт, поэтому
   он исключён из сверки.

**Итог:**
- `data/processed` — все 49 файлов побайтно совпадают (включая `landing_data.json`: он есть в обоих каталогах, но в git не хранится — исключён в `.gitignore`);
- `reports/figures` — все 8 рисунков побайтно совпадают;
- `docs/index.html` и геометрия карты (`data/interim/mo.topo.json`) побайтно совпадают.

Подробные таблицы сверки:

## Сверка: прогон с пустого каталога против опубликованных результатов (data/processed)

А: `чистый прогон: data/processed`  
Б: опубликованный `data/processed`

Итого: побайтно совпадает — 49

| Файл | Результат | Подробности |
|---|---|---|
| `candidates.csv` | побайтно совпадает |  |
| `cases.csv` | побайтно совпадает |  |
| `community_strength.csv` | побайтно совпадает |  |
| `cons_vs_labor.csv` | побайтно совпадает |  |
| `cons_vs_labor_no_federal.csv` | побайтно совпадает |  |
| `descriptives.json` | побайтно совпадает |  |
| `edge_overlap.csv` | побайтно совпадает |  |
| `excluded_profile.csv` | побайтно совпадает |  |
| `external_validation.csv` | побайтно совпадает |  |
| `external_validation_rosstat.csv` | побайтно совпадает |  |
| `fca_descriptions.csv` | побайтно совпадает |  |
| `final_types.csv` | побайтно совпадает |  |
| `finalists.csv` | побайтно совпадает |  |
| `finalists_crosstab.csv` | побайтно совпадает |  |
| `gap_model.csv` | побайтно совпадает |  |
| `graph_summary.csv` | побайтно совпадает |  |
| `kefrin_restarts.csv` | побайтно совпадает |  |
| `kefrin_xi.json` | побайтно совпадает |  |
| `labels_candidates.parquet` | побайтно совпадает |  |
| `labels_sensitivity.parquet` | побайтно совпадает |  |
| `labor_age_exclusions.csv` | побайтно совпадает |  |
| `labor_diagnostics.json` | побайтно совпадает |  |
| `labor_mismatch.csv` | побайтно совпадает |  |
| `labor_oktmo_map.csv` | побайтно совпадает |  |
| `labor_passport.csv` | побайтно совпадает |  |
| `labor_population_outliers.csv` | побайтно совпадает |  |
| `labor_profiles.csv` | побайтно совпадает |  |
| `labor_ranking.csv` | побайтно совпадает |  |
| `labor_sector_rounding.csv` | побайтно совпадает |  |
| `labor_sector_violations.csv` | побайтно совпадает |  |
| `labor_summary.json` | побайтно совпадает |  |
| `labor_table.csv` | побайтно совпадает |  |
| `landing_data.json` | побайтно совпадает |  |
| `layer_mass.json` | побайтно совпадает |  |
| `leadlag_regions.csv` | побайтно совпадает |  |
| `mo_reliability.csv` | побайтно совпадает |  |
| `moran.csv` | побайтно совпадает |  |
| `network_value.csv` | побайтно совпадает |  |
| `ranking_main.csv` | побайтно совпадает |  |
| `ranking_sensitivity_top5.csv` | побайтно совпадает |  |
| `robust_indices.csv` | побайтно совпадает |  |
| `robustness.json` | побайтно совпадает |  |
| `sensitivity.csv` | побайтно совпадает |  |
| `summary.json` | побайтно совпадает |  |
| `synthetic_benchmark.csv` | побайтно совпадает |  |
| `transition_yearly.csv` | побайтно совпадает |  |
| `type_profiles.csv` | побайтно совпадает |  |
| `type_stability.csv` | побайтно совпадает |  |
| `types_quarterly.csv` | побайтно совпадает |  |

## Рисунки

А: `чистый прогон: reports/figures`  
Б: `reports/figures`

Итого: побайтно совпадает — 8

| Файл | Результат | Подробности |
|---|---|---|
| `cons_vs_labor.png` | побайтно совпадает |  |
| `edge_overlap.png` | побайтно совпадает |  |
| `gap_model.png` | побайтно совпадает |  |
| `icvi_corr.png` | побайтно совпадает |  |
| `mismatch_maps.png` | побайтно совпадает |  |
| `moran.png` | побайтно совпадает |  |
| `pareto_sw_mq.png` | побайтно совпадает |  |
| `types_map.png` | побайтно совпадает |  |
