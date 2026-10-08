# Данные СберИндекса: архив конкурса

Сюда распаковывается архив конкурса СберИндекса `hackathonlicence.zip`. Сами данные в репозиторий не
входят: архив скачивает с официального адреса команда `python src/download.py`, контрольные суммы
проверяет `python src/download.py --check`. Описание полей и условия использования — в файле
`Данные_СберИндекс_лицензия.pdf` из того же архива.

**Лицензия.** Данные Лаборатории СберИндекс доступны по международной лицензии CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/legalcode.ru). Производные материалы этого
репозитория распространяются на тех же условиях (`DATA_LICENSE.md`).

**Цитирование** (в формате, указанном СберИндексом):

- Потребительские безналичные расходы на уровне муниципальных образований по категориям трат.
  СберИндекс. Данные доступны по адресу
  https://sberindex.ru/ru/research/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim
  (данные скачаны 03.10.2026). — `consumption.parquet`
- Индекс доступности рынков на уровне муниципальных образований. СберИндекс. Данные доступны по
  адресу https://sberindex.ru/ru/research/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim
  (данные скачаны 03.10.2026). — `market_access.parquet`
- Автодорожные и железнодорожные связи между муниципальными образованиями. СберИндекс. Данные
  доступны по адресу https://sberindex.ru/ru/research/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim
  (данные скачаны 03.10.2026). — `connection.parquet`

| Файл | SHA-256 |
|---|---|
| `consumption.parquet` | `9833ddaaee7b2a182ed4cceeed16469031700ea87d508bd976150c6770ef8a61` |
| `connection.parquet` | `20cbd5213d3ac1d0a867f097b485431811614b283d7366eb5cba9f65dc80493d` |
| `market_access.parquet` | `434258afe322b7e6e6610b2552d129ae47094613de72dae3d13f6965a28d1dc1` |
| `Данные_СберИндекс_лицензия.pdf` | `030269e726f969250b911a1fcb72131a4b61c3952aad10c8470ce9c0736c472f` |
