"""Версионный справочник МО СберИндекса (t_dict_municipal_districts, покрытие 2018–01.01.2024).

territory_id — МО в постоянных границах (ключ данных конкурса). У одного territory_id может быть
несколько версий (смена ОКТМО, названия, типа, статуса, границ). Для подписей берётся последняя
версия; для сопоставления со статистикой Росстата — все версии ОКТМО с годами действия.
Координаты центра для городов федерального значения не заданы — берётся центроид полигона.
Три МО конкурса (Павловский Посад, Электрогорск, Сасово) объединены с соседями в 2024 г. и в
актуальной версии отсутствуют — для них используется последняя существовавшая версия.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import get_logger, path

log = get_logger("reference")


def _digits(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\D", "", regex=True)


def load_versions(cfg: dict) -> pd.DataFrame:
    d = pd.read_excel(path(cfg, "external", cfg["reference"]["table"]), dtype={"oktmo": str, "shape_linked_oktmo": str})
    d["oktmo8"] = _digits(d.oktmo).str[:8]
    return d


def load_reference(cfg: dict, versions: pd.DataFrame | None = None) -> pd.DataFrame:
    d = load_versions(cfg) if versions is None else versions
    last = d.sort_values(["territory_id", "year_from"]).groupby("territory_id").tail(1).set_index("territory_id")
    ref = last[["municipal_district_name", "municipal_district_name_short", "municipal_district_type",
                "municipal_district_status", "municipal_district_center", "region_code", "region_name",
                "municipal_district_center_lat", "municipal_district_center_lon", "oktmo8", "year_to"]].rename(
        columns={"municipal_district_name": "name", "municipal_district_name_short": "name_short",
                 "municipal_district_type": "mo_type", "municipal_district_status": "status",
                 "municipal_district_center": "center", "municipal_district_center_lat": "lat",
                 "municipal_district_center_lon": "lon"})
    ref["active_2024"] = ref.year_to > 2024
    ref["federal_city"] = ref.region_name.isin(["Москва", "Санкт-Петербург", "Севастополь"])
    if ref.lat.isna().any():
        cen = polygon_centroids(cfg)
        miss = ref.lat.isna() & ref.index.isin(cen.index)
        ref.loc[miss, "lat"] = cen.loc[ref.index[miss], "lat"].values
        ref.loc[miss, "lon"] = cen.loc[ref.index[miss], "lon"].values
        log.info("координаты центра из центроидов полигонов: %d МО", int(miss.sum()))
    return ref.drop(columns="year_to")


def oktmo_versions(versions: pd.DataFrame) -> pd.DataFrame:
    """Все пары (ОКТМО-8, territory_id) с годами действия — для привязки данных Росстата."""
    v = versions[["oktmo8", "territory_id", "year_from", "year_to"]].drop_duplicates()
    amb = v.groupby("oktmo8").territory_id.nunique()
    if (amb > 1).any():
        log.warning("ОКТМО, соответствующих нескольким territory_id: %d (разные годы)", int((amb > 1).sum()))
    return v


def load_polygons(cfg: dict, simplify_m: float | None = None):
    import geopandas as gpd
    g = gpd.read_file(path(cfg, "external", cfg["reference"]["polygons"]))
    g["territory_id"] = g.territory_id.astype(int)
    g = g.sort_values(["territory_id", "year_from"]).groupby("territory_id").tail(1)
    if simplify_m:
        g = g.to_crs(3576)                     # равноплощадная полярная проекция для России
        g["geometry"] = g.geometry.simplify(simplify_m, preserve_topology=True)
        g = g.to_crs(4326)
    return g[["territory_id", "geometry"]].set_index("territory_id")


def polygon_centroids(cfg: dict) -> pd.DataFrame:
    g = load_polygons(cfg)
    c = g.to_crs(3576).geometry.centroid.to_crs(4326)
    return pd.DataFrame({"lat": c.y, "lon": c.x}, index=g.index)


def export_topojson(cfg: dict, ids=None, simplify_m: float = 1500, quant: float = 1e5) -> str:
    """Полигоны МО для веб-карты: упрощение, внешние кольца по часовой стрелке (так их
    понимает d3-geo на сфере; в GeoJSON принято против — иначе d3 закрашивает дополнение),
    TopoJSON с общими границами."""
    import topojson as tp
    from shapely.geometry.polygon import orient
    from shapely.geometry import MultiPolygon, Polygon
    g = load_polygons(cfg, simplify_m=simplify_m)
    g = (g if ids is None else g[g.index.isin(list(ids))]).reset_index()

    def cw(geom):
        if isinstance(geom, Polygon):
            return orient(geom, sign=-1.0)
        if isinstance(geom, MultiPolygon):
            return MultiPolygon([orient(p, sign=-1.0) for p in geom.geoms])
        return geom

    g["geometry"] = g.geometry.apply(cw)
    topo = tp.Topology(g[["territory_id", "geometry"]], prequantize=quant, topology=True)
    out = path(cfg, "interim", "mo.topo.json")
    out.write_text(topo.to_json(), encoding="utf-8")
    return str(out)
