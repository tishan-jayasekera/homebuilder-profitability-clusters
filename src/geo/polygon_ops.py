from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon


def union_polygons(polygons):
    if not polygons:
        return None
    return unary_union(polygons)


def centroid_in(polygon, suburb_geom) -> bool:
    return polygon.contains(suburb_geom.centroid)


def intersection_ratio(polygon, suburb_geom) -> float:
    if suburb_geom.is_empty:
        return 0.0
    inter = polygon.intersection(suburb_geom)
    if inter.is_empty:
        return 0.0
    denom = suburb_geom.area
    return float(inter.area / denom) if denom > 0 else 0.0
