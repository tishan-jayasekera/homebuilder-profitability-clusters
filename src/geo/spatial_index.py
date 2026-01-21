from dataclasses import dataclass

import pandas as pd
import shapely
import streamlit as st
from shapely.strtree import STRtree


@dataclass
class SpatialIndex:
    tree: STRtree
    geometries: list
    id_to_idx: dict


def _build_index(geoms: list) -> SpatialIndex:
    tree = STRtree(geoms)
    id_to_idx = {id(g): i for i, g in enumerate(geoms)}
    return SpatialIndex(tree=tree, geometries=geoms, id_to_idx=id_to_idx)


@st.cache_resource(show_spinner=False)
def build_spatial_index(gdf: pd.DataFrame, state: str) -> SpatialIndex:
    geoms = gdf["geometry"].tolist()
    return _build_index(geoms)


def query_candidates(index: SpatialIndex, polygon) -> list[int]:
    hits = index.tree.query(polygon)
    candidates = []
    for geom in hits:
        idx = index.id_to_idx.get(id(geom))
        if idx is not None:
            candidates.append(idx)
    return candidates
