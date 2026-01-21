import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import streamlit as st


DATA_DIR = Path("data")


def _normalize_suburb_name(value: str) -> str:
    if value is None:
        return ""
    return (
        str(value)
        .replace("(VIC)", "")
        .replace("(NSW)", "")
        .replace("(QLD)", "")
        .replace("(SA)", "")
        .replace("(WA)", "")
        .replace("(TAS)", "")
        .replace("(ACT)", "")
        .replace("(NT)", "")
        .strip()
        .upper()
    )


@st.cache_data(show_spinner=False)
def load_suburb_postcode_map() -> pd.DataFrame:
    csv_path = DATA_DIR / "suburb_postcode.csv"
    if not csv_path.exists():
        return pd.DataFrame(columns=["suburb_name", "state", "postcode"])
    df = pd.read_csv(csv_path)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df = df.rename(columns={
        "suburb": "suburb_name",
        "postcode": "postcode",
        "state": "state"
    })
    df["suburb_name"] = df["suburb_name"].map(_normalize_suburb_name)
    df["state"] = df["state"].astype(str).str.strip().str.upper()
    df["postcode"] = df["postcode"].astype(str).str.strip().str.zfill(4)
    df = df.dropna(subset=["suburb_name", "state", "postcode"])
    return df


@st.cache_data(show_spinner=False)
def load_suburb_boundaries(state: str | None = None) -> gpd.GeoDataFrame:
    geo_dir = DATA_DIR / "suburbs_geojson"
    if geo_dir.exists() and geo_dir.is_dir():
        if state:
            path = geo_dir / f"{state}.geojson"
            if not path.exists():
                return gpd.GeoDataFrame(columns=["suburb_name", "state", "geometry"], geometry="geometry")
            gdf = gpd.read_file(path)
        else:
            paths = list(geo_dir.glob("*.geojson"))
            if not paths:
                return gpd.GeoDataFrame(columns=["suburb_name", "state", "geometry"], geometry="geometry")
            gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(p) for p in paths], ignore_index=True))
    else:
        national = DATA_DIR / "suburbs.geojson"
        if not national.exists():
            return gpd.GeoDataFrame(columns=["suburb_name", "state", "geometry"], geometry="geometry")
        gdf = gpd.read_file(national)

    cols = {c.lower(): c for c in gdf.columns}
    suburb_col = cols.get("suburb_name") or cols.get("suburb") or cols.get("name")
    state_col = cols.get("state") or cols.get("state_code")
    if suburb_col:
        gdf["suburb_name"] = gdf[suburb_col].map(_normalize_suburb_name)
    else:
        gdf["suburb_name"] = ""
    if state_col:
        gdf["state"] = gdf[state_col].astype(str).str.strip().str.upper()
    else:
        gdf["state"] = ""
    gdf = gdf[gdf["suburb_name"] != ""]
    if state:
        gdf = gdf[gdf["state"] == state]
    gdf = gdf.set_geometry("geometry")
    gdf = gdf[~gdf["geometry"].isna()]
    return gdf[["suburb_name", "state", "geometry"]].reset_index(drop=True)
