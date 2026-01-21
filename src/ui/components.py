from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from shapely.geometry import shape
import folium
from folium.plugins import Draw
from st_folium import st_folium
import cv2

from src.geo.reference_data import load_suburb_boundaries, load_suburb_postcode_map
from src.geo.spatial_index import build_spatial_index, query_candidates
from src.geo.polygon_ops import union_polygons, centroid_in, intersection_ratio
from src.image.segment_region import segment_region
from src.image.alignment import compute_affine, apply_affine, alignment_rms
from src.io.export import export_csv, export_json


STATE_CENTER = {
    "NSW": (-31.0, 147.0),
    "VIC": (-37.0, 144.0),
    "QLD": (-22.0, 144.0),
    "SA": (-30.0, 135.0),
    "WA": (-25.0, 121.0),
    "TAS": (-42.0, 147.0),
    "ACT": (-35.5, 149.0),
    "NT": (-19.0, 133.0),
}


def _kpi_row(items: list[tuple[str, str]]):
    st.markdown("<div class='kpi-row'>", unsafe_allow_html=True)
    for label, value in items:
        st.markdown(
            f"<div class='kpi'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div></div>",
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _normalize_suburb_name(value: str) -> str:
    return str(value).strip().upper()


def _map_suburb_to_postcodes(suburb_map: pd.DataFrame, state: str, suburb: str) -> list[str]:
    matches = suburb_map[
        (suburb_map["state"] == state) &
        (suburb_map["suburb_name"] == _normalize_suburb_name(suburb))
    ]
    if matches.empty:
        return []
    return matches["postcode"].unique().tolist()


def _build_results(builder_name, state, region_name, suburbs, method, confidence, notes, suburb_map):
    rows = []
    for suburb in suburbs:
        postcodes = _map_suburb_to_postcodes(suburb_map, state, suburb)
        if not postcodes:
            postcodes = [""]
        for pc in postcodes:
            rows.append({
                "builder_name": builder_name,
                "state": state,
                "region_name": region_name or "",
                "suburb": suburb,
                "postcode": pc,
                "method": method,
                "confidence": float(confidence),
                "notes": notes,
                "created_at": datetime.utcnow().isoformat()
            })
    if not rows:
        return pd.DataFrame(columns=[
            "builder_name", "state", "region_name", "suburb", "postcode",
            "method", "confidence", "notes", "created_at"
        ])
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["state", "suburb", "postcode"])
    df = df.sort_values(["postcode", "suburb"])
    return df


def render_build_region_extractor():
    st.markdown("""
    <div class="section-card">
        <div class="section-header">
            <span class="section-num">8</span>
            <span class="section-title">Build Region Extractor</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    builder_name = st.text_input("Builder name", key="bre_builder")
    state = st.selectbox("State", ["VIC", "NSW", "QLD", "SA", "WA", "TAS", "ACT", "NT"], key="bre_state")
    region_name = st.text_input("Region name (optional)", key="bre_region")
    mode = st.radio("Mode", ["Draw on Map", "Upload Screenshot"], horizontal=True, key="bre_mode")

    suburb_map = load_suburb_postcode_map()
    gdf = load_suburb_boundaries(state)
    if gdf.empty:
        st.error("Missing suburb boundary data. Expected files under data/suburbs_geojson/ or data/suburbs.geojson.")
        return

    index = build_spatial_index(gdf, state)
    results_df = st.session_state.get("bre_results", pd.DataFrame())

    if mode == "Draw on Map":
        st.markdown("**Draw build regions**")
        method = "drawn_polygon"
        use_intersection = st.checkbox("Use intersection threshold (>=5%)", value=False)
        rule = "area>=5%" if use_intersection else "centroid-in"

        center = STATE_CENTER.get(state, (-25.0, 133.0))
        m = folium.Map(location=center, zoom_start=5, tiles="cartodbpositron")
        Draw(export=True, draw_options={"polygon": True, "rectangle": True, "circle": False}).add_to(m)
        output = st_folium(m, height=500, width=None, returned_objects=["all_drawings"])
        polygons = []
        if output and output.get("all_drawings"):
            for feature in output["all_drawings"]:
                geom = shape(feature.get("geometry"))
                polygons.append(geom)
        if st.button("Clear drawings"):
            st.session_state["bre_results"] = pd.DataFrame()
            st.rerun()

        union = union_polygons(polygons)
        if union:
            candidate_idx = query_candidates(index, union)
            matches = []
            for idx in candidate_idx:
                suburb_geom = gdf.iloc[idx]["geometry"]
                suburb_name = gdf.iloc[idx]["suburb_name"]
                if use_intersection:
                    ratio = intersection_ratio(union, suburb_geom)
                    if ratio >= 0.05:
                        matches.append(suburb_name)
                else:
                    if centroid_in(union, suburb_geom):
                        matches.append(suburb_name)
            confidence = 0.85 if use_intersection else 0.95
            notes = rule
            results_df = _build_results(builder_name, state, region_name, sorted(set(matches)), method, confidence, notes, suburb_map)
            st.session_state["bre_results"] = results_df
        elif output and output.get("all_drawings"):
            st.warning("Drawn polygon is too small or invalid. Try a larger shape.")

    else:
        st.markdown("**Upload builder coverage screenshot**")
        method = "image_extraction"
        uploaded = st.file_uploader("Upload map screenshot (png/jpg)", type=["png", "jpg", "jpeg"])
        target_hex = st.text_input("Highlight color (hex)", value="#ff0000")
        tol = st.slider("Color tolerance", 5, 60, 25, step=1)
        min_area = st.slider("Minimum component area (px)", 100, 5000, 600, step=100)
        polygons = []
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(image_bgr, channels="BGR", caption="Uploaded map")
            target_hex = target_hex.lstrip("#")
            target_rgb = tuple(int(target_hex[i:i+2], 16) for i in (0, 2, 4))
            polygons = segment_region(image_bgr, target_rgb, tolerance=tol, min_area=min_area)
            st.caption(f"Detected {len(polygons)} region component(s).")

            st.markdown("**Alignment control points**")
            points_df = st.session_state.get("bre_points", pd.DataFrame(columns=["pixel_x", "pixel_y", "lat", "lon"]))
            points_df = st.data_editor(points_df, num_rows="dynamic", use_container_width=True, key="bre_points_editor")
            st.session_state["bre_points"] = points_df

            center = STATE_CENTER.get(state, (-25.0, 133.0))
            m = folium.Map(location=center, zoom_start=5, tiles="cartodbpositron")
            map_click = st_folium(m, height=350, width=None, returned_objects=["last_clicked"])
            if map_click and map_click.get("last_clicked"):
                st.caption(f"Last map click: {map_click['last_clicked']}")

            if len(points_df) >= 3 and polygons:
                src_pts = points_df[["pixel_x", "pixel_y"]].astype(float).values.tolist()
                dst_pts = points_df[["lon", "lat"]].astype(float).values.tolist()
                params = compute_affine(src_pts, dst_pts)
                warped = [apply_affine(p, params) for p in polygons if params is not None]
                union = union_polygons(warped)
                rms = alignment_rms(src_pts, dst_pts, params)
                rms_note = f"{rms:.4f}" if rms is not None else "n/a"
                notes = f"image_extraction; rms={rms_note}; components={len(polygons)}; tol={tol}"
                confidence = 0.80
                if image_bgr is not None and min(image_bgr.shape[:2]) < 900:
                    confidence -= 0.10
                if rms and rms > 0.01:
                    confidence -= 0.10
                if len(polygons) > 5:
                    confidence -= 0.05
                confidence = max(0.0, min(1.0, confidence))

                candidate_idx = query_candidates(index, union)
                matches = []
                for idx in candidate_idx:
                    suburb_geom = gdf.iloc[idx]["geometry"]
                    suburb_name = gdf.iloc[idx]["suburb_name"]
                    if centroid_in(union, suburb_geom):
                        matches.append(suburb_name)
                results_df = _build_results(builder_name, state, region_name, sorted(set(matches)), method, confidence, notes, suburb_map)
                st.session_state["bre_results"] = results_df
                st.markdown("**Preview overlay**")
                preview = folium.Map(location=center, zoom_start=6, tiles="cartodbpositron")
                folium.GeoJson(union.__geo_interface__, style_function=lambda x: {"fillColor": "#6366f1", "color": "#4f46e5", "weight": 2, "fillOpacity": 0.35}).add_to(preview)
                st_folium(preview, height=350, width=None)
            else:
                st.caption("Add at least 3 control points to enable alignment.")

    results_df = st.session_state.get("bre_results", pd.DataFrame())
    if results_df is None:
        results_df = pd.DataFrame()

    st.markdown("**Review & export**")
    manual_row = st.text_input("Add manual suburb", key="bre_manual_suburb")
    manual_pc = st.text_input("Add manual postcode", key="bre_manual_pc")
    if st.button("Add manual row"):
        if manual_row and manual_pc:
            manual_df = pd.DataFrame([{
                "builder_name": builder_name,
                "state": state,
                "region_name": region_name or "",
                "suburb": _normalize_suburb_name(manual_row),
                "postcode": str(manual_pc).zfill(4),
                "method": "manual_override",
                "confidence": 1.0,
                "notes": "manual_override",
                "created_at": datetime.utcnow().isoformat()
            }])
            results_df = pd.concat([results_df, manual_df], ignore_index=True).drop_duplicates(subset=["state", "suburb", "postcode"])
            results_df = results_df.sort_values(["postcode", "suburb"])
            st.session_state["bre_results"] = results_df

    if results_df.empty:
        st.caption("No results yet.")
        return

    kpis = [
        ("#Suburbs", str(results_df["suburb"].nunique())),
        ("#Postcodes", str(results_df["postcode"].nunique())),
        ("Method", results_df["method"].iloc[0]),
        ("Avg Confidence", f"{results_df['confidence'].mean():.0%}")
    ]
    _kpi_row(kpis)

    edited = st.data_editor(results_df, num_rows="dynamic", use_container_width=True)
    st.session_state["bre_results"] = edited

    if not builder_name or not state or edited.empty:
        st.caption("Set builder name and state to enable export.")
        return

    today = datetime.utcnow().strftime("%Y%m%d")
    filename_base = f"builder_coverage_{builder_name}_{state}_{today}".replace(" ", "_")
    st.download_button("Download CSV", export_csv(edited), file_name=f"{filename_base}.csv")
    st.download_button("Download JSON", export_json(edited), file_name=f"{filename_base}.json")
