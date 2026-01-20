"""
Postcode Opportunity Insights
Understand conversion rates and campaign density by postcode/suburb.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import sys
from pathlib import Path

root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data_loader import load_events
from src.normalization import normalize_events

st.set_page_config(page_title="Postcode Opportunity Insights", page_icon="📍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, .stDeployButton { display: none; }

.page-header { border-bottom: 1px solid #e5e7eb; padding-bottom: 1rem; margin-bottom: 1.5rem; }
.page-title { font-size: 1.6rem; font-weight: 700; color: #111827; margin: 0; }
.page-subtitle { color: #6b7280; font-size: 0.9rem; margin-top: 0.25rem; }

.section { margin-bottom: 2rem; }
.section-header { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.75rem; }
.section-num { background: linear-gradient(135deg, #111827, #374151); color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 600; }
.section-title { font-size: 1rem; font-weight: 600; color: #111827; }

.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.75rem; margin-bottom: 1rem; }
.kpi { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 0.9rem; box-shadow: 0 1px 0 rgba(17, 24, 39, 0.04); }
.kpi-label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280; margin-bottom: 0.2rem; }
.kpi-value { font-size: 1.25rem; font-weight: 700; color: #111827; }

.insight { background: #eff6ff; border-left: 3px solid #3b82f6; padding: 0.75rem 1rem; margin: 0.75rem 0; border-radius: 0 8px 8px 0; }
.insight-text { color: #1e40af; font-size: 0.85rem; line-height: 1.4; }

.explainer { background: #f8fafc; border: 1px solid #e5e7eb; padding: 0.75rem 0.9rem; border-radius: 10px; margin: 0.6rem 0 1rem 0; }
.explainer-title { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280; margin-bottom: 0.35rem; font-weight: 600; }
.explainer-text { color: #374151; font-size: 0.85rem; line-height: 1.45; }
.section-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 1rem; margin: 0.5rem 0 1.25rem 0; box-shadow: 0 10px 24px -20px rgba(17, 24, 39, 0.45); }
.section-card .section-header { margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data():
    if 'events_file' not in st.session_state:
        return None
    events = load_events(st.session_state['events_file'])
    return normalize_events(events) if events is not None else None


def _find_col(columns, candidates):
    cols = {c.lower(): c for c in columns}
    for c in candidates:
        if c in columns:
            return c
        if c.lower() in cols:
            return cols[c.lower()]
    return None


@st.cache_data(show_spinner=False)
def load_postcode_geo():
    geo_path = Path("data/au-postcodes.geojson")
    if not geo_path.exists():
        return None
    with geo_path.open() as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_postcode_meta():
    meta_path = Path("data/PostcodeData-final.txt")
    if not meta_path.exists():
        return None
    df = pd.read_csv(meta_path)
    df["Postcode"] = df["Postcode"].astype(str).str.zfill(4)
    df["Suburb"] = df["Suburb"].astype(str).str.strip().str.upper()
    df["State"] = df["State"].astype(str).str.strip().str.upper()
    return df[["Postcode", "Suburb", "State", "Lat", "Lng"]]


def main():
    events = load_data()
    if events is None:
        st.warning("⚠️ Please upload Events data on the Home page.")
        st.page_link("app.py", label="← Go to Home", icon="🏠")
        return

    postcode_col = _find_col(events.columns, ["Postcode"])
    suburb_col = _find_col(events.columns, ["Suburb"])
    if not postcode_col or not suburb_col:
        st.error("Missing required columns: Postcode and Suburb.")
        return

    geojson_data = load_postcode_geo()
    postcode_meta = load_postcode_meta()

    # Sidebar filters
    with st.sidebar:
        st.markdown("### Filters")
        dates = pd.to_datetime(events['lead_date'], errors='coerce').dropna()
        min_d, max_d = dates.min().date(), dates.max().date()
        date_range = st.date_input("Date Range", value=(min_d, max_d))
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2 and all(date_range):
            start_d, end_d = date_range[0], date_range[1]
        else:
            start_d, end_d = min_d, max_d
        min_leads = st.slider("Min leads per postcode", 1, 100, 10, step=1)
        zone_method = st.radio(
            "Zone thresholds",
            ["Median-based", "Target-based"],
            horizontal=False
        )
        target_conv = st.slider("Target conversion rate", 0.01, 0.5, 0.15, step=0.01)
        if postcode_meta is not None and not postcode_meta.empty:
            state_options = sorted(postcode_meta["State"].unique().tolist())
            state_filter = st.multiselect("State/Region", state_options, default=state_options)
        else:
            state_filter = []
        builder_scope = "Map only"
        builder_filter = []
        if "Dest_BuilderRegionKey" in events.columns:
            tmp = events.copy()
            tmp = tmp[(tmp['lead_date'] >= pd.Timestamp(start_d)) & (tmp['lead_date'] <= pd.Timestamp(end_d))]
            tmp[postcode_col] = tmp[postcode_col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
            tmp[postcode_col] = tmp[postcode_col].str.replace(r"\.0$", "", regex=True).str.zfill(4)
            if state_filter and postcode_meta is not None and not postcode_meta.empty:
                state_map = postcode_meta[["Postcode", "State"]].drop_duplicates()
                tmp = tmp.merge(state_map, left_on=postcode_col, right_on="Postcode", how="left")
                tmp = tmp[tmp["State"].isin(state_filter)]
            builder_options = sorted(tmp["Dest_BuilderRegionKey"].dropna().unique().tolist())
            builder_filter = st.multiselect(
                "Builders (marketing regions)",
                builder_options,
                key="builder_filter"
            )
            builder_scope = st.radio(
                "Apply builder filter to",
                ["Map only", "Whole page"],
                horizontal=True
            )

    df = events.copy()
    df = df[(df['lead_date'] >= pd.Timestamp(start_d)) & (df['lead_date'] <= pd.Timestamp(end_d))]

    df[postcode_col] = df[postcode_col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    df[postcode_col] = df[postcode_col].str.replace(r"\.0$", "", regex=True).str.zfill(4)
    df[suburb_col] = df[suburb_col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    df[suburb_col] = df[suburb_col].str.upper()
    df = df.dropna(subset=[postcode_col, suburb_col])
    if builder_filter and builder_scope == "Whole page" and "Dest_BuilderRegionKey" in df.columns:
        df = df[df["Dest_BuilderRegionKey"].isin(builder_filter)]

    if df.empty:
        st.warning("No events found for the selected filters.")
        return

    campaign_col = _find_col(df.columns, ["utm_campaign", "utm_key", "ad_key"])
    spend_col = _find_col(df.columns, ["MediaCost_referral_event", "MediaCost_builder_touch", "MediaCost_origin_lead"])

    # KPI header
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">📍 Postcode Opportunity Insights</h1>
        <p class="page-subtitle">Conversion efficiency by postcode + suburb and campaign density opportunities.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="explainer">
        <div class="explainer-title">How to use this page</div>
        <div class="explainer-text">
            Use the map to find strong and weak areas, then open the Forecast tab to estimate how many leads a region
            can deliver for your planned spend. Use Optimization to decide where to scale, fix, test, or reduce spend.
            This page is designed for daily budgeting and creative targeting decisions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    lead_id_col = "LeadId" if "LeadId" in df.columns else None
    is_referral = df["is_referral"].fillna(False).astype(bool) if "is_referral" in df.columns else pd.Series(False, index=df.index)

    if "is_referral" in df.columns:
        lead_df = df[df["is_referral"].fillna(False).astype(bool) == False].copy()
        refs_df = df[df["is_referral"].fillna(False).astype(bool)].copy()
        group = lead_df.groupby([postcode_col, suburb_col], as_index=False).agg(
            Leads=(lead_id_col, "nunique") if lead_id_col else (postcode_col, "size"),
            Media_Spend=(spend_col, "sum") if spend_col else (postcode_col, "size"),
            Campaigns=(campaign_col, "nunique") if campaign_col else (postcode_col, "size")
        )
        if lead_id_col:
            qualified = (
                refs_df.groupby([postcode_col, suburb_col])[lead_id_col]
                .nunique()
                .reset_index(name="Qualified_Leads")
            )
            referrals = (
                refs_df.groupby([postcode_col, suburb_col])
                .size()
                .reset_index(name="Referrals")
            )
        else:
            qualified = (
                refs_df.groupby([postcode_col, suburb_col])
                .size()
                .reset_index(name="Qualified_Leads")
            )
            referrals = qualified.rename(columns={"Qualified_Leads": "Referrals"})
        group = group.merge(qualified, on=[postcode_col, suburb_col], how="left")
        group = group.merge(referrals, on=[postcode_col, suburb_col], how="left")
    else:
        group = df.groupby([postcode_col, suburb_col], as_index=False).agg(
            Leads=(lead_id_col, "nunique") if lead_id_col else (postcode_col, "size"),
            Media_Spend=(spend_col, "sum") if spend_col else (postcode_col, "size"),
            Campaigns=(campaign_col, "nunique") if campaign_col else (postcode_col, "size")
        )
        group["Qualified_Leads"] = 0
        group["Referrals"] = 0
    group["Qualified_Leads"] = group["Qualified_Leads"].fillna(0)
    group["Referrals"] = group["Referrals"].fillna(0)
    if not spend_col:
        group["Media_Spend"] = 0
    if not campaign_col:
        group["Campaigns"] = 0

    group = group[group["Leads"] >= min_leads].copy()
    if postcode_meta is not None and not postcode_meta.empty:
        group = group.merge(
            postcode_meta,
            left_on=[postcode_col, suburb_col],
            right_on=["Postcode", "Suburb"],
            how="left"
        )
        if state_filter:
            group = group[group["State"].isin(state_filter)]
    group["Conversion_Rate"] = np.where(group["Leads"] > 0, group["Qualified_Leads"] / group["Leads"], 0)
    denom = group["Leads"] + group["Qualified_Leads"]
    group["CPR"] = np.where(denom > 0, group["Media_Spend"] / denom, np.nan)
    if campaign_col:
        group["Opportunity_Score"] = (1 - group["Conversion_Rate"]) * group["Leads"] * np.log1p(group["Campaigns"])
    else:
        group["Opportunity_Score"] = (1 - group["Conversion_Rate"]) * group["Leads"]

    avg_conv = group["Conversion_Rate"].mean() if not group.empty else 0
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi"><div class="kpi-label">Postcodes</div><div class="kpi-value">{group[postcode_col].nunique():,}</div></div>
        <div class="kpi"><div class="kpi-label">Avg Conversion</div><div class="kpi-value">{avg_conv:.0%}</div></div>
        <div class="kpi"><div class="kpi-label">Campaigns Tracked</div><div class="kpi-value">{group["Campaigns"].sum():,}</div></div>
        <div class="kpi"><div class="kpi-label">Total Leads</div><div class="kpi-value">{group["Leads"].sum():,}</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="explainer">
        <div class="explainer-title">KPI definitions (plain language)</div>
        <div class="explainer-text">
            <b>Avg conversion</b> is qualified leads ÷ total leads, so it shows how often a lead becomes qualified.
            <b>Campaigns tracked</b> is how many campaigns touched these regions, which helps spot crowding.
            <b>Total leads</b> and <b>postcodes</b> show the scale of your market coverage.
        </div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Metric glossary", expanded=False):
        st.markdown("""
        - **Conversion Rate**: Qualified leads ÷ total leads. Higher means more leads become qualified.
        - **Opportunity Score**: Higher when leads are high and conversion is low (plus campaign density). Targets fast improvement areas.
        - **CPL (Cost per Lead)**: Spend ÷ Leads. Lower means cheaper lead delivery.
        - **CPR (Cost per Referral)**: Spend ÷ (Unique Leads + Unique Referrals). Lower means more efficient coverage.
        - **Capacity (Spend)**: Planned spend ÷ Forecast CPL. How many leads spend can buy.
        - **Capacity (Pace)**: Recent delivery speed scaled to the forecast window. Avoids over‑allocating.
        - **Recommended Cap**: The smaller of spend capacity and pace capacity, to reduce overspend risk.
        """)

    tabs = st.tabs([
        "1) Map",
        "2) Opportunities",
        "3) Forecast",
        "4) Benchmarks",
        "5) Optimization",
        "6) Overlap",
        "7) Creative"
    ])

    with tabs[0]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">
                <span class="section-num">1</span>
                <span class="section-title">Australia Postcode Opportunity Map</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="explainer">
            <div class="explainer-title">What you are seeing</div>
            <div class="explainer-text">
                Performance view colors postcodes by conversion, leads, or opportunity score. Marketing Regions groups
                postcodes by the builder that services the most referrals there. Use this to align spend and ownership.
            </div>
        </div>
        """, unsafe_allow_html=True)

        map_mode = st.radio(
            "Map view",
            ["Performance", "Marketing Regions"],
            horizontal=True,
            label_visibility="collapsed",
            key="postcode_map_mode"
        )
        map_metric = st.radio(
            "Color by",
            ["Conversion Rate", "Opportunity Score", "Leads", "Referrals"],
            horizontal=True,
            label_visibility="collapsed",
            key="postcode_map_metric"
        )
        metric_map = {
            "Conversion Rate": "Conversion_Rate",
            "Opportunity Score": "Opportunity_Score",
            "Leads": "Leads",
            "Referrals": "Referrals"
        }
        metric_col = metric_map[map_metric]

        if geojson_data and not group.empty:
            active_postcodes = set(group["Postcode"].dropna().astype(str).str.zfill(4).tolist())
            if active_postcodes:
                geojson_filtered = {
                    "type": "FeatureCollection",
                    "features": [
                        f for f in geojson_data.get("features", [])
                        if f.get("properties", {}).get("POA_CODE") in active_postcodes
                    ]
                }
            else:
                geojson_filtered = geojson_data
            center = {"lat": -25.5, "lon": 134.0}
            zoom = 4.2
            postcode_rollup = (
                group.groupby("Postcode", as_index=False)
                .agg(
                    Leads=("Leads", "sum"),
                    Referrals=("Referrals", "sum"),
                    Media_Spend=("Media_Spend", "sum"),
                    Campaigns=("Campaigns", "sum"),
                    Conversion_Rate=("Conversion_Rate", "mean"),
                    Opportunity_Score=("Opportunity_Score", "sum")
                )
            )
            postcode_rollup["Postcode"] = (
                postcode_rollup["Postcode"]
                .astype(str)
                .str.strip()
                .replace({"nan": np.nan})
                .str.zfill(4)
            )
            postcode_rollup = postcode_rollup.dropna(subset=["Postcode"])
            postcode_rollup["CPR"] = np.where(
                postcode_rollup["Referrals"] > 0,
                postcode_rollup["Media_Spend"] / postcode_rollup["Referrals"],
                np.nan
            )

            region_view = "Dominant"
            if map_mode == "Marketing Regions":
                region_view = st.radio(
                    "Region coverage view",
                    ["Dominant", "Complete (overlap)"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="postcode_region_view"
                )

            if map_mode == "Marketing Regions" and "Dest_BuilderRegionKey" in df.columns:
                mask_referral = df["is_referral"].fillna(False).astype(bool) if "is_referral" in df.columns else pd.Series(False, index=df.index)
                mask_cross_payer = (
                    df["MediaPayer_BuilderRegionKey"].notna() &
                    df["Dest_BuilderRegionKey"].notna() &
                    (df["MediaPayer_BuilderRegionKey"] != df["Dest_BuilderRegionKey"])
                )
                referrals_df = df[mask_referral | mask_cross_payer].copy()
                builder_flows = pd.DataFrame()
                builder_filter_map = builder_filter if builder_filter else []
                if builder_filter_map:
                    referrals_df = referrals_df[referrals_df["Dest_BuilderRegionKey"].isin(builder_filter_map)]
                if not referrals_df.empty:
                    referrals_df["Postcode"] = referrals_df[postcode_col].astype(str).str.zfill(4)
                    builder_flows = (
                        referrals_df.groupby(["Postcode", "Dest_BuilderRegionKey"], as_index=False)
                        .agg(Referrals=("LeadId", "nunique") if "LeadId" in referrals_df.columns else ("lead_date", "size"))
                    )
                    if builder_filter_map:
                        builder_flows["Builder"] = builder_flows["Dest_BuilderRegionKey"]
                    else:
                        totals = builder_flows.groupby("Dest_BuilderRegionKey", as_index=False)["Referrals"].sum()
                        top_builders = totals.sort_values("Referrals", ascending=False).head(12)["Dest_BuilderRegionKey"].tolist()
                        builder_flows["Builder"] = np.where(
                            builder_flows["Dest_BuilderRegionKey"].isin(top_builders),
                            builder_flows["Dest_BuilderRegionKey"],
                            "Other"
                        )
                    overlap = (
                        builder_flows.groupby("Postcode", as_index=False)["Builder"]
                        .nunique()
                        .rename(columns={"Builder": "Builder_Count"})
                    )
                    primary = (
                        builder_flows.groupby(["Postcode", "Builder"], as_index=False)["Referrals"]
                        .sum()
                        .sort_values(["Postcode", "Referrals"], ascending=[True, False])
                    )
                    primary = primary.drop_duplicates("Postcode")
                    primary = primary.rename(columns={"Builder": "Primary Builder"})
                    map_df = postcode_rollup.merge(
                        primary[["Postcode", "Primary Builder", "Referrals"]],
                        on="Postcode",
                        how="left",
                        suffixes=("_metric", "_primary")
                    )
                    map_df = map_df.merge(overlap, on="Postcode", how="left")
                    if builder_filter_map:
                        map_df = map_df[map_df["Builder_Count"].fillna(0) > 0]

                    builder_summary = (
                        builder_flows.groupby("Dest_BuilderRegionKey", as_index=False)
                        .agg(
                            Referrals=("Referrals", "sum"),
                            Postcodes=("Postcode", "nunique")
                        )
                        .sort_values("Referrals", ascending=False)
                    )
                    if builder_filter_map:
                        builder_summary = builder_summary[builder_summary["Dest_BuilderRegionKey"].isin(builder_filter_map)]
                    if not builder_summary.empty:
                        st.markdown("**Builder coverage summary**")
                        st.dataframe(
                            builder_summary.rename(columns={"Dest_BuilderRegionKey": "Builder"}),
                            hide_index=True,
                            use_container_width=True
                        )
                else:
                    map_df = postcode_rollup.copy()
                    map_df["Primary Builder"] = "Unknown"
            else:
                map_df = postcode_rollup.copy()
                map_df["Primary Builder"] = None

            if "Referrals" not in map_df.columns and "Referrals_metric" in map_df.columns:
                map_df = map_df.rename(columns={"Referrals_metric": "Referrals"})
            if "Campaigns" not in map_df.columns and "Campaigns" in postcode_rollup.columns:
                map_df["Campaigns"] = postcode_rollup["Campaigns"]
            if "Leads" not in map_df.columns and "Leads" in postcode_rollup.columns:
                map_df["Leads"] = postcode_rollup["Leads"]

            if map_df.empty:
                st.caption("No postcode data available for the selected filters.")
            elif map_mode == "Performance":
                metric_series = map_df[metric_col].fillna(0)
                if metric_series.nunique() <= 1:
                    map_df["Metric Bin"] = "Mid"
                else:
                    try:
                        metric_bins = pd.qcut(
                            metric_series,
                            q=5,
                            labels=False,
                            duplicates="drop"
                        )
                        labels = ["Very Low", "Low", "Mid", "High", "Very High"]
                    except ValueError:
                        metric_bins = pd.qcut(
                            metric_series,
                            q=3,
                            labels=False,
                            duplicates="drop"
                        )
                        labels = ["Low", "Mid", "High"]
                    if metric_bins.isna().all():
                        map_df["Metric Bin"] = "Mid"
                    else:
                        metric_bins = metric_bins.astype("Int64")
                        max_bin = int(metric_bins.max()) if metric_bins.max() is not pd.NA else 0
                        safe_labels = labels[: max_bin + 1]
                        map_df["Metric Bin"] = metric_bins.map(
                            lambda x: safe_labels[int(x)] if pd.notna(x) and int(x) < len(safe_labels) else "Mid"
                        )
                hover_cols = {c: True for c in ["Leads", "Referrals", "Campaigns", "CPR"] if c in map_df.columns}
                fig = px.choropleth_mapbox(
                    map_df,
                    geojson=geojson_filtered,
                    locations="Postcode",
                    featureidkey="properties.POA_CODE",
                    color="Metric Bin",
                    color_discrete_map={
                        "Very Low": "#e0f2fe",
                        "Low": "#bae6fd",
                        "Mid": "#7dd3fc",
                        "High": "#38bdf8",
                        "Very High": "#0284c7"
                    },
                    hover_data=hover_cols,
                    mapbox_style="carto-positron",
                    zoom=zoom,
                    center=center,
                    opacity=0.6,
                    title="Postcode performance (select a state to focus)"
                )
            else:
                if "Primary Builder" not in map_df.columns or map_df["Primary Builder"].isna().all():
                    map_df["Primary Builder"] = "Unknown"
                map_df["Primary Builder"] = map_df["Primary Builder"].fillna("Unknown").astype(str)
                if region_view == "Complete (overlap)":
                    map_df["Builder_Count"] = map_df.get("Builder_Count", 0).fillna(0).astype(int)
                    builder_pool = []
                    if "Builder" in builder_flows.columns:
                        builder_pool = sorted(builder_flows["Builder"].dropna().unique().tolist())
                    if builder_filter_map:
                        builder_pool = builder_filter_map
                    selected_builders = builder_pool
                    if builder_pool:
                        selected_builders = st.multiselect(
                            "Display builder regions",
                            builder_pool,
                            default=builder_pool
                        )
                    if not selected_builders:
                        selected_builders = builder_pool
                    builder_region = builder_flows[builder_flows["Builder"].isin(selected_builders)].copy()
                    builder_region = builder_region.rename(columns={"Referrals": "Builder Referrals"})
                    map_df = map_df.merge(
                        builder_region[["Postcode", "Builder Referrals", "Builder"]],
                        on="Postcode",
                        how="left"
                    )
                    map_df = map_df[map_df["Builder Referrals"].fillna(0) > 0]
                    primary_selected = (
                        builder_region.groupby(["Postcode", "Builder"], as_index=False)["Builder Referrals"]
                        .sum()
                        .sort_values(["Postcode", "Builder Referrals"], ascending=[True, False])
                        .drop_duplicates("Postcode")
                        .rename(columns={"Builder": "Primary Builder"})
                    )
                    map_df = map_df.drop(columns=["Primary Builder"], errors="ignore").merge(
                        primary_selected[["Postcode", "Primary Builder"]],
                        on="Postcode",
                        how="left"
                    )
                    overlap_selected = (
                        builder_region.groupby("Postcode", as_index=False)["Builder"]
                        .nunique()
                        .rename(columns={"Builder": "Builder_Count"})
                    )
                    map_df = map_df.drop(columns=["Builder_Count"], errors="ignore").merge(
                        overlap_selected,
                        on="Postcode",
                        how="left"
                    )
                    map_df["Primary Builder"] = map_df["Primary Builder"].fillna("Unknown").astype(str)

                    def overlap_bucket(val):
                        if val <= 1:
                            return "1 builder"
                        if val <= 3:
                            return "2-3 builders"
                        if val <= 5:
                            return "4-5 builders"
                        return "6+ builders"

                    map_df["Overlap Bin"] = map_df["Builder_Count"].fillna(0).astype(int).map(overlap_bucket)

                    c1, c2 = st.columns(2)
                    hover_cols = {c: True for c in ["Leads", "Referrals", "Campaigns", "Builder Referrals", "Builder_Count"] if c in map_df.columns}
                    with c1:
                        fig_builders = px.choropleth_mapbox(
                            map_df,
                            geojson=geojson_filtered,
                            locations="Postcode",
                            featureidkey="properties.POA_CODE",
                            color="Primary Builder",
                            hover_data=hover_cols,
                            mapbox_style="carto-positron",
                            zoom=zoom,
                            center=center,
                            opacity=0.6,
                            title="Builder regions (selected builders)"
                        )
                        fig_builders.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                        st.plotly_chart(
                            fig_builders,
                            use_container_width=True,
                            config={"displayModeBar": True, "scrollZoom": True}
                        )
                    with c2:
                        fig_overlap = px.choropleth_mapbox(
                            map_df,
                            geojson=geojson_filtered,
                            locations="Postcode",
                            featureidkey="properties.POA_CODE",
                            color="Overlap Bin",
                            color_discrete_map={
                                "1 builder": "#dbeafe",
                                "2-3 builders": "#93c5fd",
                                "4-5 builders": "#60a5fa",
                                "6+ builders": "#2563eb"
                            },
                            hover_data=hover_cols,
                            mapbox_style="carto-positron",
                            zoom=zoom,
                            center=center,
                            opacity=0.6,
                            title="Overlap intensity (shared postcodes)"
                        )
                        fig_overlap.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                        st.plotly_chart(
                            fig_overlap,
                            use_container_width=True,
                            config={"displayModeBar": True, "scrollZoom": True}
                        )
                    overlap_hotspots = map_df[map_df["Builder_Count"] >= 2].copy()
                    if not overlap_hotspots.empty:
                        st.markdown("**Overlap hotspots (multiple builders in same postcode)**")
                        st.dataframe(
                            overlap_hotspots[["Postcode", "Builder_Count", "Leads", "Referrals"]].head(25),
                            hide_index=True,
                            use_container_width=True
                        )
                else:
                    hover_cols = {c: True for c in ["Leads", "Referrals", "Campaigns"] if c in map_df.columns}
                    fig = px.choropleth_mapbox(
                        map_df,
                        geojson=geojson_filtered,
                        locations="Postcode",
                        featureidkey="properties.POA_CODE",
                        color="Primary Builder",
                        hover_data=hover_cols,
                        mapbox_style="carto-positron",
                        zoom=zoom,
                        center=center,
                        opacity=0.6,
                        title="Marketing regions (dominant referral-serving builder)"
                    )
            if not map_df.empty and "fig" in locals():
                fig.update_layout(
                    height=560,
                    margin=dict(l=0, r=0, t=40, b=0),
                    uirevision="postcode-map"
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": True, "scrollZoom": True}
                )
        else:
            st.caption("Postcode geojson or joined metrics unavailable for mapping.")

    with tabs[1]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">
                <span class="section-num">2</span>
                <span class="section-title">Opportunity Areas</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="explainer">
            <div class="explainer-title">Opportunity score explained</div>
            <div class="explainer-text">
                Opportunity Score increases when a postcode has many leads but low conversion, and when many campaigns
                are already active. This flags areas where creative or targeting improvements can unlock fast gains.
            </div>
        </div>
        """, unsafe_allow_html=True)

        top_opps = group.sort_values("Opportunity_Score", ascending=False).head(15)
        opp_chart = top_opps.rename(columns={
            postcode_col: "Postcode",
            suburb_col: "Suburb"
        })
        opp_chart["Label"] = opp_chart["Postcode"] + " • " + opp_chart["Suburb"]
        fig2 = px.bar(
            opp_chart,
            x="Opportunity_Score",
            y="Label",
            orientation="h",
            color="Campaigns",
            title="Top opportunity postcodes (low conversion + high campaign density)"
        )
        fig2.update_layout(height=360, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        display = group.rename(columns={
            postcode_col: "Postcode",
            suburb_col: "Suburb",
            "Conversion_Rate": "Conversion Rate",
            "Opportunity_Score": "Opportunity Score",
            "Media_Spend": "Ad Spend"
        }).sort_values("Opportunity Score", ascending=False)
        st.dataframe(
            display[["Postcode", "Suburb", "Leads", "Qualified_Leads", "Referrals", "Conversion Rate", "Campaigns", "Ad Spend", "CPR", "Opportunity Score"]]
            .head(50),
            hide_index=True,
            use_container_width=True
        )

        if not group.empty:
            st.markdown("**Suburbs in selected regions**")
            suburb_list = (
                group[["Postcode", "Suburb", "Leads", "Qualified_Leads", "Referrals", "Conversion_Rate", "Campaigns"]]
                .sort_values(["Postcode", "Suburb"])
                .head(200)
            )
            st.dataframe(suburb_list, hide_index=True, use_container_width=True)

    with tabs[2]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">
                <span class="section-num">3</span>
                <span class="section-title">Region Forecast & Recommendations</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="explainer">
            <div class="explainer-title">Key metrics explained</div>
            <div class="explainer-text">
                <b>CPL</b> is spend ÷ leads. <b>Forecast CPL</b> is the recent average CPL. <b>Capacity (Spend)</b> is
                planned spend ÷ forecast CPL. <b>Capacity (Pace)</b> is recent lead pace scaled to your forecast window.
                We use the smaller of these as the recommended capacity to avoid over-spending.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="explainer">
            <div class="explainer-title">Levers you control</div>
            <div class="explainer-text">
                Use <b>Forecast horizon</b> to set how far ahead you are planning. <b>Lookback period</b> controls how
                many recent periods are used to estimate CPL. <b>Planned spend</b> and <b>Target leads</b> let you plan
                either budget-first or lead-first.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if group.empty:
            st.caption("Not enough data to build a regional forecast.")
        else:
            region_level = st.radio(
                "Region level",
                ["Postcode", "Suburb"],
                horizontal=True
            )
        if region_level == "Postcode":
            region_options = sorted(group["Postcode"].dropna().astype(str).unique().tolist())
            region_value = st.selectbox("Select postcode", region_options)
            region_df = df[df[postcode_col].astype(str).str.zfill(4) == str(region_value).zfill(4)].copy()
        else:
            region_map = (
                group[["Postcode", "Suburb"]]
                .dropna()
                .drop_duplicates()
            )
            region_map["Label"] = region_map["Suburb"] + " (" + region_map["Postcode"] + ")"
            region_options = region_map["Label"].sort_values().tolist()
            region_value = st.selectbox("Select suburb", region_options)
            selected = region_map[region_map["Label"] == region_value]
            if not selected.empty:
                sel_postcode = selected["Postcode"].iloc[0]
                sel_suburb = selected["Suburb"].iloc[0]
                region_df = df[
                    (df[postcode_col].astype(str).str.zfill(4) == str(sel_postcode).zfill(4)) &
                    (df[suburb_col].str.upper() == str(sel_suburb).upper())
                ].copy()
            else:
                region_df = df.iloc[0:0].copy()

        budget_col = _find_col(df.columns, ["Budget"])
        finance_col = _find_col(df.columns, ["Finance Status"])
        timeframe_col = _find_col(df.columns, ["Timeframe"])
        land_col = _find_col(df.columns, ["Do you have land"])
        house_col = _find_col(df.columns, ["House type"])
        beds_col = _find_col(df.columns, ["IBN_Bedrooms"])

        st.markdown("**Targeting filters**")
        f_cols = st.columns(3)
        seg_df = region_df.copy()

        def apply_filter(col_name, label, col_idx):
            nonlocal seg_df
            if not col_name or col_name not in seg_df.columns:
                return
            options = seg_df[col_name].dropna().astype(str).unique().tolist()
            if not options:
                return
            with f_cols[col_idx]:
                selected = st.multiselect(label, sorted(options), default=sorted(options))
            if selected:
                seg_df = seg_df[seg_df[col_name].astype(str).isin(selected)]

        apply_filter(finance_col, "Finance Status", 0)
        apply_filter(timeframe_col, "Timeframe", 1)
        apply_filter(land_col, "Do you have land", 2)

        f_cols2 = st.columns(3)
        with f_cols2[0]:
            if house_col and house_col in seg_df.columns:
                options = seg_df[house_col].dropna().astype(str).unique().tolist()
                house_sel = st.multiselect("House type", sorted(options), default=sorted(options)) if options else []
                if house_sel:
                    seg_df = seg_df[seg_df[house_col].astype(str).isin(house_sel)]
        with f_cols2[1]:
            if beds_col and beds_col in seg_df.columns:
                options = seg_df[beds_col].dropna().astype(str).unique().tolist()
                bed_sel = st.multiselect("Bedrooms", sorted(options), default=sorted(options)) if options else []
                if bed_sel:
                    seg_df = seg_df[seg_df[beds_col].astype(str).isin(bed_sel)]
        with f_cols2[2]:
            if budget_col and budget_col in seg_df.columns:
                budget_vals = pd.to_numeric(seg_df[budget_col], errors="coerce").dropna()
                if not budget_vals.empty:
                    min_b, max_b = float(budget_vals.min()), float(budget_vals.max())
                    if min_b == max_b:
                        st.caption("Budget range: single value")
                    else:
                        b_range = st.slider("Budget range", min_b, max_b, (min_b, max_b))
                        seg_df = seg_df[
                            pd.to_numeric(seg_df[budget_col], errors="coerce").between(b_range[0], b_range[1])
                        ]

        if seg_df.empty:
            st.caption("No leads match the selected filters.")
        else:
            seg_df["lead_date"] = pd.to_datetime(seg_df["lead_date"], errors="coerce")
            seg_df = seg_df.dropna(subset=["lead_date"])
            if "is_referral" in seg_df.columns:
                seg_leads = seg_df[seg_df["is_referral"].fillna(False).astype(bool) == False]
            else:
                seg_leads = seg_df
            lead_count = seg_leads[lead_id_col].nunique() if lead_id_col else len(seg_leads)
            spend_total = seg_df[spend_col].sum() if spend_col else 0
            cpl = spend_total / lead_count if lead_count > 0 else 0

            controls_col, output_col = st.columns([1, 2])
            with controls_col:
                st.markdown("**Forecast inputs**")
                horizon_days = st.slider("Forecast horizon (days)", 7, 90, 30, step=1)
                planned_spend = st.number_input("Planned spend ($)", min_value=0.0, value=5000.0, step=500.0)
                target_leads = st.number_input("Target leads", min_value=0.0, value=100.0, step=10.0)
                boost_spend = st.number_input(
                    "Stretch spend ($)",
                    min_value=0.0,
                    value=max(5000.0, planned_spend * 1.5),
                    step=500.0
                )
                ts_freq = st.radio("Trend period", ["Weekly", "Monthly"], horizontal=True)
                period_freq = "W" if ts_freq == "Weekly" else "M"
                lookback_periods = st.slider("CPL lookback periods", 2, 12, 6, step=1)

            if "is_referral" in seg_df.columns:
                lead_ts = seg_df[seg_df["is_referral"].fillna(False).astype(bool) == False].copy()
            else:
                lead_ts = seg_df.copy()
            ts = (
                lead_ts.assign(period=lead_ts["lead_date"].dt.to_period(period_freq).dt.start_time)
                .groupby("period", as_index=False)
                .agg(
                    Leads=(lead_id_col, "nunique") if lead_id_col else ("lead_date", "size"),
                    Spend=(spend_col, "sum") if spend_col else ("lead_date", "size")
                )
                .sort_values("period")
            )
            ts["CPL"] = np.where(ts["Leads"] > 0, ts["Spend"] / ts["Leads"], np.nan)
            if "is_referral" in seg_df.columns:
                qualified_ts = (
                    seg_df[seg_df["is_referral"] == True]
                    .assign(period=seg_df["lead_date"].dt.to_period(period_freq).dt.start_time)
                    .groupby("period", as_index=False)
                    .agg(Qualified_Leads=(lead_id_col, "nunique") if lead_id_col else ("lead_date", "size"))
                )
                ts = ts.merge(qualified_ts, on="period", how="left")
            ts["Qualified_Leads"] = ts.get("Qualified_Leads", 0).fillna(0)
            denom_ts = ts["Leads"] + ts["Qualified_Leads"]
            ts["CPR"] = np.where(denom_ts > 0, ts["Spend"] / denom_ts, np.nan)
            lookback = min(lookback_periods, len(ts))
            cpl_forecast = ts["CPL"].tail(lookback).mean() if lookback > 0 else cpl
            cpl_forecast = cpl_forecast if cpl_forecast and cpl_forecast > 0 else cpl
            cpl_std = ts["CPL"].tail(lookback).std() if lookback > 1 else 0

            end_date = seg_df["lead_date"].max()
            recent_mask = seg_df["lead_date"] >= (end_date - pd.Timedelta(days=14))
            prev_mask = (seg_df["lead_date"] < (end_date - pd.Timedelta(days=14))) & (seg_df["lead_date"] >= (end_date - pd.Timedelta(days=28)))
            if "is_referral" in seg_df.columns:
                recent_leads_df = seg_df[recent_mask & (seg_df["is_referral"].fillna(False).astype(bool) == False)]
                prev_leads_df = seg_df[prev_mask & (seg_df["is_referral"].fillna(False).astype(bool) == False)]
            else:
                recent_leads_df = seg_df[recent_mask]
                prev_leads_df = seg_df[prev_mask]
            recent_leads = recent_leads_df[lead_id_col].nunique() if lead_id_col else len(recent_leads_df)
            prev_leads = prev_leads_df[lead_id_col].nunique() if lead_id_col else len(prev_leads_df)
            growth = (recent_leads - prev_leads) / prev_leads if prev_leads > 0 else 0
            growth = float(np.clip(growth, -0.5, 0.5))
            pace = recent_leads / 14 if recent_leads > 0 else 0
            recent_7 = seg_df["lead_date"] >= (end_date - pd.Timedelta(days=7))
            recent_28 = seg_df["lead_date"] >= (end_date - pd.Timedelta(days=28))
            if "is_referral" in seg_df.columns:
                lead_mask = seg_df["is_referral"].fillna(False).astype(bool) == False
                recent_7_df = seg_df[recent_7 & lead_mask]
                recent_28_df = seg_df[recent_28 & lead_mask]
            else:
                recent_7_df = seg_df[recent_7]
                recent_28_df = seg_df[recent_28]
            pace_7 = (recent_7_df[lead_id_col].nunique() if lead_id_col else len(recent_7_df)) / 7 if not recent_7_df.empty else 0
            pace_14 = pace
            pace_28 = (recent_28_df[lead_id_col].nunique() if lead_id_col else len(recent_28_df)) / 28 if not recent_28_df.empty else 0
            capacity_pace = pace * (1 + growth) * horizon_days
            capacity_spend = planned_spend / cpl_forecast if cpl_forecast > 0 else 0
            capacity = min(capacity_spend, capacity_pace) if capacity_pace > 0 else capacity_spend
            binding = "Pace-limited" if capacity_pace < capacity_spend else "Spend-limited"
            required_spend = target_leads * cpl_forecast if cpl_forecast > 0 else 0
            boost_capacity_spend = boost_spend / cpl_forecast if cpl_forecast > 0 else 0
            boost_capacity = min(boost_capacity_spend, capacity_pace) if capacity_pace > 0 else boost_capacity_spend

            with output_col:
                st.markdown(f"""
                <div class="kpi-row">
                    <div class="kpi"><div class="kpi-label">Region Leads</div><div class="kpi-value">{lead_count:,.0f}</div></div>
                    <div class="kpi"><div class="kpi-label">Current CPL</div><div class="kpi-value">${cpl:,.0f}</div></div>
                    <div class="kpi"><div class="kpi-label">Forecast CPL</div><div class="kpi-value">${cpl_forecast:,.0f}</div></div>
                    <div class="kpi"><div class="kpi-label">Capacity (Spend)</div><div class="kpi-value">{capacity_spend:,.0f} leads</div></div>
                    <div class="kpi"><div class="kpi-label">Capacity (Pace)</div><div class="kpi-value">{capacity_pace:,.0f} leads</div></div>
                    <div class="kpi"><div class="kpi-label">Recommended Cap</div><div class="kpi-value">{capacity:,.0f} leads</div></div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="explainer">
                    <div class="explainer-title">What these outputs mean</div>
                    <div class="explainer-text">
                        <b>Binding</b> shows what limits delivery right now: pace‑limited means the region isn’t producing
                        fast enough; spend‑limited means budget is the main constraint. <b>Recommended cap</b> is the
                        smaller of spend capacity and pace capacity, so you don’t over‑allocate into thin supply.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"Binding constraint: {binding}")
                st.markdown(f"**Spend to hit target:** ${required_spend:,.0f} for {target_leads:,.0f} leads")

                horizon_table = pd.DataFrame([
                    {"Horizon (days)": 7, "Pace Capacity (leads)": pace * (1 + growth) * 7},
                    {"Horizon (days)": 30, "Pace Capacity (leads)": pace * (1 + growth) * 30},
                    {"Horizon (days)": 60, "Pace Capacity (leads)": pace * (1 + growth) * 60},
                    {"Horizon (days)": horizon_days, "Pace Capacity (leads)": capacity_pace}
                ])
                st.markdown("**Horizon impact on capacity**")
                st.dataframe(horizon_table, hide_index=True, use_container_width=True)

                horizon_fig = go.Figure()
                horizon_fig.add_trace(go.Scatter(
                    x=horizon_table["Horizon (days)"],
                    y=horizon_table["Pace Capacity (leads)"],
                    mode="lines+markers",
                    name="Pace capacity"
                ))
                horizon_fig.update_layout(height=220, margin=dict(l=0, r=0, t=40, b=0), yaxis_title="Leads", title="Capacity grows with the forecast horizon")
                st.plotly_chart(horizon_fig, use_container_width=True, config={"displayModeBar": False})

                if not ts.empty:
                    low = max(cpl_forecast - (cpl_std or 0), 0)
                    high = cpl_forecast + (cpl_std or 0)
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(
                        x=ts["period"],
                        y=ts["CPL"],
                        mode="lines+markers",
                        name="Actual CPL"
                    ))
                    fig_ts.add_trace(go.Scatter(
                        x=ts["period"],
                        y=ts["CPR"],
                        mode="lines+markers",
                        name="Actual CPR",
                        line=dict(dash="dot")
                    ))
                    fig_ts.add_trace(go.Scatter(
                        x=ts["period"],
                        y=[cpl_forecast] * len(ts),
                        mode="lines",
                        name="Forecast CPL",
                        line=dict(dash="dash")
                    ))
                    fig_ts.add_trace(go.Scatter(
                        x=ts["period"],
                        y=[high] * len(ts),
                        mode="lines",
                        name="Range",
                        line=dict(width=0),
                        showlegend=False
                    ))
                    fig_ts.add_trace(go.Scatter(
                        x=ts["period"],
                        y=[low] * len(ts),
                        mode="lines",
                        fill="tonexty",
                        name="CPL range",
                        line=dict(width=0),
                        opacity=0.2
                    ))
                    fig_ts.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0), yaxis_title="CPL")
                    st.plotly_chart(fig_ts, use_container_width=True, config={"displayModeBar": False})

                pace_fig = go.Figure()
                pace_fig.add_trace(go.Bar(
                    x=["Last 7d", "Last 14d", "Last 28d"],
                    y=[pace_7, pace_14, pace_28],
                    marker_color=["#60a5fa", "#3b82f6", "#1d4ed8"],
                    name="Leads per day"
                ))
                pace_fig.update_layout(height=220, margin=dict(l=0, r=0, t=40, b=0), yaxis_title="Leads / day", title="Recent delivery pace")
                st.plotly_chart(pace_fig, use_container_width=True, config={"displayModeBar": False})

                scenario = pd.DataFrame([
                    {"Scenario": "Planned", "Spend": planned_spend, "Capacity (Spend)": capacity_spend, "Recommended Cap": capacity, "Binding": binding},
                    {"Scenario": "Stretch", "Spend": boost_spend, "Capacity (Spend)": boost_capacity_spend, "Recommended Cap": boost_capacity, "Binding": binding}
                ])
                st.markdown("**Scenario comparison**")
                st.dataframe(scenario, hide_index=True, use_container_width=True)

            if campaign_col:
                if "is_referral" in seg_df.columns:
                    lead_campaign_df = seg_df[seg_df["is_referral"].fillna(False).astype(bool) == False].copy()
                    ref_campaign_df = seg_df[seg_df["is_referral"].fillna(False).astype(bool)].copy()
                else:
                    lead_campaign_df = seg_df.copy()
                    ref_campaign_df = seg_df.iloc[0:0].copy()

                camp_leads = (
                    lead_campaign_df.groupby(campaign_col, as_index=False)
                    .agg(Leads=(lead_id_col, "nunique") if lead_id_col else ("lead_date", "size"))
                )
                if not ref_campaign_df.empty:
                    camp_refs = (
                        ref_campaign_df.groupby(campaign_col, as_index=False)
                        .agg(Referrals=(lead_id_col, "nunique") if lead_id_col else ("lead_date", "size"))
                    )
                else:
                    camp_refs = pd.DataFrame(columns=[campaign_col, "Referrals"])
                camp_spend = (
                    seg_df.groupby(campaign_col, as_index=False)
                    .agg(Spend=(spend_col, "sum") if spend_col else ("lead_date", "size"))
                )
                camp = camp_leads.merge(camp_refs, on=campaign_col, how="left").merge(camp_spend, on=campaign_col, how="left")
                camp["Referrals"] = camp["Referrals"].fillna(0)
                camp["CPL"] = np.where(camp["Leads"] > 0, camp["Spend"] / camp["Leads"], np.nan)
                camp_denom = camp["Leads"] + camp["Referrals"]
                camp["CPR"] = np.where(camp_denom > 0, camp["Spend"] / camp_denom, np.nan)
                camp = camp.sort_values(["Referrals", "Leads"], ascending=False).head(10)

                st.markdown("**Recommended campaigns for this region**")
                st.dataframe(
                    camp.rename(columns={campaign_col: "Campaign"}),
                    hide_index=True,
                    use_container_width=True
                )

                total_leads = camp["Leads"].sum()
                total_refs = camp["Referrals"].sum()
                top_share = camp.head(3)["Referrals"].sum() / total_refs if total_refs > 0 else 0
                st.markdown(f"""
                <div class="explainer">
                    <div class="explainer-title">Justified by campaigns</div>
                    <div class="explainer-text">
                        The top 3 campaigns account for <b>{top_share:.0%}</b> of referral activity in this region.
                        Forecast assumptions are grounded in these campaigns because they are driving the majority of qualified outcomes.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                comp_fields = [
                    (budget_col, "Budget"),
                    (finance_col, "Finance Status"),
                    (timeframe_col, "Timeframe"),
                    (land_col, "Do you have land"),
                    (house_col, "House type"),
                    (beds_col, "Bedrooms")
                ]
                if not camp.empty:
                    comp_rows = []
                    top_campaigns = camp[campaign_col].head(5).tolist()
                    for c in top_campaigns:
                        c_df = campaign_df[campaign_df[campaign_col] == c]
                        row = {"Campaign": c}
                        for col, label in comp_fields:
                            if col and col in c_df.columns:
                                top_val = c_df[col].dropna().astype(str).value_counts().head(1)
                                if not top_val.empty:
                                    row[label] = f"{top_val.index[0]} ({top_val.iloc[0]})"
                        comp_rows.append(row)
                    if comp_rows:
                        st.markdown("**Campaign composition snapshot**")
                        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

    with tabs[3]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">
                <span class="section-num">4</span>
                <span class="section-title">Regional Benchmarks</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="explainer">
            <div class="explainer-title">How to read benchmarks</div>
            <div class="explainer-text">
                Benchmarks compare each state to the overall average. Underperforming areas are high volume but low
                conversion, making them the best candidates for creative or funnel fixes.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if group.empty:
            st.caption("Not enough data to build benchmarks.")
        else:
            if "State" in group.columns and group["State"].notna().any():
                state_benchmark = (
                    group.groupby("State", as_index=False)
                    .agg(
                        Leads=("Leads", "sum"),
                        Referrals=("Referrals", "sum"),
                        Avg_Conversion=("Conversion_Rate", "mean"),
                        Avg_CPR=("CPR", "mean")
                    )
                )
                st.markdown("**State benchmarks**")
                st.dataframe(
                    state_benchmark.rename(columns={
                        "Avg_Conversion": "Avg Conversion",
                        "Avg_CPR": "Avg CPR"
                    }),
                    hide_index=True,
                    use_container_width=True
                )

            conv_median = group["Conversion_Rate"].median() if group["Conversion_Rate"].notna().any() else 0
            lead_median = group["Leads"].median() if group["Leads"].notna().any() else 0
            benchmark_conv = group["Conversion_Rate"].mean() if group["Conversion_Rate"].notna().any() else 0
            benchmark_cpr = group["CPR"].mean() if group["CPR"].notna().any() else 0
            group["Conv_vs_Avg"] = group["Conversion_Rate"] - benchmark_conv
            group["CPR_vs_Avg"] = group["CPR"] - benchmark_cpr
            outliers = group[
                (group["Leads"] >= lead_median) &
                (group["Conversion_Rate"] < benchmark_conv * 0.8)
            ].sort_values("Opportunity_Score", ascending=False)
            st.markdown("**Underperforming high-volume postcodes**")
            st.dataframe(
                outliers[["Postcode", "Suburb", "Leads", "Conversion_Rate", "CPR", "Campaigns"]].head(20),
                hide_index=True,
                use_container_width=True
            )

    with tabs[4]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">
                <span class="section-num">5</span>
                <span class="section-title">Media Spend Optimization Plan</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="explainer">
            <div class="explainer-title">How to use this plan</div>
            <div class="explainer-text">
                Scale regions that are high volume and high conversion. Fix regions that are high volume but low
                conversion. Test small budgets in high conversion but low volume areas. Reduce spend in low volume,
                low conversion areas.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if group.empty:
            st.caption("Not enough data to build a regional optimization plan.")
        else:
            conv_median = group["Conversion_Rate"].median() if group["Conversion_Rate"].notna().any() else 0
            lead_median = group["Leads"].median() if group["Leads"].notna().any() else 0
            conv_threshold = target_conv if zone_method == "Target-based" else conv_median

            def zone_for_row(row):
                high_conv = row["Conversion_Rate"] >= conv_threshold
                high_vol = row["Leads"] >= lead_median
                if high_conv and high_vol:
                    return "Scale"
                if (not high_conv) and high_vol:
                    return "Fix"
                if high_conv and (not high_vol):
                    return "Test"
                return "Deprioritize"

            group["Zone"] = group.apply(zone_for_row, axis=1)
            action_map = {
                "Scale": "Increase budget 20-40%, prioritize best campaigns",
                "Fix": "Hold spend, localize creative and landing pages",
                "Test": "Run small tests, replicate top creatives",
                "Deprioritize": "Reduce spend, reallocate to Scale/Fix"
            }
            group["Recommended Action"] = group["Zone"].map(action_map)

            zone_summary = (
                group.groupby("Zone", as_index=False)
                .agg(
                    Postcodes=("Postcode", "nunique"),
                    Leads=("Leads", "sum"),
                    Referrals=("Referrals", "sum"),
                    Spend=("Media_Spend", "sum"),
                    Avg_Conversion=("Conversion_Rate", "mean"),
                    Avg_CPR=("CPR", "mean")
                )
            )

            def budget_shift(row):
                if row["Zone"] == "Scale":
                    return "+30%"
                if row["Zone"] == "Fix":
                    return "0% (optimize)"
                if row["Zone"] == "Test":
                    return "+10% (capped)"
                return "-25%"

            zone_summary["Suggested Budget Shift"] = zone_summary.apply(budget_shift, axis=1)
            zone_summary = zone_summary.sort_values("Postcodes", ascending=False)

            st.markdown("**Zone summary**")
            st.dataframe(
                zone_summary.rename(columns={
                    "Avg_Conversion": "Avg Conversion",
                    "Avg_CPR": "Avg CPR"
                }),
                hide_index=True,
                use_container_width=True
            )

            st.markdown("**Priority actions by postcode**")
            action_table = group.rename(columns={
                "Conversion_Rate": "Conversion Rate",
                "Media_Spend": "Ad Spend"
            }).sort_values(
                ["Zone", "Opportunity_Score"],
                ascending=[True, False]
            )
            st.dataframe(
                action_table[[
                    "Postcode", "Suburb", "Leads", "Referrals", "Conversion Rate",
                    "Campaigns", "Ad Spend", "CPR", "Zone", "Recommended Action"
                ]].head(50),
                hide_index=True,
                use_container_width=True
            )

    with tabs[5]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">
                <span class="section-num">6</span>
                <span class="section-title">Campaign Overlap Diagnostics</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="explainer">
            <div class="explainer-title">Why overlap matters</div>
            <div class="explainer-text">
                When too many campaigns target the same postcodes, results can dilute. Use this to consolidate budget
                into the few campaigns that are already converting well in those areas.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if group.empty or not campaign_col:
            st.caption("Campaign overlap requires campaign fields (utm_campaign/utm_key/ad_key).")
        else:
            crowd = group.copy()
            crowd["Campaigns_per_Lead"] = np.where(crowd["Leads"] > 0, crowd["Campaigns"] / crowd["Leads"], 0)
            crowded = crowd.sort_values("Campaigns", ascending=False).head(15)
            st.markdown("**Most crowded postcodes**")
            st.dataframe(
                crowded[["Postcode", "Suburb", "Leads", "Campaigns", "Campaigns_per_Lead", "Conversion_Rate"]],
                hide_index=True,
                use_container_width=True
            )

            if "LeadId" in df.columns:
                campaign_list = (
                    df.groupby([postcode_col, suburb_col, campaign_col], as_index=False)["LeadId"]
                    .nunique()
                    .rename(columns={"LeadId": "Leads"})
                )
            else:
                campaign_list = (
                    df.groupby([postcode_col, suburb_col, campaign_col], as_index=False)
                    .size()
                    .rename(columns={"size": "Leads"})
                )
            campaign_list = campaign_list.sort_values("Leads", ascending=False).head(50)
            campaign_list = campaign_list.rename(columns={
                postcode_col: "Postcode",
                suburb_col: "Suburb",
                campaign_col: "Campaign"
            })
            st.markdown("**Top campaigns in high-overlap areas**")
            st.dataframe(campaign_list, hide_index=True, use_container_width=True)

    with tabs[6]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">
                <span class="section-num">7</span>
                <span class="section-title">Creative Guidance</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="explainer">
            <div class="explainer-title">Commercial value</div>
            <div class="explainer-text">
                Mentioning suburbs or postcodes in your creative typically lifts relevance and conversion. Use the
                opportunity list to decide which areas to feature in ads and landing pages.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight">
        <div class="insight-text">
            <b>Recommendation:</b> Focus more advertising in the opportunity postcodes above and mention those areas directly in the creative.
            This typically lifts conversion where campaign density is already high but referral efficiency lags.
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    
