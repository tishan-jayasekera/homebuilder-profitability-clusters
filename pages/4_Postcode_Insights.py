"""
Postcode Opportunity Insights
Understand conversion rates and campaign density by postcode/suburb.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2 and all(date_range):
        start_d, end_d = date_range[0], date_range[1]
    else:
        start_d, end_d = min_d, max_d

    df = events.copy()
    df = df[(df['lead_date'] >= pd.Timestamp(start_d)) & (df['lead_date'] <= pd.Timestamp(end_d))]

    df[postcode_col] = df[postcode_col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    df[postcode_col] = df[postcode_col].str.replace(r"\.0$", "", regex=True).str.zfill(4)
    df[suburb_col] = df[suburb_col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    df[suburb_col] = df[suburb_col].str.upper()
    df = df.dropna(subset=[postcode_col, suburb_col])

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

    lead_id_col = "LeadId" if "LeadId" in df.columns else None
    is_referral = df["is_referral"].fillna(False).astype(bool) if "is_referral" in df.columns else pd.Series(False, index=df.index)

    group = df.groupby([postcode_col, suburb_col], as_index=False).agg(
        Leads=(lead_id_col, "nunique") if lead_id_col else (postcode_col, "size"),
        Referrals=("is_referral", "sum") if "is_referral" in df.columns else (postcode_col, "size"),
        Media_Spend=(spend_col, "sum") if spend_col else (postcode_col, "size"),
        Campaigns=(campaign_col, "nunique") if campaign_col else (postcode_col, "size")
    )
    if "is_referral" not in df.columns:
        group["Referrals"] = 0
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
    group["Conversion_Rate"] = np.where(group["Leads"] > 0, group["Referrals"] / group["Leads"], 0)
    group["CPR"] = np.where(group["Referrals"] > 0, group["Media_Spend"] / group["Referrals"], np.nan)
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

    # Section 1: Relationship view
    st.markdown("""
    <div class="section">
        <div class="section-header">
            <span class="section-num">1</span>
            <span class="section-title">Australia Postcode Opportunity Map</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
        if "Lat" in group.columns and "Lng" in group.columns and group["Lat"].notna().any():
            center = {"lat": float(group["Lat"].mean()), "lon": float(group["Lng"].mean())}
        else:
            center = {"lat": -25.5, "lon": 134.0}
        if state_filter:
            zoom = 6.5 if len(state_filter) == 1 else 5
        else:
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
        metric_series = postcode_rollup[metric_col].fillna(0)
        if metric_series.nunique() <= 1:
            postcode_rollup["Metric Bin"] = "Mid"
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
                postcode_rollup["Metric Bin"] = "Mid"
            else:
                metric_bins = metric_bins.astype("Int64")
                max_bin = int(metric_bins.max()) if metric_bins.max() is not pd.NA else 0
                safe_labels = labels[: max_bin + 1]
                postcode_rollup["Metric Bin"] = metric_bins.map(
                    lambda x: safe_labels[int(x)] if pd.notna(x) and int(x) < len(safe_labels) else "Mid"
                )
        fig = px.choropleth_mapbox(
            postcode_rollup,
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
            hover_data={
                "Leads": True,
                "Referrals": True,
                "Campaigns": True,
                "CPR": True
            },
            mapbox_style="carto-positron",
            zoom=zoom,
            center=center,
            opacity=0.6,
            title="Postcode performance (select a state to focus)"
        )
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

    # Section 2: Opportunity ranking
    st.markdown("""
    <div class="section">
        <div class="section-header">
            <span class="section-num">2</span>
            <span class="section-title">Opportunity Areas</span>
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
        display[["Postcode", "Suburb", "Leads", "Referrals", "Conversion Rate", "Campaigns", "Ad Spend", "CPR", "Opportunity Score"]]
        .head(50),
        hide_index=True,
        use_container_width=True
    )

    if not group.empty:
        st.markdown("**Suburbs in selected regions**")
        suburb_list = (
            group[["Postcode", "Suburb", "Leads", "Referrals", "Conversion_Rate", "Campaigns"]]
            .sort_values(["Postcode", "Suburb"])
            .head(200)
        )
        st.dataframe(suburb_list, hide_index=True, use_container_width=True)

    # Section 3: Regional benchmarks and outliers
    st.markdown("""
    <div class="section">
        <div class="section-header">
            <span class="section-num">3</span>
            <span class="section-title">Regional Benchmarks</span>
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

    # Section 4: Media spend optimization by region
    st.markdown("""
    <div class="section">
        <div class="section-header">
            <span class="section-num">4</span>
            <span class="section-title">Media Spend Optimization Plan</span>
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

    # Section 5: Campaign overlap diagnostics
    st.markdown("""
    <div class="section">
        <div class="section-header">
            <span class="section-num">5</span>
            <span class="section-title">Campaign Overlap Diagnostics</span>
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

    # Section 6: Creative guidance
    st.markdown("""
    <div class="section">
        <div class="section-header">
            <span class="section-num">6</span>
            <span class="section-title">Creative Guidance</span>
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
    
