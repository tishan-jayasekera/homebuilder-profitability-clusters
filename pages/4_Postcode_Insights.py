"""
Postcode Opportunity Insights
Understand conversion rates and campaign density by postcode/suburb.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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

    # Sidebar filters
    with st.sidebar:
        st.markdown("### Filters")
        dates = pd.to_datetime(events['lead_date'], errors='coerce').dropna()
        min_d, max_d = dates.min().date(), dates.max().date()
        date_range = st.date_input("Date Range", value=(min_d, max_d))
        min_leads = st.slider("Min leads per postcode", 1, 100, 10, step=1)

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2 and all(date_range):
        start_d, end_d = date_range[0], date_range[1]
    else:
        start_d, end_d = min_d, max_d

    df = events.copy()
    df = df[(df['lead_date'] >= pd.Timestamp(start_d)) & (df['lead_date'] <= pd.Timestamp(end_d))]

    df[postcode_col] = df[postcode_col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    df[postcode_col] = df[postcode_col].str.replace(r"\.0$", "", regex=True)
    df[suburb_col] = df[suburb_col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
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
            <span class="section-title">Conversion vs Campaign Density</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    scatter = group.rename(columns={
        postcode_col: "Postcode",
        suburb_col: "Suburb"
    })
    scatter["Label"] = scatter["Postcode"] + " • " + scatter["Suburb"]
    fig = px.scatter(
        scatter,
        x="Leads",
        y="Conversion_Rate",
        size="Campaigns",
        color="Campaigns",
        hover_name="Label",
        hover_data={"Leads": True, "Referrals": True, "Campaigns": True, "CPR": True},
        title="Leads vs Conversion Rate (size/color = campaign density)"
    )
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0), yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

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

    # Section 3: Creative guidance
    st.markdown("""
    <div class="section">
        <div class="section-header">
            <span class="section-num">3</span>
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
