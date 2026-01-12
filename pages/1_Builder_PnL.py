"""
Builder P&L Dashboard - Premium Edition
Filename: pages/1_Builder_PnL.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

import sys
from pathlib import Path

root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data_loader import load_events, export_to_excel
from src.normalization import normalize_events
from src.builder_pnl import build_builder_pnl, apply_status_bands, compute_paid_share
from src.utils import fmt_currency, fmt_percent, fmt_roas, get_status_color

# Page config
st.set_page_config(page_title="Builder P&L", page_icon="📊", layout="wide")

# Premium CSS
st.markdown("""
<style>
    /* Main containers */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2.5rem; }
    .main-header p { color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; }
    
    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 12px;
        padding: 1.25rem;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .kpi-card.profit { border-left-color: #22c55e; }
    .kpi-card.loss { border-left-color: #ef4444; }
    .kpi-card.warning { border-left-color: #f59e0b; }
    .kpi-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value { font-size: 1.75rem; font-weight: 700; color: #1e293b; margin: 0.25rem 0; }
    .kpi-delta { font-size: 0.875rem; }
    .kpi-delta.positive { color: #22c55e; }
    .kpi-delta.negative { color: #ef4444; }
    
    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .section-header h2 { margin: 0; font-size: 1.5rem; color: #1e293b; }
    
    /* Insight cards */
    .insight-card {
        background: linear-gradient(135deg, #fefce8 0%, #fef9c3 100%);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        border-left: 4px solid #eab308;
        margin: 1rem 0;
    }
    .insight-card.success {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left-color: #22c55e;
    }
    .insight-card.danger {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left-color: #ef4444;
    }
    
    /* Quadrant labels */
    .quadrant-label {
        font-size: 0.75rem;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
    }
    .q-star { background: #dcfce7; color: #166534; }
    .q-growth { background: #dbeafe; color: #1e40af; }
    .q-optimize { background: #fef3c7; color: #92400e; }
    .q-review { background: #fee2e2; color: #991b1b; }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: #3b82f6 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


def load_data():
    if 'events_file' not in st.session_state:
        return None
    events = load_events(st.session_state['events_file'])
    if events is None:
        return None
    return normalize_events(events)


def render_kpi_card(label, value, delta=None, delta_label=None, card_type="default"):
    delta_html = ""
    if delta is not None:
        delta_class = "positive" if delta >= 0 else "negative"
        delta_sign = "+" if delta >= 0 else ""
        delta_html = f'<div class="kpi-delta {delta_class}">{delta_sign}{delta:.1%} {delta_label or ""}</div>'
    
    return f"""
    <div class="kpi-card {card_type}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """


def generate_insights(pnl, total_profit, total_rev, total_cost):
    """Generate AI-like insights from the data."""
    insights = []
    
    # Profit insight
    margin = total_profit / total_rev if total_rev > 0 else 0
    if margin > 0.3:
        insights.append(("success", f"💰 **Strong profitability** — {margin:.1%} margin indicates healthy unit economics"))
    elif margin < 0.1:
        insights.append(("danger", f"⚠️ **Margin pressure** — {margin:.1%} margin needs attention"))
    
    # Loss makers
    loss_makers = pnl[pnl["Profit"] < 0]
    if len(loss_makers) > 0:
        loss_total = abs(loss_makers["Profit"].sum())
        insights.append(("danger", f"🔴 **{len(loss_makers)} builders** are loss-making, totaling **{fmt_currency(loss_total)}** in losses"))
    
    # Top performer
    if not pnl.empty:
        top = pnl.nlargest(1, "Profit").iloc[0]
        insights.append(("success", f"⭐ **Top performer:** {top['BuilderRegionKey']} with **{fmt_currency(top['Profit'])}** profit"))
    
    # ROAS distribution
    high_roas = pnl[pnl["ROAS"] > 3]
    if len(high_roas) > 0:
        insights.append(("success", f"🚀 **{len(high_roas)} builders** achieving ROAS > 3x — consider scaling"))
    
    # Concentration risk
    if not pnl.empty and total_profit > 0:
        top_5_profit = pnl.nlargest(5, "Profit")["Profit"].sum()
        concentration = top_5_profit / total_profit
        if concentration > 0.7:
            insights.append(("warning", f"⚡ **Concentration risk** — Top 5 builders drive {concentration:.0%} of profit"))
    
    return insights


def create_waterfall_chart(pnl):
    """Create profit waterfall chart."""
    # Get top contributors and aggregate rest
    top_n = 8
    sorted_pnl = pnl.nlargest(top_n, "Profit")
    others_profit = pnl[~pnl["BuilderRegionKey"].isin(sorted_pnl["BuilderRegionKey"])]["Profit"].sum()
    
    labels = sorted_pnl["BuilderRegionKey"].tolist()
    values = sorted_pnl["Profit"].tolist()
    
    if others_profit != 0:
        labels.append("Others")
        values.append(others_profit)
    
    labels.append("Total")
    values.append(sum(values[:-1]) if "Others" in labels else sum(values))
    
    measures = ["relative"] * (len(labels) - 1) + ["total"]
    
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in values[:-1]] + ["#3b82f6"]
    
    fig = go.Figure(go.Waterfall(
        x=labels,
        y=values,
        measure=measures,
        connector={"line": {"color": "#e2e8f0"}},
        increasing={"marker": {"color": "#22c55e"}},
        decreasing={"marker": {"color": "#ef4444"}},
        totals={"marker": {"color": "#3b82f6"}},
        textposition="outside",
        text=[f"${v/1000:.0f}K" for v in values],
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title={"text": "Profit Contribution Waterfall", "font": {"size": 16}},
        height=400,
        showlegend=False,
        yaxis_title="Profit ($)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#e2e8f0")
    )
    
    return fig


def create_treemap(pnl):
    """Create revenue/profit treemap."""
    df = pnl[pnl["Revenue"] > 0].copy()
    if df.empty:
        return None
    
    df["ProfitMargin"] = df["Profit"] / df["Revenue"]
    df["Label"] = df["BuilderRegionKey"].str[:15]
    
    fig = px.treemap(
        df.nlargest(30, "Revenue"),
        path=["Status", "Label"],
        values="Revenue",
        color="ProfitMargin",
        color_continuous_scale=["#ef4444", "#fbbf24", "#22c55e"],
        color_continuous_midpoint=0.15,
        hover_data={"Revenue": ":$,.0f", "Profit": ":$,.0f", "ProfitMargin": ":.1%"}
    )
    
    fig.update_layout(
        title={"text": "Revenue Distribution by Builder", "font": {"size": 16}},
        height=450,
        margin=dict(t=50, l=10, r=10, b=10)
    )
    
    return fig


def create_quadrant_chart(pnl):
    """Create strategic quadrant analysis."""
    df = pnl[(pnl["MediaCost"] > 0) & (pnl["Revenue"] > 0)].copy()
    if df.empty:
        return None
    
    df["Margin_pct"] = df["Profit"] / df["Revenue"]
    
    roas_med = df["ROAS"].median()
    margin_med = df["Margin_pct"].median()
    
    def get_quadrant(row):
        if row["ROAS"] >= roas_med and row["Margin_pct"] >= margin_med:
            return "⭐ Stars"
        elif row["ROAS"] >= roas_med and row["Margin_pct"] < margin_med:
            return "🚀 Growth"
        elif row["ROAS"] < roas_med and row["Margin_pct"] >= margin_med:
            return "⚙️ Optimize"
        else:
            return "🔍 Review"
    
    df["Quadrant"] = df.apply(get_quadrant, axis=1)
    
    color_map = {
        "⭐ Stars": "#22c55e",
        "🚀 Growth": "#3b82f6", 
        "⚙️ Optimize": "#f59e0b",
        "🔍 Review": "#ef4444"
    }
    
    fig = go.Figure()
    
    for quadrant, color in color_map.items():
        q_df = df[df["Quadrant"] == quadrant]
        if not q_df.empty:
            fig.add_trace(go.Scatter(
                x=q_df["ROAS"],
                y=q_df["Margin_pct"] * 100,
                mode="markers",
                name=quadrant,
                marker=dict(
                    size=8 + np.sqrt(q_df["MediaCost"]) / 50,
                    color=color,
                    opacity=0.7,
                    line=dict(width=1, color="white")
                ),
                text=q_df["BuilderRegionKey"],
                hovertemplate="<b>%{text}</b><br>ROAS: %{x:.2f}x<br>Margin: %{y:.1f}%<br><extra></extra>"
            ))
    
    # Quadrant lines
    fig.add_hline(y=margin_med * 100, line_dash="dash", line_color="#94a3b8", line_width=1)
    fig.add_vline(x=roas_med, line_dash="dash", line_color="#94a3b8", line_width=1)
    
    # Quadrant labels
    x_max, y_max = df["ROAS"].max() * 1.1, df["Margin_pct"].max() * 100 * 1.1
    x_min, y_min = 0, df["Margin_pct"].min() * 100 * 1.1 if df["Margin_pct"].min() < 0 else 0
    
    fig.add_annotation(x=x_max*0.85, y=y_max*0.9, text="⭐ STARS", showarrow=False, font=dict(size=12, color="#22c55e"))
    fig.add_annotation(x=x_max*0.85, y=y_min + (margin_med*100 - y_min)*0.2, text="🚀 GROWTH", showarrow=False, font=dict(size=12, color="#3b82f6"))
    fig.add_annotation(x=roas_med*0.3, y=y_max*0.9, text="⚙️ OPTIMIZE", showarrow=False, font=dict(size=12, color="#f59e0b"))
    fig.add_annotation(x=roas_med*0.3, y=y_min + (margin_med*100 - y_min)*0.2, text="🔍 REVIEW", showarrow=False, font=dict(size=12, color="#ef4444"))
    
    fig.update_layout(
        title={"text": "Strategic Quadrant Analysis", "font": {"size": 16}},
        xaxis_title="ROAS",
        yaxis_title="Profit Margin %",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#e2e8f0", zeroline=True, zerolinecolor="#94a3b8"),
        yaxis=dict(gridcolor="#e2e8f0", zeroline=True, zerolinecolor="#94a3b8")
    )
    
    return fig, df


def create_trend_chart(pnl):
    """Create enhanced trend analysis."""
    ts = (
        pnl.groupby("period_start")
        .agg(
            Revenue=("Revenue", "sum"),
            MediaCost=("MediaCost", "sum"),
            Profit=("Profit", "sum"),
            Builders=("BuilderRegionKey", "nunique")
        )
        .reset_index()
    )
    ts["period_start"] = pd.to_datetime(ts["period_start"])
    ts["Margin"] = np.where(ts["Revenue"] > 0, ts["Profit"] / ts["Revenue"], 0)
    ts["ROAS"] = np.where(ts["MediaCost"] > 0, ts["Revenue"] / ts["MediaCost"], 0)
    
    # Moving averages
    ts["Revenue_MA"] = ts["Revenue"].rolling(3, min_periods=1).mean()
    ts["Profit_MA"] = ts["Profit"].rolling(3, min_periods=1).mean()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Revenue & Profit", "ROAS Trend", "Margin Trend", "Active Builders"),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # Revenue & Profit
    fig.add_trace(go.Bar(x=ts["period_start"], y=ts["Revenue"], name="Revenue", marker_color="#3b82f6", opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts["period_start"], y=ts["Profit_MA"], name="Profit (MA)", line=dict(color="#22c55e", width=3)), row=1, col=1)
    
    # ROAS
    fig.add_trace(go.Scatter(x=ts["period_start"], y=ts["ROAS"], name="ROAS", fill="tozeroy", line=dict(color="#8b5cf6"), fillcolor="rgba(139,92,246,0.2)"), row=1, col=2)
    fig.add_hline(y=1, line_dash="dash", line_color="#ef4444", row=1, col=2)
    
    # Margin
    fig.add_trace(go.Scatter(x=ts["period_start"], y=ts["Margin"]*100, name="Margin %", fill="tozeroy", line=dict(color="#f59e0b"), fillcolor="rgba(245,158,11,0.2)"), row=2, col=1)
    
    # Builders
    fig.add_trace(go.Bar(x=ts["period_start"], y=ts["Builders"], name="Builders", marker_color="#06b6d4"), row=2, col=2)
    
    fig.update_layout(
        height=500,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    fig.update_xaxes(gridcolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#e2e8f0")
    
    return fig


def create_distribution_charts(pnl):
    """Create distribution analysis."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Profit Distribution", "ROAS Distribution", "Revenue Distribution")
    )
    
    # Profit histogram
    fig.add_trace(go.Histogram(x=pnl["Profit"], nbinsx=30, marker_color="#3b82f6", opacity=0.7, name="Profit"), row=1, col=1)
    fig.add_vline(x=0, line_color="#ef4444", line_width=2, row=1, col=1)
    fig.add_vline(x=pnl["Profit"].median(), line_dash="dash", line_color="#22c55e", row=1, col=1)
    
    # ROAS histogram
    roas_clipped = pnl["ROAS"].clip(upper=10)
    fig.add_trace(go.Histogram(x=roas_clipped, nbinsx=30, marker_color="#8b5cf6", opacity=0.7, name="ROAS"), row=1, col=2)
    fig.add_vline(x=1, line_color="#ef4444", line_width=2, row=1, col=2)
    
    # Revenue histogram
    fig.add_trace(go.Histogram(x=pnl["Revenue"], nbinsx=30, marker_color="#22c55e", opacity=0.7, name="Revenue"), row=1, col=3)
    
    fig.update_layout(
        height=300,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    fig.update_xaxes(gridcolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#e2e8f0")
    
    return fig


def create_builder_comparison(pnl, selected_builders):
    """Create builder comparison radar chart."""
    if len(selected_builders) < 2:
        return None
    
    df = pnl[pnl["BuilderRegionKey"].isin(selected_builders)].copy()
    
    # Normalize metrics
    metrics = ["Revenue", "Profit", "ROAS", "Margin_pct", "N_events"]
    available_metrics = [m for m in metrics if m in df.columns]
    
    for m in available_metrics:
        max_val = df[m].max()
        df[f"{m}_norm"] = df[m] / max_val if max_val > 0 else 0
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set2
    for i, (_, row) in enumerate(df.iterrows()):
        values = [row.get(f"{m}_norm", 0) for m in available_metrics]
        values.append(values[0])  # Close the polygon
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=available_metrics + [available_metrics[0]],
            fill='toself',
            name=row["BuilderRegionKey"][:20],
            line_color=colors[i % len(colors)],
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=400,
        title="Builder Comparison"
    )
    
    return fig


def main():
    events = load_data()
    
    if events is None:
        st.warning("⚠️ Please upload events data on the Home page first.")
        st.page_link("app.py", label="← Go to Home", icon="🏠")
        return
    
    st.session_state['events_df'] = events
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        lens = st.selectbox(
            "Attribution Lens",
            options=["recipient", "payer", "origin"],
            format_func=lambda x: {"recipient": "📥 Recipient", "payer": "💰 Payer", "origin": "🎯 Origin"}[x]
        )
        
        date_basis = st.radio("Date Basis", options=["lead_date", "RefDate"], format_func=lambda x: "Lead Date" if x == "lead_date" else "Referral Date", horizontal=True)
        
        freq = st.radio("Time Grain", options=["ALL", "M", "W"], format_func=lambda x: {"ALL": "All Time", "M": "Monthly", "W": "Weekly"}[x], horizontal=True)
        
        # Month filter
        month_filter = None
        if freq in ("M", "W"):
            month_col = "lead_month_start" if date_basis == "lead_date" else "ref_month_start"
            if month_col in events.columns:
                months = sorted(events[month_col].dropna().unique())
                if months:
                    month_options = ["All"] + [m.strftime("%Y-%m") for m in months]
                    selected_month = st.selectbox("📅 Period", month_options)
                    if selected_month != "All":
                        month_filter = pd.Timestamp(selected_month + "-01")
        
        st.divider()
        st.markdown("### 🔍 Filters")
        
        min_revenue = st.number_input("Min Revenue", value=0, step=1000, format="%d")
        min_media = st.number_input("Min Media Spend", value=0, step=1000, format="%d")
        include_zero = st.checkbox("Include zero activity", value=False)
        
        st.divider()
        st.markdown("### 📊 Display")
        
        top_n = st.slider("Builders to show", 10, 100, 25, 5)
        sort_by = st.selectbox("Sort by", ["Profit", "Revenue", "MediaCost", "ROAS"])
    
    # Filter events
    filtered_events = events.copy()
    if month_filter is not None:
        month_col = "lead_month_start" if date_basis == "lead_date" else "ref_month_start"
        filtered_events = filtered_events[filtered_events[month_col] == month_filter]
    
    if filtered_events.empty:
        st.warning("No data for selected filters.")
        return
    
    # Build P&L
    try:
        pnl = build_builder_pnl(filtered_events, lens=lens, date_basis=date_basis, freq=freq)
    except Exception as e:
        st.error(f"Error building P&L: {e}")
        return
    
    if pnl.empty:
        st.warning("No P&L data available.")
        return
    
    pnl = apply_status_bands(pnl)
    pnl = compute_paid_share(filtered_events, pnl, lens)
    
    # Apply filters
    if not include_zero:
        pnl = pnl[~((pnl["Revenue"] == 0) & (pnl["MediaCost"] == 0))]
    if min_revenue > 0:
        pnl = pnl[pnl["Revenue"] >= min_revenue]
    if min_media > 0:
        pnl = pnl[pnl["MediaCost"] >= min_media]
    
    if pnl.empty:
        st.warning("No builders match filters.")
        return
    
    # Calculate totals
    total_rev = pnl["Revenue"].sum()
    total_cost = pnl["MediaCost"].sum()
    total_profit = pnl["Profit"].sum()
    overall_roas = total_rev / total_cost if total_cost > 0 else 0
    overall_margin = total_profit / total_rev if total_rev > 0 else 0
    n_builders = pnl["BuilderRegionKey"].nunique()
    
    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1>📊 Builder P&L Dashboard</h1>
        <p>Analyzing {n_builders:,} builders | {lens.title()} lens | {freq if freq != "ALL" else "All Time"}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(render_kpi_card("Revenue", fmt_currency(total_rev)), unsafe_allow_html=True)
    with col2:
        st.markdown(render_kpi_card("Media Cost", fmt_currency(total_cost)), unsafe_allow_html=True)
    with col3:
        card_type = "profit" if total_profit > 0 else "loss"
        st.markdown(render_kpi_card("Gross Profit", fmt_currency(total_profit), card_type=card_type), unsafe_allow_html=True)
    with col4:
        st.markdown(render_kpi_card("ROAS", f"{overall_roas:.2f}x"), unsafe_allow_html=True)
    with col5:
        st.markdown(render_kpi_card("Margin", f"{overall_margin:.1%}"), unsafe_allow_html=True)
    with col6:
        st.markdown(render_kpi_card("Builders", f"{n_builders:,}"), unsafe_allow_html=True)
    
    # Insights
    insights = generate_insights(pnl, total_profit, total_rev, total_cost)
    if insights:
        st.markdown("### 💡 Key Insights")
        for insight_type, insight_text in insights[:3]:
            st.markdown(f'<div class="insight-card {insight_type}">{insight_text}</div>', unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Trends", "🎯 Strategy", "🏆 Rankings", "🔬 Deep Dive"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_waterfall = create_waterfall_chart(pnl)
            st.plotly_chart(fig_waterfall, use_container_width=True)
        
        with col2:
            fig_treemap = create_treemap(pnl)
            if fig_treemap:
                st.plotly_chart(fig_treemap, use_container_width=True)
        
        # Distribution
        st.markdown("### 📊 Distribution Analysis")
        fig_dist = create_distribution_charts(pnl)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab2:
        if freq in ("M", "W") and "period_start" in pnl.columns:
            fig_trend = create_trend_chart(pnl)
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Select Monthly or Weekly time grain to see trends.")
    
    with tab3:
        result = create_quadrant_chart(pnl)
        if result:
            fig_quad, quad_df = result
            st.plotly_chart(fig_quad, use_container_width=True)
            
            # Quadrant summary
            st.markdown("### Quadrant Summary")
            quad_summary = quad_df.groupby("Quadrant").agg(
                Builders=("BuilderRegionKey", "count"),
                Revenue=("Revenue", "sum"),
                Profit=("Profit", "sum"),
                Avg_ROAS=("ROAS", "mean")
            ).reset_index()
            
            q_cols = st.columns(4)
            for i, (_, row) in enumerate(quad_summary.iterrows()):
                with q_cols[i % 4]:
                    st.metric(row["Quadrant"], f"{row['Builders']} builders", f"${row['Profit']/1000:.0f}K profit")
    
    with tab4:
        st.markdown("### 🏆 Builder Performance Ranking")
        
        # Builder search
        search = st.text_input("🔍 Search builders", placeholder="Type to filter...")
        
        display_pnl = pnl.copy()
        if search:
            display_pnl = display_pnl[display_pnl["BuilderRegionKey"].str.lower().str.contains(search.lower(), na=False)]
        
        pnl_sorted = display_pnl.sort_values(sort_by, ascending=False).head(top_n)
        
        # Format table
        display_cols = ["BuilderRegionKey", "Status", "Revenue", "MediaCost", "Profit", "ROAS", "Margin_pct"]
        display_df = pnl_sorted[display_cols].copy()
        display_df.columns = ["Builder", "Status", "Revenue", "Media Cost", "Profit", "ROAS", "Margin"]
        
        st.dataframe(
            display_df.style.format({
                "Revenue": "${:,.0f}",
                "Media Cost": "${:,.0f}",
                "Profit": "${:,.0f}",
                "ROAS": "{:.2f}x",
                "Margin": "{:.1%}"
            }).background_gradient(subset=["Profit"], cmap="RdYlGn", vmin=-display_df["Profit"].abs().max(), vmax=display_df["Profit"].abs().max()),
            hide_index=True,
            use_container_width=True,
            height=500
        )
        
        # Download
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.download_button("📥 Download CSV", pnl_sorted.to_csv(index=False), f"builder_pnl_{lens}.csv", "text/csv")
        with col2:
            st.download_button("📥 Download Excel", export_to_excel(pnl_sorted, "pnl.xlsx"), f"builder_pnl_{lens}.xlsx")
    
    with tab5:
        st.markdown("### 🔬 Builder Comparison")
        
        builder_list = pnl.nlargest(50, "Revenue")["BuilderRegionKey"].tolist()
        selected_builders = st.multiselect("Select builders to compare (2-5)", builder_list, default=builder_list[:3] if len(builder_list) >= 3 else builder_list)
        
        if len(selected_builders) >= 2:
            fig_compare = create_builder_comparison(pnl, selected_builders)
            if fig_compare:
                st.plotly_chart(fig_compare, use_container_width=True)
            
            # Comparison table
            compare_df = pnl[pnl["BuilderRegionKey"].isin(selected_builders)][["BuilderRegionKey", "Revenue", "MediaCost", "Profit", "ROAS", "Margin_pct"]].copy()
            st.dataframe(compare_df.style.format({
                "Revenue": "${:,.0f}", "MediaCost": "${:,.0f}", "Profit": "${:,.0f}", "ROAS": "{:.2f}x", "Margin_pct": "{:.1%}"
            }), hide_index=True, use_container_width=True)
        else:
            st.info("Select at least 2 builders to compare.")


if __name__ == "__main__":
    main()