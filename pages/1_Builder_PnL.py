"""
Builder P&L Dashboard - Executive Edition
McKinsey-style storytelling: High-level to granular
Filename: pages/1_Builder_PnL.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import sys
from pathlib import Path

root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data_loader import load_events, export_to_excel
from src.normalization import normalize_events
from src.builder_pnl import build_builder_pnl, apply_status_bands, compute_paid_share
from src.utils import fmt_currency, fmt_percent, fmt_roas

st.set_page_config(page_title="Builder Economics", page_icon="📊", layout="wide")

# Executive styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .executive-title {
        font-size: 2rem;
        font-weight: 300;
        color: #1a1a2e;
        letter-spacing: -0.02em;
        margin-bottom: 0;
        border-bottom: 3px solid #1a1a2e;
        padding-bottom: 0.75rem;
    }
    .executive-subtitle {
        font-size: 0.9rem;
        color: #6b7280;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    .section-divider {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 3rem 0 2rem 0;
    }
    
    .section-number {
        font-size: 0.75rem;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.25rem;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 1rem;
    }
    
    .executive-summary {
        background: #f8fafc;
        border-left: 4px solid #1a1a2e;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
    }
    .executive-summary h3 {
        font-size: 1rem;
        font-weight: 600;
        color: #1a1a2e;
        margin: 0 0 0.75rem 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .executive-summary p {
        font-size: 1.1rem;
        color: #374151;
        line-height: 1.6;
        margin: 0;
    }
    
    .metric-row {
        display: flex;
        justify-content: space-between;
        padding: 1rem 0;
        border-bottom: 1px solid #f3f4f6;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        font-weight: 400;
    }
    .metric-value {
        font-size: 0.875rem;
        color: #1a1a2e;
        font-weight: 600;
    }
    
    .insight-box {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        border-radius: 4px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }
    .insight-box.positive {
        background: #f0fdf4;
        border-color: #86efac;
    }
    .insight-box.negative {
        background: #fef2f2;
        border-color: #fca5a5;
    }
    .insight-box.neutral {
        background: #f8fafc;
        border-color: #cbd5e1;
    }
    .insight-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .insight-label.positive { color: #166534; }
    .insight-label.negative { color: #991b1b; }
    .insight-label.neutral { color: #475569; }
    .insight-text {
        font-size: 0.9rem;
        color: #1f2937;
        line-height: 1.5;
    }
    
    .kpi-large {
        text-align: center;
        padding: 1.5rem 1rem;
    }
    .kpi-large-value {
        font-size: 2.5rem;
        font-weight: 300;
        color: #1a1a2e;
        line-height: 1;
    }
    .kpi-large-value.positive { color: #059669; }
    .kpi-large-value.negative { color: #dc2626; }
    .kpi-large-label {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }
    
    .recommendation-box {
        background: #1a1a2e;
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 4px;
        margin: 1.5rem 0;
    }
    .recommendation-box h4 {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #9ca3af;
        margin: 0 0 0.5rem 0;
    }
    .recommendation-box p {
        font-size: 1rem;
        line-height: 1.6;
        margin: 0;
        color: #f3f4f6;
    }
    
    .chart-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.5rem;
    }
    .chart-subtitle {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-bottom: 1rem;
    }
    
    .data-table {
        font-size: 0.85rem;
    }
    
    .footer-note {
        font-size: 0.75rem;
        color: #9ca3af;
        text-align: right;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


def load_data():
    if 'events_file' not in st.session_state:
        return None
    events = load_events(st.session_state['events_file'])
    return normalize_events(events) if events is not None else None


def calculate_portfolio_health(pnl):
    """Calculate portfolio health metrics."""
    total_builders = len(pnl)
    profitable = len(pnl[pnl["Profit"] > 0])
    loss_making = len(pnl[pnl["Profit"] < 0])
    
    total_profit = pnl["Profit"].sum()
    total_loss = pnl[pnl["Profit"] < 0]["Profit"].sum()
    
    # Concentration
    top_5_profit = pnl.nlargest(5, "Profit")["Profit"].sum()
    concentration = top_5_profit / total_profit if total_profit > 0 else 0
    
    # ROAS distribution
    high_roas = len(pnl[pnl["ROAS"] >= 3])
    mid_roas = len(pnl[(pnl["ROAS"] >= 1) & (pnl["ROAS"] < 3)])
    low_roas = len(pnl[(pnl["ROAS"] > 0) & (pnl["ROAS"] < 1)])
    
    return {
        "total_builders": total_builders,
        "profitable": profitable,
        "profitable_pct": profitable / total_builders if total_builders > 0 else 0,
        "loss_making": loss_making,
        "total_loss": total_loss,
        "concentration": concentration,
        "high_roas": high_roas,
        "mid_roas": mid_roas,
        "low_roas": low_roas
    }


def generate_executive_summary(pnl, total_rev, total_cost, total_profit, health):
    """Generate the executive summary narrative."""
    margin = total_profit / total_rev if total_rev > 0 else 0
    roas = total_rev / total_cost if total_cost > 0 else 0
    
    # Determine overall health
    if margin > 0.25 and health["profitable_pct"] > 0.7:
        health_status = "strong"
        health_text = "The builder portfolio is performing well"
    elif margin > 0.1 and health["profitable_pct"] > 0.5:
        health_status = "moderate"
        health_text = "The portfolio shows mixed performance requiring attention"
    else:
        health_status = "concerning"
        health_text = "The portfolio faces significant profitability challenges"
    
    summary = f"""{health_text}, generating **{fmt_currency(total_profit)}** in gross profit 
    on **{fmt_currency(total_rev)}** revenue ({margin:.1%} margin). Of {health['total_builders']} builders analyzed, 
    **{health['profitable']}** ({health['profitable_pct']:.0%}) are profitable while **{health['loss_making']}** 
    are operating at a loss, collectively eroding **{fmt_currency(abs(health['total_loss']))}** from the bottom line."""
    
    if health["concentration"] > 0.6:
        summary += f" **Concentration risk is elevated** — the top 5 builders drive {health['concentration']:.0%} of total profit."
    
    return summary, health_status


def create_profit_bridge(pnl):
    """Create profit bridge visualization."""
    # Segment builders
    top_performers = pnl[pnl["Profit"] > 0].nlargest(5, "Profit")
    other_profitable = pnl[(pnl["Profit"] > 0) & (~pnl["BuilderRegionKey"].isin(top_performers["BuilderRegionKey"]))]
    loss_makers = pnl[pnl["Profit"] < 0]
    
    categories = []
    values = []
    colors = []
    
    # Top 5
    for _, row in top_performers.iterrows():
        name = row["BuilderRegionKey"][:18] + "..." if len(row["BuilderRegionKey"]) > 18 else row["BuilderRegionKey"]
        categories.append(name)
        values.append(row["Profit"])
        colors.append("#059669")
    
    # Other profitable
    if len(other_profitable) > 0:
        categories.append(f"Other profitable ({len(other_profitable)})")
        values.append(other_profitable["Profit"].sum())
        colors.append("#10b981")
    
    # Loss makers
    if len(loss_makers) > 0:
        categories.append(f"Loss-making ({len(loss_makers)})")
        values.append(loss_makers["Profit"].sum())
        colors.append("#dc2626")
    
    # Total
    categories.append("NET PROFIT")
    values.append(sum(values))
    colors.append("#1a1a2e")
    
    fig = go.Figure(go.Waterfall(
        x=categories,
        y=values,
        measure=["relative"] * (len(categories) - 1) + ["total"],
        connector={"line": {"color": "#e5e7eb", "width": 1}},
        increasing={"marker": {"color": "#059669"}},
        decreasing={"marker": {"color": "#dc2626"}},
        totals={"marker": {"color": "#1a1a2e"}},
        textposition="outside",
        text=[f"${v/1000:,.0f}K" if abs(v) >= 1000 else f"${v:,.0f}" for v in values],
        textfont={"size": 10, "color": "#374151"}
    ))
    
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=True, gridcolor="#f3f4f6", title="", tickformat="$,.0f"),
        xaxis=dict(tickangle=-35),
        showlegend=False
    )
    
    return fig


def create_roas_margin_scatter(pnl):
    """Create ROAS vs Margin strategic view."""
    df = pnl[(pnl["MediaCost"] > 0) & (pnl["Revenue"] > 0)].copy()
    if df.empty:
        return None, None
    
    df["Margin"] = df["Profit"] / df["Revenue"]
    
    roas_med = df["ROAS"].median()
    margin_med = df["Margin"].median()
    
    def classify(row):
        if row["ROAS"] >= roas_med and row["Margin"] >= margin_med:
            return "Scale"
        elif row["ROAS"] >= roas_med:
            return "Improve margin"
        elif row["Margin"] >= margin_med:
            return "Improve efficiency"
        else:
            return "Review"
    
    df["Action"] = df.apply(classify, axis=1)
    
    color_map = {"Scale": "#059669", "Improve margin": "#3b82f6", "Improve efficiency": "#f59e0b", "Review": "#dc2626"}
    
    fig = go.Figure()
    
    for action, color in color_map.items():
        mask = df["Action"] == action
        if mask.sum() == 0:
            continue
        
        subset = df[mask]
        fig.add_trace(go.Scatter(
            x=subset["ROAS"],
            y=subset["Margin"] * 100,
            mode="markers",
            name=action,
            marker=dict(
                size=8 + np.log1p(subset["MediaCost"]) * 2,
                color=color,
                opacity=0.7,
                line=dict(width=1, color="white")
            ),
            text=subset["BuilderRegionKey"],
            hovertemplate="<b>%{text}</b><br>ROAS: %{x:.2f}x<br>Margin: %{y:.1f}%<extra></extra>"
        ))
    
    fig.add_hline(y=margin_med * 100, line_dash="dot", line_color="#9ca3af", line_width=1)
    fig.add_vline(x=roas_med, line_dash="dot", line_color="#9ca3af", line_width=1)
    
    fig.update_layout(
        height=400,
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="ROAS", gridcolor="#f3f4f6", zeroline=False),
        yaxis=dict(title="Profit Margin %", gridcolor="#f3f4f6", zeroline=True, zerolinecolor="#e5e7eb"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10))
    )
    
    # Summary stats
    summary = df.groupby("Action").agg(
        Count=("BuilderRegionKey", "count"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum")
    ).reset_index()
    
    return fig, summary


def create_trend_analysis(pnl):
    """Create trend analysis for time-series data."""
    ts = (
        pnl.groupby("period_start")
        .agg(Revenue=("Revenue", "sum"), MediaCost=("MediaCost", "sum"), Profit=("Profit", "sum"))
        .reset_index()
    )
    ts["period_start"] = pd.to_datetime(ts["period_start"])
    ts = ts.sort_values("period_start")
    
    ts["Margin"] = np.where(ts["Revenue"] > 0, ts["Profit"] / ts["Revenue"], 0)
    ts["ROAS"] = np.where(ts["MediaCost"] > 0, ts["Revenue"] / ts["MediaCost"], 0)
    
    # Calculate MoM changes
    ts["Profit_pct_change"] = ts["Profit"].pct_change()
    ts["Revenue_pct_change"] = ts["Revenue"].pct_change()
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.12,
        subplot_titles=("", "")
    )
    
    # Revenue bars
    fig.add_trace(go.Bar(
        x=ts["period_start"],
        y=ts["Revenue"],
        name="Revenue",
        marker_color="#e5e7eb",
        opacity=0.8
    ), row=1, col=1)
    
    # Profit line
    fig.add_trace(go.Scatter(
        x=ts["period_start"],
        y=ts["Profit"],
        name="Profit",
        line=dict(color="#059669", width=3),
        mode="lines+markers",
        marker=dict(size=8)
    ), row=1, col=1)
    
    # Margin area
    fig.add_trace(go.Scatter(
        x=ts["period_start"],
        y=ts["Margin"] * 100,
        name="Margin %",
        fill="tozeroy",
        line=dict(color="#3b82f6", width=2),
        fillcolor="rgba(59, 130, 246, 0.1)"
    ), row=2, col=1)
    
    fig.update_layout(
        height=450,
        margin=dict(l=40, r=20, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
        hovermode="x unified"
    )
    
    fig.update_xaxes(gridcolor="#f3f4f6")
    fig.update_yaxes(gridcolor="#f3f4f6", row=1, col=1, tickformat="$,.0f")
    fig.update_yaxes(gridcolor="#f3f4f6", row=2, col=1, tickformat=".0f", title="Margin %")
    
    # Trend direction
    if len(ts) >= 3:
        recent_profit = ts["Profit"].iloc[-3:].mean()
        prior_profit = ts["Profit"].iloc[-6:-3].mean() if len(ts) >= 6 else ts["Profit"].iloc[:-3].mean()
        trend = "improving" if recent_profit > prior_profit else "declining"
        trend_pct = (recent_profit - prior_profit) / abs(prior_profit) if prior_profit != 0 else 0
    else:
        trend, trend_pct = "stable", 0
    
    return fig, trend, trend_pct


def create_concentration_chart(pnl):
    """Create Pareto/concentration analysis."""
    df = pnl.sort_values("Profit", ascending=False).copy()
    df["Cumulative_Profit"] = df["Profit"].cumsum()
    df["Cumulative_Pct"] = df["Cumulative_Profit"] / df["Profit"].sum()
    df["Builder_Pct"] = (np.arange(len(df)) + 1) / len(df)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df["Builder_Pct"] * 100,
        y=df["Cumulative_Pct"] * 100,
        fill="tozeroy",
        line=dict(color="#1a1a2e", width=2),
        fillcolor="rgba(26, 26, 46, 0.1)",
        name="Cumulative Profit"
    ))
    
    # 80/20 reference lines
    fig.add_hline(y=80, line_dash="dot", line_color="#9ca3af", annotation_text="80%", annotation_position="right")
    
    # Find where 80% profit is achieved
    pct_80_idx = (df["Cumulative_Pct"] >= 0.8).idxmax()
    pct_80_builders = df.loc[:pct_80_idx, "Builder_Pct"].iloc[-1] * 100 if pct_80_idx in df.index else 100
    
    fig.add_vline(x=pct_80_builders, line_dash="dot", line_color="#9ca3af")
    
    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="% of Builders", gridcolor="#f3f4f6", range=[0, 100]),
        yaxis=dict(title="% of Profit", gridcolor="#f3f4f6", range=[0, 105]),
        showlegend=False
    )
    
    return fig, pct_80_builders


def main():
    events = load_data()
    
    if events is None:
        st.warning("⚠️ Please upload events data on the Home page first.")
        st.page_link("app.py", label="← Go to Home", icon="🏠")
        return
    
    # Sidebar - minimal, functional
    with st.sidebar:
        st.markdown("#### Parameters")
        
        lens = st.selectbox("Attribution", ["recipient", "payer", "origin"],
            format_func=lambda x: {"recipient": "Recipient", "payer": "Payer", "origin": "Origin"}[x])
        
        freq = st.selectbox("Period", ["ALL", "M", "W"],
            format_func=lambda x: {"ALL": "All Time", "M": "Monthly", "W": "Weekly"}[x])
        
        date_basis = st.selectbox("Date Basis", ["lead_date", "RefDate"],
            format_func=lambda x: "Lead Date" if x == "lead_date" else "Referral Date")
        
        # Month filter
        month_filter = None
        if freq in ("M", "W"):
            month_col = "lead_month_start" if date_basis == "lead_date" else "ref_month_start"
            if month_col in events.columns:
                months = sorted(events[month_col].dropna().unique())
                if months:
                    selected = st.selectbox("Month", ["All"] + [m.strftime("%Y-%m") for m in months])
                    if selected != "All":
                        month_filter = pd.Timestamp(selected + "-01")
        
        st.divider()
        st.markdown("#### Filters")
        min_revenue = st.number_input("Min Revenue", value=0, step=5000)
        min_media = st.number_input("Min Media", value=0, step=5000)
    
    # Filter and build P&L
    filtered_events = events.copy()
    if month_filter is not None:
        month_col = "lead_month_start" if date_basis == "lead_date" else "ref_month_start"
        filtered_events = filtered_events[filtered_events[month_col] == month_filter]
    
    if filtered_events.empty:
        st.warning("No data for selected parameters.")
        return
    
    try:
        pnl = build_builder_pnl(filtered_events, lens=lens, date_basis=date_basis, freq=freq)
    except Exception as e:
        st.error(f"Error: {e}")
        return
    
    if pnl.empty:
        st.warning("No P&L data available.")
        return
    
    pnl = apply_status_bands(pnl)
    pnl = compute_paid_share(filtered_events, pnl, lens)
    
    # Apply filters
    if min_revenue > 0:
        pnl = pnl[pnl["Revenue"] >= min_revenue]
    if min_media > 0:
        pnl = pnl[pnl["MediaCost"] >= min_media]
    
    pnl = pnl[~((pnl["Revenue"] == 0) & (pnl["MediaCost"] == 0))]
    
    if pnl.empty:
        st.warning("No builders match filters.")
        return
    
    # Calculate metrics
    total_rev = pnl["Revenue"].sum()
    total_cost = pnl["MediaCost"].sum()
    total_profit = pnl["Profit"].sum()
    overall_roas = total_rev / total_cost if total_cost > 0 else 0
    overall_margin = total_profit / total_rev if total_rev > 0 else 0
    
    health = calculate_portfolio_health(pnl)
    summary_text, health_status = generate_executive_summary(pnl, total_rev, total_cost, total_profit, health)
    
    # ═══════════════════════════════════════════════════════════════
    # TITLE
    # ═══════════════════════════════════════════════════════════════
    period_str = month_filter.strftime("%B %Y") if month_filter else "All Time"
    
    st.markdown(f'<h1 class="executive-title">Builder Economics Review</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="executive-subtitle">{lens.title()} Attribution | {period_str} | {health["total_builders"]} Builders Analyzed</p>', unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 1: EXECUTIVE SUMMARY
    # ═══════════════════════════════════════════════════════════════
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-number">01</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Executive Summary</h2>', unsafe_allow_html=True)
    
    st.markdown(f'<div class="executive-summary"><h3>Key Finding</h3><p>{summary_text}</p></div>', unsafe_allow_html=True)
    
    # Primary KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        profit_class = "positive" if total_profit > 0 else "negative"
        st.markdown(f'''
        <div class="kpi-large">
            <div class="kpi-large-value {profit_class}">{fmt_currency(total_profit)}</div>
            <div class="kpi-large-label">Gross Profit</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="kpi-large">
            <div class="kpi-large-value">{overall_margin:.1%}</div>
            <div class="kpi-large-label">Profit Margin</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="kpi-large">
            <div class="kpi-large-value">{overall_roas:.2f}x</div>
            <div class="kpi-large-label">Portfolio ROAS</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="kpi-large">
            <div class="kpi-large-value">{health["profitable_pct"]:.0%}</div>
            <div class="kpi-large-label">Profitable Builders</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 2: PROFIT COMPOSITION
    # ═══════════════════════════════════════════════════════════════
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-number">02</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Profit Composition</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<p class="chart-title">Profit contribution by builder segment</p>', unsafe_allow_html=True)
        st.markdown('<p class="chart-subtitle">Top performers drive majority of profit; loss-makers erode gains</p>', unsafe_allow_html=True)
        fig_bridge = create_profit_bridge(pnl)
        st.plotly_chart(fig_bridge, use_container_width=True)
    
    with col2:
        # Key metrics
        top_5 = pnl.nlargest(5, "Profit")
        loss_makers = pnl[pnl["Profit"] < 0]
        
        st.markdown(f'''
        <div class="metric-row">
            <span class="metric-label">Total Revenue</span>
            <span class="metric-value">{fmt_currency(total_rev)}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Total Media Cost</span>
            <span class="metric-value">{fmt_currency(total_cost)}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Top 5 Contribution</span>
            <span class="metric-value">{top_5["Profit"].sum() / total_profit:.0%}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Loss Maker Drag</span>
            <span class="metric-value" style="color: #dc2626;">{fmt_currency(loss_makers["Profit"].sum())}</span>
        </div>
        ''', unsafe_allow_html=True)
        
        # Concentration insight
        fig_conc, pct_80 = create_concentration_chart(pnl)
        st.markdown(f'''
        <div class="insight-box neutral">
            <p class="insight-label neutral">Concentration</p>
            <p class="insight-text"><strong>{pct_80:.0f}%</strong> of builders generate 80% of profit</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 3: STRATEGIC POSITIONING
    # ═══════════════════════════════════════════════════════════════
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-number">03</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Strategic Positioning</h2>', unsafe_allow_html=True)
    
    result = create_roas_margin_scatter(pnl)
    if result[0]:
        fig_scatter, action_summary = result
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<p class="chart-title">Builder segmentation by efficiency and profitability</p>', unsafe_allow_html=True)
            st.markdown('<p class="chart-subtitle">Bubble size indicates media spend; position determines recommended action</p>', unsafe_allow_html=True)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            st.markdown("**Recommended Actions**")
            
            for _, row in action_summary.iterrows():
                action = row["Action"]
                count = int(row["Count"])
                profit = row["Profit"]
                
                if action == "Scale":
                    box_class, label = "positive", "SCALE"
                    text = f"**{count} builders** ready for increased investment. Combined profit: **{fmt_currency(profit)}**"
                elif action == "Improve margin":
                    box_class, label = "neutral", "IMPROVE MARGIN"
                    text = f"**{count} builders** efficient but low margin. Focus on pricing/costs."
                elif action == "Improve efficiency":
                    box_class, label = "neutral", "IMPROVE EFFICIENCY"
                    text = f"**{count} builders** profitable but inefficient. Optimize media spend."
                else:
                    box_class, label = "negative", "REVIEW"
                    text = f"**{count} builders** underperforming. Evaluate continuation."
                
                st.markdown(f'''
                <div class="insight-box {box_class}">
                    <p class="insight-label {box_class}">{label}</p>
                    <p class="insight-text">{text}</p>
                </div>
                ''', unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 4: TREND ANALYSIS (if time-series)
    # ═══════════════════════════════════════════════════════════════
    if freq in ("M", "W") and "period_start" in pnl.columns and pnl["period_start"].notna().sum() > 2:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown('<p class="section-number">04</p>', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">Performance Trajectory</h2>', unsafe_allow_html=True)
        
        fig_trend, trend_dir, trend_pct = create_trend_analysis(pnl)
        
        trend_class = "positive" if trend_dir == "improving" else "negative"
        st.markdown(f'''
        <div class="insight-box {trend_class}">
            <p class="insight-label {trend_class}">TREND</p>
            <p class="insight-text">Profit is <strong>{trend_dir}</strong> — recent 3-period average vs prior: <strong>{trend_pct:+.1%}</strong></p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 5: DETAILED BUILDER VIEW
    # ═══════════════════════════════════════════════════════════════
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-number">05</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Builder Detail</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("Search builders", placeholder="Filter by name...", label_visibility="collapsed")
    with col2:
        sort_col = st.selectbox("Sort by", ["Profit", "Revenue", "MediaCost", "ROAS"], label_visibility="collapsed")
    
    display_pnl = pnl.copy()
    if search:
        display_pnl = display_pnl[display_pnl["BuilderRegionKey"].str.lower().str.contains(search.lower(), na=False)]
    
    display_pnl = display_pnl.sort_values(sort_col, ascending=False)
    
    # Format for display
    table_df = display_pnl[["BuilderRegionKey", "Revenue", "MediaCost", "Profit", "ROAS", "Margin_pct"]].head(30).copy()
    table_df.columns = ["Builder", "Revenue", "Media Cost", "Profit", "ROAS", "Margin"]
    
    st.dataframe(
        table_df.style.format({
            "Revenue": "${:,.0f}",
            "Media Cost": "${:,.0f}",
            "Profit": "${:,.0f}",
            "ROAS": "{:.2f}x",
            "Margin": "{:.1%}"
        }).applymap(lambda x: "color: #059669" if isinstance(x, (int, float)) and x > 0 else ("color: #dc2626" if isinstance(x, (int, float)) and x < 0 else ""), subset=["Profit"]),
        hide_index=True,
        use_container_width=True,
        height=400
    )
    
    # Download
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        st.download_button("Export Excel", export_to_excel(display_pnl, "pnl.xlsx"), "builder_pnl.xlsx")
    with col2:
        st.download_button("Export CSV", display_pnl.to_csv(index=False), "builder_pnl.csv", "text/csv")
    
    # ═══════════════════════════════════════════════════════════════
    # RECOMMENDATION
    # ═══════════════════════════════════════════════════════════════
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    # Generate recommendation
    scale_builders = pnl[(pnl["ROAS"] >= pnl["ROAS"].median()) & (pnl["Profit"] > 0)]
    review_builders = pnl[pnl["Profit"] < 0]
    
    rec_text = f"""Based on this analysis, we recommend: (1) <strong>Increase investment</strong> in the top {len(scale_builders)} high-ROAS 
    profitable builders to accelerate growth; (2) <strong>Conduct deep-dive reviews</strong> on {len(review_builders)} loss-making builders 
    to determine turnaround potential or exit; (3) <strong>Monitor concentration risk</strong> given top 5 builders drive {health['concentration']:.0%} of profit."""
    
    st.markdown(f'''
    <div class="recommendation-box">
        <h4>Recommendation</h4>
        <p>{rec_text}</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Footer
    from datetime import datetime
    st.markdown(f'<p class="footer-note">Analysis generated {datetime.now().strftime("%B %d, %Y")} | Data as of {period_str}</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()