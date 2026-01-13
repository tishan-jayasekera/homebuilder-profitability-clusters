"""
Referral Network Command Center v2.0
Operational dashboard for risk detection and budget-constrained optimization.

Key Features:
- At-a-glance critical builder monitor with velocity tracking
- Budget-constrained network optimization
- Full traceability and reasoning for all recommendations
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
from src.network_optimization import (
    calculate_shortfalls_v2,
    calculate_velocity_metrics,
    analyze_network_leverage_v2,
    run_budget_constrained_optimization,
    optimization_result_to_dataframe,
    RiskLevel
)
from src.utils import fmt_currency

# =============================================================================
# PAGE CONFIG & STYLES
# =============================================================================

st.set_page_config(
    page_title="Network Command Center",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.command-header {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}
.command-title { font-size: 1.8rem; font-weight: 700; margin: 0; }
.command-sub { font-size: 0.95rem; color: #94a3b8; margin-top: 0.3rem; }

.alert-critical {
    background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
    color: white;
    padding: 1rem 1.2rem;
    border-radius: 10px;
    margin-bottom: 0.75rem;
}
.alert-warning {
    background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
    color: white;
    padding: 1rem 1.2rem;
    border-radius: 10px;
    margin-bottom: 0.75rem;
}
.alert-title { font-weight: 600; font-size: 1rem; margin-bottom: 0.3rem; }
.alert-body { font-size: 0.9rem; opacity: 0.95; }

.metric-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.75rem;
    margin: 1rem 0;
}
.metric-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.9rem;
    text-align: center;
}
.metric-box.critical { border-left: 4px solid #dc2626; }
.metric-box.warning { border-left: 4px solid #f97316; }
.metric-box.success { border-left: 4px solid #16a34a; }
.metric-label { font-size: 0.72rem; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em; }
.metric-value { font-size: 1.4rem; font-weight: 700; color: #0f172a; margin-top: 0.2rem; }
.metric-delta { font-size: 0.8rem; margin-top: 0.2rem; }
.metric-delta.neg { color: #dc2626; }
.metric-delta.pos { color: #16a34a; }

.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e293b;
    margin: 1.5rem 0 0.5rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e2e8f0;
}

.builder-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.5rem;
}
.builder-card.critical { border-left: 4px solid #dc2626; background: #fef2f2; }
.builder-card.high { border-left: 4px solid #f97316; background: #fff7ed; }

.tag { display: inline-block; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 500; }
.tag-critical { background: #dc2626; color: white; }
.tag-high { background: #f97316; color: white; }
.tag-medium { background: #eab308; color: black; }
.tag-low { background: #22c55e; color: white; }
.tag-decel { background: #7c3aed; color: white; }

.plan-summary {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white;
    padding: 1.2rem;
    border-radius: 12px;
    margin: 1rem 0;
}
.plan-title { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; }
.plan-value { font-size: 1.6rem; font-weight: 700; margin-top: 0.3rem; }
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load and normalize events data."""
    if "events_file" not in st.session_state:
        return None
    events = load_events(st.session_state["events_file"])
    return normalize_events(events) if events is not None else None


def get_date_info(events):
    """Get date column and available months."""
    date_col = "lead_date" if "lead_date" in events.columns else "RefDate"
    if date_col not in events.columns:
        return None, []
    
    dates = pd.to_datetime(events[date_col], errors="coerce").dropna()
    if dates.empty:
        return date_col, []
    
    months = pd.date_range(
        start=dates.min().replace(day=1),
        end=dates.max().replace(day=1),
        freq="MS"
    )
    return date_col, months


# =============================================================================
# VISUALIZATION COMPONENTS
# =============================================================================

def render_critical_alerts(critical_builders: pd.DataFrame, decel_builders: pd.DataFrame):
    """Render top-of-page critical alerts."""
    alerts = []
    
    # Critical shortfall alerts
    if not critical_builders.empty:
        top_critical = critical_builders.head(3)
        names = ", ".join(top_critical["BuilderRegionKey"].tolist())
        total_gap = critical_builders["Projected_Shortfall"].sum()
        alerts.append({
            "type": "critical",
            "title": f"🚨 {len(critical_builders)} Builders at Critical Risk",
            "body": f"Top: {names}. Total gap: {total_gap:.0f} leads. Immediate action required."
        })
    
    # Deceleration alerts
    severe_decel = decel_builders[decel_builders["decel_severity"] == "Severe"]
    if not severe_decel.empty:
        names = ", ".join(severe_decel["BuilderRegionKey"].head(3).tolist())
        alerts.append({
            "type": "warning",
            "title": f"📉 {len(severe_decel)} Builders with Severe Velocity Drop",
            "body": f"Affected: {names}. Recent pace crashed vs 14-day average."
        })
    
    # Render alerts
    for alert in alerts:
        css_class = "alert-critical" if alert["type"] == "critical" else "alert-warning"
        st.markdown(f'''
        <div class="{css_class}">
            <div class="alert-title">{alert["title"]}</div>
            <div class="alert-body">{alert["body"]}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    return len(alerts) > 0


def render_kpi_row(shortfall_df: pd.DataFrame, velocity_df: pd.DataFrame, budget: float):
    """Render summary KPI metrics."""
    total_builders = len(shortfall_df)
    at_risk = len(shortfall_df[shortfall_df["Projected_Shortfall"] > 0])
    total_shortfall = shortfall_df["Projected_Shortfall"].sum()
    total_surplus = shortfall_df["Projected_Surplus"].sum()
    decel_count = len(velocity_df[velocity_df["is_decelerating"]]) if not velocity_df.empty else 0
    
    risk_pct = at_risk / total_builders * 100 if total_builders > 0 else 0
    
    st.markdown(f'''
    <div class="metric-row">
        <div class="metric-box {'critical' if risk_pct > 30 else 'warning' if risk_pct > 15 else ''}">
            <div class="metric-label">Builders at Risk</div>
            <div class="metric-value">{at_risk} / {total_builders}</div>
            <div class="metric-delta neg">{risk_pct:.0f}% of portfolio</div>
        </div>
        <div class="metric-box critical">
            <div class="metric-label">Total Lead Gap</div>
            <div class="metric-value">{total_shortfall:,.0f}</div>
            <div class="metric-delta neg">leads needed</div>
        </div>
        <div class="metric-box success">
            <div class="metric-label">Available Surplus</div>
            <div class="metric-value">{total_surplus:,.0f}</div>
            <div class="metric-delta pos">leads excess</div>
        </div>
        <div class="metric-box {'warning' if decel_count > 0 else ''}">
            <div class="metric-label">Decelerating</div>
            <div class="metric-value">{decel_count}</div>
            <div class="metric-delta neg">velocity drops</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Budget Available</div>
            <div class="metric-value">${budget:,.0f}</div>
            <div class="metric-delta">optimization pool</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)


def render_critical_builder_table(shortfall_df: pd.DataFrame, velocity_df: pd.DataFrame):
    """Render the critical builders monitoring table."""
    # Merge velocity data
    df = shortfall_df.copy()
    if not velocity_df.empty:
        df = df.merge(velocity_df, on="BuilderRegionKey", how="left")
    
    # Filter to at-risk only
    df = df[df["Projected_Shortfall"] > 0].copy()
    if df.empty:
        st.success("✅ All builders on track! No immediate risks detected.")
        return
    
    # Sort by risk score
    df = df.sort_values("Risk_Score", ascending=False)
    
    # Display columns
    display_cols = [
        "BuilderRegionKey", "Risk_Level", "Projected_Shortfall", "Days_Remaining",
        "Pct_to_Target", "Velocity_LeadsPerDay", "Risk_Score"
    ]
    
    # Add velocity columns if available
    if "pace_2d" in df.columns:
        display_cols.extend(["pace_2d", "pace_7d", "decel_severity"])
    
    available_cols = [c for c in display_cols if c in df.columns]
    display_df = df[available_cols].head(20).copy()
    
    # Rename for display
    rename_map = {
        "BuilderRegionKey": "Builder",
        "Risk_Level": "Risk",
        "Projected_Shortfall": "Gap",
        "Days_Remaining": "Days Left",
        "Pct_to_Target": "% to Target",
        "Velocity_LeadsPerDay": "Pace/Day",
        "Risk_Score": "Risk Score",
        "pace_2d": "2D Pace",
        "pace_7d": "7D Pace",
        "decel_severity": "Velocity Alert"
    }
    display_df = display_df.rename(columns=rename_map)
    
    # Style function
    def style_risk(val):
        if "Critical" in str(val):
            return "background-color: #fee2e2; color: #991b1b; font-weight: bold"
        if "High" in str(val):
            return "background-color: #ffedd5; color: #9a3412; font-weight: bold"
        if "Medium" in str(val):
            return "background-color: #fef9c3; color: #854d0e"
        return ""
    
    def style_decel(val):
        if val == "Severe":
            return "background-color: #f3e8ff; color: #6b21a8; font-weight: bold"
        if val == "Moderate":
            return "background-color: #fae8ff; color: #86198f"
        return ""
    
    # Format dict
    fmt = {
        "Gap": "{:.0f}",
        "Days Left": "{:.0f}",
        "% to Target": "{:.0f}%",
        "Pace/Day": "{:.2f}",
        "Risk Score": "{:.1f}",
        "2D Pace": "{:.2f}",
        "7D Pace": "{:.2f}"
    }
    fmt = {k: v for k, v in fmt.items() if k in display_df.columns}
    
    styled = display_df.style.format(fmt)
    if "Risk" in display_df.columns:
        styled = styled.applymap(style_risk, subset=["Risk"])
    if "Velocity Alert" in display_df.columns:
        styled = styled.applymap(style_decel, subset=["Velocity Alert"])
    
    st.dataframe(styled, use_container_width=True, hide_index=True, height=400)


def render_risk_scatter(shortfall_df: pd.DataFrame):
    """Risk matrix scatter plot."""
    df = shortfall_df[shortfall_df["Projected_Shortfall"] > 0].copy()
    if df.empty:
        return
    
    # Color by risk level
    color_map = {
        "🔴 Critical": "#dc2626",
        "🟠 High": "#f97316",
        "🟡 Medium": "#eab308",
        "🟢 Low": "#22c55e",
        "✅ On Track": "#94a3b8"
    }
    df["color"] = df["Risk_Level"].map(color_map).fillna("#94a3b8")
    
    fig = go.Figure()
    
    for level, color in color_map.items():
        subset = df[df["Risk_Level"] == level]
        if subset.empty:
            continue
        
        fig.add_trace(go.Scatter(
            x=subset["Days_Remaining"],
            y=subset["Projected_Shortfall"],
            mode="markers+text",
            name=level,
            text=subset["BuilderRegionKey"].str[:12],
            textposition="top center",
            marker=dict(
                size=subset["Risk_Score"].clip(10, 50),
                color=color,
                opacity=0.7,
                line=dict(width=1, color="white")
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Gap: %{y:.0f} leads<br>"
                "Days Left: %{x:.0f}<br>"
                "<extra></extra>"
            )
        ))
    
    # Urgency zones
    fig.add_vrect(x0=0, x1=14, fillcolor="red", opacity=0.05, line_width=0)
    fig.add_vrect(x0=14, x1=30, fillcolor="orange", opacity=0.05, line_width=0)
    
    fig.add_annotation(x=7, y=df["Projected_Shortfall"].max() * 0.95,
                      text="CRITICAL ZONE", showarrow=False,
                      font=dict(color="red", size=10))
    
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Days Until Campaign End", gridcolor="#e2e8f0"),
        yaxis=dict(title="Lead Shortfall", gridcolor="#e2e8f0"),
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_optimization_summary(result, budget: float):
    """Render optimization result summary."""
    used_pct = result.total_budget_used / budget * 100 if budget > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="plan-summary">
            <div class="plan-title">Budget Deployed</div>
            <div class="plan-value">${result.total_budget_used:,.0f}</div>
            <div style="font-size:0.85rem; color:#94a3b8">{used_pct:.0f}% of ${budget:,.0f}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="plan-summary">
            <div class="plan-title">Leads Recovered</div>
            <div class="plan-value">{result.total_leads_recovered:,.0f}</div>
            <div style="font-size:0.85rem; color:#94a3b8">via network paths</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="plan-summary">
            <div class="plan-title">Unmet Demand</div>
            <div class="plan-value">{result.unmet_demand:,.0f}</div>
            <div style="font-size:0.85rem; color:#94a3b8">leads still needed</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        eff = result.efficiency_score * 1000 if result.efficiency_score else 0
        st.markdown(f'''
        <div class="plan-summary">
            <div class="plan-title">Efficiency</div>
            <div class="plan-value">{eff:.1f}</div>
            <div style="font-size:0.85rem; color:#94a3b8">leads per $1K</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Warnings
    if result.warnings:
        for warning in result.warnings:
            st.warning(f"⚠️ {warning}")


def render_allocation_table(plan_df: pd.DataFrame):
    """Render the detailed allocation table with full traceability."""
    if plan_df.empty:
        st.info("No allocations generated. Check network connectivity or adjust filters.")
        return
    
    # Format for display
    display = plan_df.copy()
    
    st.dataframe(
        display.style.format({
            "Transfer_Rate": "{:.1%}",
            "Source_CPR": "${:,.0f}",
            "Effective_CPR": "${:,.0f}",
            "Leads_Needed": "{:,.0f}",
            "Leads_Allocated": "{:,.1f}",
            "Investment": "${:,.0f}",
            "Source_Surplus": "{:,.0f}"
        }).background_gradient(subset=["Investment"], cmap="Reds"),
        use_container_width=True,
        hide_index=True,
        height=450
    )


def render_reconciliation_view(plan_df: pd.DataFrame, shortfall_df: pd.DataFrame):
    """Show which surplus builders help which deficit builders."""
    if plan_df.empty:
        return
    
    # Aggregate by source -> target
    st.markdown("### 🔄 Reconciliation: Who Helps Whom")
    
    # Source summary
    source_summary = (
        plan_df.groupby("Source_Builder")
        .agg({
            "Target_Builder": lambda x: ", ".join(x.unique()[:3]),
            "Leads_Allocated": "sum",
            "Investment": "sum",
            "Source_Surplus": "first"
        })
        .reset_index()
        .sort_values("Investment", ascending=False)
    )
    source_summary.columns = ["Source", "Targets Helped", "Total Leads Given", "Total Investment", "Surplus Available"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sources Tapped (Over-serviced Builders)**")
        st.dataframe(
            source_summary.style.format({
                "Total Leads Given": "{:.0f}",
                "Total Investment": "${:,.0f}",
                "Surplus Available": "{:.0f}"
            }),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        # Target summary
        target_summary = (
            plan_df.groupby("Target_Builder")
            .agg({
                "Source_Builder": lambda x: ", ".join(x.unique()[:3]),
                "Leads_Allocated": "sum",
                "Leads_Needed": "first",
                "Investment": "sum"
            })
            .reset_index()
            .sort_values("Investment", ascending=False)
        )
        target_summary["Gap_Closed_%"] = target_summary["Leads_Allocated"] / target_summary["Leads_Needed"] * 100
        target_summary.columns = ["Target", "Sources Used", "Leads Received", "Original Gap", "Investment", "Gap Closed %"]
        
        st.markdown("**Targets Helped (Under-serviced Builders)**")
        st.dataframe(
            target_summary.style.format({
                "Leads Received": "{:.0f}",
                "Original Gap": "{:.0f}",
                "Investment": "${:,.0f}",
                "Gap Closed %": "{:.0f}%"
            }),
            use_container_width=True,
            hide_index=True
        )


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Header
    st.markdown('''
    <div class="command-header">
        <div class="command-title">🎯 Network Command Center</div>
        <div class="command-sub">Risk detection • Budget-constrained optimization • Full traceability</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Load data
    events_full = load_data()
    
    if events_full is None:
        st.warning("⚠️ Please upload events data on the Home page first.")
        st.page_link("app.py", label="← Go to Home", icon="🏠")
        return
    
    # ==========================================================================
    # SIDEBAR CONTROLS
    # ==========================================================================
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Date range
        st.subheader("📅 Analysis Period")
        date_col, available_months = get_date_info(events_full)
        
        use_all = st.checkbox("Use All Time", value=False)
        
        if use_all or len(available_months) == 0:
            events_selected = events_full
            period_days = 90
        else:
            start_month, end_month = st.select_slider(
                "Date Range",
                options=available_months,
                value=(available_months[0], available_months[-1]),
                format_func=lambda x: x.strftime("%b %Y")
            )
            
            end_date = end_month + pd.offsets.MonthEnd(1)
            mask = (
                (pd.to_datetime(events_full[date_col]) >= start_month) &
                (pd.to_datetime(events_full[date_col]) <= end_date)
            )
            events_selected = events_full.loc[mask].copy()
            period_days = max((end_date - start_month).days, 1)
        
        st.caption(f"📊 {len(events_selected):,} events in period")
        
        st.divider()
        
        # Budget
        st.subheader("💰 Budget Constraint")
        budget = st.number_input(
            "Total Media Budget ($)",
            min_value=1000,
            max_value=1000000,
            value=50000,
            step=5000,
            help="Maximum spend for optimization"
        )
        
        st.divider()
        
        # Exclusions
        st.subheader("🚫 Source Exclusions")
        all_sources = sorted(
            events_selected["MediaPayer_BuilderRegionKey"].dropna().unique().tolist()
        )
        excluded_sources = st.multiselect(
            "Exclude from optimization",
            options=all_sources,
            default=[],
            help="These builders won't be used as sources"
        )
        
        st.divider()
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            min_transfer_rate = st.slider(
                "Min Transfer Rate",
                min_value=0.01,
                max_value=0.20,
                value=0.02,
                step=0.01,
                help="Minimum connection strength to consider"
            )
    
    # ==========================================================================
    # CORE CALCULATIONS
    # ==========================================================================
    with st.spinner("Analyzing network..."):
        # 1. Shortfall analysis
        shortfall_df = calculate_shortfalls_v2(
            events_df=events_selected,
            total_events_df=events_full,
            period_days=period_days
        )
        
        # 2. Velocity metrics
        velocity_df = calculate_velocity_metrics(
            events_selected,
            builder_col="Dest_BuilderRegionKey",
            date_col=date_col
        )
        
        # 3. Network leverage (with exclusions)
        leverage_df = analyze_network_leverage_v2(
            events_selected,
            excluded_sources=excluded_sources
        )
        
        # 4. Budget-constrained optimization
        optimization_result = run_budget_constrained_optimization(
            shortfall_df=shortfall_df,
            leverage_df=leverage_df,
            budget=budget,
            excluded_sources=excluded_sources,
            min_transfer_rate=min_transfer_rate
        )
        
        plan_df = optimization_result_to_dataframe(optimization_result)
    
    # Identify critical builders
    critical_builders = shortfall_df[
        shortfall_df["Risk_Level"].str.contains("Critical", na=False)
    ]
    decel_builders = velocity_df[velocity_df["is_decelerating"]] if not velocity_df.empty else pd.DataFrame()
    
    # ==========================================================================
    # RENDER: CRITICAL ALERTS (Top of Page)
    # ==========================================================================
    has_alerts = render_critical_alerts(critical_builders, decel_builders)
    
    # ==========================================================================
    # RENDER: KPI ROW
    # ==========================================================================
    render_kpi_row(shortfall_df, velocity_df, budget)
    
    # ==========================================================================
    # TABS
    # ==========================================================================
    tab_monitor, tab_optimize, tab_trace, tab_explore = st.tabs([
        "🔍 Risk Monitor",
        "🚀 Optimization Plan",
        "📋 Full Traceability",
        "🕸️ Network Explorer"
    ])
    
    # --------------------------------------------------------------------------
    # TAB 1: RISK MONITOR
    # --------------------------------------------------------------------------
    with tab_monitor:
        st.markdown('<div class="section-header">🚨 Critical Builder Monitor</div>', unsafe_allow_html=True)
        st.caption("Builders sorted by Risk Score (Shortfall × Urgency). Velocity alerts flag recent performance crashes.")
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            render_critical_builder_table(shortfall_df, velocity_df)
        
        with col2:
            st.markdown("**Risk Distribution**")
            render_risk_scatter(shortfall_df)
            
            # Velocity breakdown
            if not velocity_df.empty:
                st.markdown("**Velocity Alerts Breakdown**")
                severity_counts = velocity_df["decel_severity"].value_counts()
                fig = px.pie(
                    values=severity_counts.values,
                    names=severity_counts.index,
                    color=severity_counts.index,
                    color_discrete_map={
                        "Severe": "#7c3aed",
                        "Moderate": "#a855f7",
                        "Mild": "#c4b5fd",
                        "None": "#e2e8f0"
                    },
                    hole=0.4
                )
                fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
    
    # --------------------------------------------------------------------------
    # TAB 2: OPTIMIZATION PLAN
    # --------------------------------------------------------------------------
    with tab_optimize:
        st.markdown('<div class="section-header">🚀 Budget-Constrained Optimization</div>', unsafe_allow_html=True)
        st.caption(f"Greedy allocation prioritizing lowest eCPR paths. Budget: ${budget:,.0f}")
        
        # Summary metrics
        render_optimization_summary(optimization_result, budget)
        
        st.markdown("---")
        
        # Allocation table
        st.markdown("### 📊 Recommended Allocations")
        st.caption("Sorted by priority. Each row shows the most efficient path to close a builder's gap.")
        render_allocation_table(plan_df)
        
        # Download
        if not plan_df.empty:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.download_button(
                    "📥 Export Plan (CSV)",
                    plan_df.to_csv(index=False),
                    "optimization_plan.csv",
                    "text/csv"
                )
            with col2:
                st.download_button(
                    "📥 Export Plan (Excel)",
                    export_to_excel(plan_df, "optimization_plan.xlsx"),
                    "optimization_plan.xlsx"
                )
    
    # --------------------------------------------------------------------------
    # TAB 3: FULL TRACEABILITY
    # --------------------------------------------------------------------------
    with tab_trace:
        st.markdown('<div class="section-header">📋 Traceability & Reasoning</div>', unsafe_allow_html=True)
        st.caption("Detailed view of why each allocation was made.")
        
        render_reconciliation_view(plan_df, shortfall_df)
        
        st.markdown("---")
        
        # Detailed reasoning table
        if not plan_df.empty:
            st.markdown("### 📝 Allocation Reasoning")
            
            reasoning_df = plan_df[[
                "Priority", "Target_Builder", "Source_Builder", 
                "Transfer_Rate", "Effective_CPR", "Investment", "Reasoning"
            ]].copy()
            
            st.dataframe(
                reasoning_df.style.format({
                    "Transfer_Rate": "{:.1%}",
                    "Effective_CPR": "${:,.0f}",
                    "Investment": "${:,.0f}"
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Reasoning": st.column_config.TextColumn(width="large")
                }
            )
    
    # --------------------------------------------------------------------------
    # TAB 4: NETWORK EXPLORER
    # --------------------------------------------------------------------------
    with tab_explore:
        st.markdown('<div class="section-header">🕸️ Network Topology</div>', unsafe_allow_html=True)
        
        if leverage_df.empty:
            st.warning("No network connections found in the selected period.")
        else:
            # Network stats
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Unique Sources", leverage_df["MediaPayer_BuilderRegionKey"].nunique())
            col2.metric("Unique Targets", leverage_df["Dest_BuilderRegionKey"].nunique())
            col3.metric("Total Connections", len(leverage_df))
            col4.metric("Avg Transfer Rate", f"{leverage_df['Transfer_Rate'].mean():.1%}")
            
            # Top connections
            st.markdown("### 🔗 Strongest Connections")
            top_connections = leverage_df.nlargest(15, "Referrals_to_Target")
            
            fig = go.Figure(go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=(
                        top_connections["MediaPayer_BuilderRegionKey"].tolist() +
                        top_connections["Dest_BuilderRegionKey"].tolist()
                    )
                ),
                link=dict(
                    source=list(range(len(top_connections))),
                    target=list(range(len(top_connections), 2 * len(top_connections))),
                    value=top_connections["Referrals_to_Target"].tolist(),
                    color="rgba(99, 102, 241, 0.3)"
                )
            ))
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            # Raw data
            with st.expander("📊 View Raw Network Data"):
                st.dataframe(
                    leverage_df.style.format({
                        "Transfer_Rate": "{:.1%}",
                        "CPR_base": "${:,.0f}",
                        "eCPR": "${:,.0f}"
                    }),
                    use_container_width=True,
                    hide_index=True
                )


if __name__ == "__main__":
    main()