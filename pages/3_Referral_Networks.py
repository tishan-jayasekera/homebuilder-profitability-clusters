"""
Referral Network Command Center v2.1
- FIXED: Restored clustering functionality
- FIXED: Velocity calculation now covers all builders
"""
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
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
from src.referral_clusters import run_referral_clustering, compute_network_metrics
from src.network_optimization import (
    calculate_shortfalls_v2,
    calculate_velocity_metrics,
    analyze_network_leverage_v2,
    run_budget_constrained_optimization,
    optimization_result_to_dataframe,
    generate_investment_strategies
)
from src.utils import fmt_currency

st.set_page_config(page_title="Network Command Center", page_icon="🎯", layout="wide")

# =============================================================================
# STYLES
# =============================================================================
st.markdown("""
<style>
.command-header {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    color: white; padding: 1.2rem; border-radius: 12px; margin-bottom: 1rem;
}
.command-title { font-size: 1.6rem; font-weight: 700; margin: 0; }
.command-sub { font-size: 0.9rem; color: #94a3b8; margin-top: 0.25rem; }

.alert-critical {
    background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
    color: white; padding: 0.9rem 1rem; border-radius: 10px; margin-bottom: 0.6rem;
}
.alert-warning {
    background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
    color: white; padding: 0.9rem 1rem; border-radius: 10px; margin-bottom: 0.6rem;
}
.alert-title { font-weight: 600; font-size: 0.95rem; }
.alert-body { font-size: 0.85rem; opacity: 0.95; margin-top: 0.2rem; }

.metric-row { display: grid; grid-template-columns: repeat(6, 1fr); gap: 0.6rem; margin: 0.8rem 0; }
.metric-box {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 0.7rem; text-align: center;
}
.metric-box.critical { border-left: 4px solid #dc2626; }
.metric-box.warning { border-left: 4px solid #f97316; }
.metric-box.success { border-left: 4px solid #16a34a; }
.metric-label { font-size: 0.68rem; text-transform: uppercase; color: #64748b; letter-spacing: 0.04em; }
.metric-value { font-size: 1.25rem; font-weight: 700; color: #0f172a; margin-top: 0.15rem; }
.metric-delta { font-size: 0.75rem; margin-top: 0.15rem; color: #64748b; }

.section-header {
    font-size: 1rem; font-weight: 600; color: #1e293b;
    margin: 1.2rem 0 0.4rem 0; padding-bottom: 0.4rem; border-bottom: 2px solid #e2e8f0;
}

.plan-summary {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;
}
.plan-title { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; }
.plan-value { font-size: 1.4rem; font-weight: 700; margin-top: 0.2rem; }

.velocity-badge {
    display: inline-block; padding: 2px 6px; border-radius: 4px;
    font-size: 0.7rem; font-weight: 500;
}
.velocity-severe { background: #7c3aed; color: white; }
.velocity-moderate { background: #a855f7; color: white; }
.velocity-mild { background: #c4b5fd; color: #1e1b4b; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data():
    if "events_file" not in st.session_state:
        return None
    events = load_events(st.session_state["events_file"])
    return normalize_events(events) if events is not None else None


def get_date_info(events):
    date_col = "lead_date" if "lead_date" in events.columns else "RefDate"
    if date_col not in events.columns:
        return None, []
    dates = pd.to_datetime(events[date_col], errors="coerce").dropna()
    if dates.empty:
        return date_col, []
    months = pd.date_range(start=dates.min().replace(day=1), end=dates.max().replace(day=1), freq="MS")
    return date_col, months


# =============================================================================
# RENDER FUNCTIONS
# =============================================================================
def render_critical_alerts(critical_df, decel_df):
    alerts = []
    if not critical_df.empty:
        top = critical_df.head(3)["BuilderRegionKey"].tolist()
        total_gap = critical_df["Projected_Shortfall"].sum()
        alerts.append({
            "type": "critical",
            "title": f"🚨 {len(critical_df)} Builders at Critical Risk",
            "body": f"Top: {', '.join(top)}. Total gap: {total_gap:.0f} leads."
        })
    
    severe = decel_df[decel_df["decel_severity"] == "Severe"] if not decel_df.empty else pd.DataFrame()
    if not severe.empty:
        names = severe.head(3)["BuilderRegionKey"].tolist()
        alerts.append({
            "type": "warning",
            "title": f"📉 {len(severe)} Builders with Severe Velocity Drop",
            "body": f"Affected: {', '.join(names)}. Recent pace crashed vs 14-day avg."
        })
    
    for a in alerts:
        cls = "alert-critical" if a["type"] == "critical" else "alert-warning"
        st.markdown(f'<div class="{cls}"><div class="alert-title">{a["title"]}</div><div class="alert-body">{a["body"]}</div></div>', unsafe_allow_html=True)
    
    return len(alerts) > 0


def render_kpi_row(shortfall_df, velocity_df, budget, cluster_count):
    total = len(shortfall_df)
    at_risk = len(shortfall_df[shortfall_df["Projected_Shortfall"] > 0])
    total_gap = shortfall_df["Projected_Shortfall"].sum()
    total_surplus = shortfall_df["Projected_Surplus"].sum()
    
    # Velocity coverage
    if not velocity_df.empty:
        has_activity = velocity_df[velocity_df["decel_severity"] != "No Activity"]
        decel_count = len(velocity_df[velocity_df["is_decelerating"]])
        vel_coverage = len(has_activity) / len(velocity_df) * 100 if len(velocity_df) > 0 else 0
    else:
        decel_count = 0
        vel_coverage = 0
    
    risk_pct = at_risk / total * 100 if total > 0 else 0
    
    st.markdown(f'''
    <div class="metric-row">
        <div class="metric-box {'critical' if risk_pct > 30 else 'warning' if risk_pct > 15 else ''}">
            <div class="metric-label">At Risk</div>
            <div class="metric-value">{at_risk}/{total}</div>
            <div class="metric-delta">{risk_pct:.0f}% of builders</div>
        </div>
        <div class="metric-box critical">
            <div class="metric-label">Total Gap</div>
            <div class="metric-value">{total_gap:,.0f}</div>
            <div class="metric-delta">leads needed</div>
        </div>
        <div class="metric-box success">
            <div class="metric-label">Surplus</div>
            <div class="metric-value">{total_surplus:,.0f}</div>
            <div class="metric-delta">excess leads</div>
        </div>
        <div class="metric-box {'warning' if decel_count > 3 else ''}">
            <div class="metric-label">Decelerating</div>
            <div class="metric-value">{decel_count}</div>
            <div class="metric-delta">{vel_coverage:.0f}% w/ activity</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Clusters</div>
            <div class="metric-value">{cluster_count}</div>
            <div class="metric-delta">ecosystems</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Budget</div>
            <div class="metric-value">${budget/1000:.0f}K</div>
            <div class="metric-delta">available</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)


def render_risk_table(shortfall_df, velocity_df):
    df = shortfall_df.copy()
    
    # Merge velocity - use left join to keep all builders
    if not velocity_df.empty:
        df = df.merge(velocity_df, on="BuilderRegionKey", how="left")
        # Fill NaN velocity data
        df["pace_2d"] = df["pace_2d"].fillna(0)
        df["pace_7d"] = df["pace_7d"].fillna(0)
        df["pace_14d"] = df["pace_14d"].fillna(0)
        df["decel_severity"] = df["decel_severity"].fillna("No Data")
        df["is_decelerating"] = df["is_decelerating"].fillna(False)
    
    # Filter and sort
    df = df[df["Projected_Shortfall"] > 0].sort_values("Risk_Score", ascending=False)
    
    if df.empty:
        st.success("✅ All builders on track!")
        return
    
    cols = ["BuilderRegionKey", "Risk_Level", "Projected_Shortfall", "Days_Remaining", 
            "Pct_to_Target", "Velocity_LeadsPerDay", "Risk_Score"]
    if "pace_7d" in df.columns:
        cols.extend(["pace_7d", "pace_14d", "decel_severity"])
    
    display = df[[c for c in cols if c in df.columns]].head(25).copy()
    display.columns = [c.replace("_", " ").replace("BuilderRegionKey", "Builder") for c in display.columns]
    
    fmt = {"Projected Shortfall": "{:.0f}", "Days Remaining": "{:.0f}", "Pct to Target": "{:.0f}%",
           "Velocity LeadsPerDay": "{:.2f}", "Risk Score": "{:.1f}", "pace 7d": "{:.2f}", "pace 14d": "{:.2f}"}
    fmt = {k: v for k, v in fmt.items() if k in display.columns}
    
    st.dataframe(display.style.format(fmt), use_container_width=True, hide_index=True, height=420)


def render_risk_scatter(shortfall_df):
    df = shortfall_df[shortfall_df["Projected_Shortfall"] > 0].copy()
    if df.empty:
        return
    
    color_map = {"🔴 Critical": "#dc2626", "🟠 High": "#f97316", "🟡 Medium": "#eab308", 
                 "🟢 Low": "#22c55e", "✅ On Track": "#94a3b8"}
    
    fig = go.Figure()
    for level, color in color_map.items():
        sub = df[df["Risk_Level"] == level]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["Days_Remaining"], y=sub["Projected_Shortfall"],
            mode="markers", name=level,
            marker=dict(size=sub["Risk_Score"].clip(8, 40), color=color, opacity=0.7),
            text=sub["BuilderRegionKey"],
            hovertemplate="<b>%{text}</b><br>Gap: %{y:.0f}<br>Days: %{x:.0f}<extra></extra>"
        ))
    
    fig.add_vrect(x0=0, x1=14, fillcolor="red", opacity=0.05, line_width=0)
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(title="Days Remaining"), yaxis=dict(title="Lead Gap"),
                      plot_bgcolor="white", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)


def render_optimization_summary(result, budget):
    used_pct = result.total_budget_used / budget * 100 if budget > 0 else 0
    eff = result.efficiency_score * 1000 if result.efficiency_score else 0
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="plan-summary"><div class="plan-title">Budget Used</div><div class="plan-value">${result.total_budget_used:,.0f}</div><div style="font-size:0.8rem;color:#94a3b8">{used_pct:.0f}%</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="plan-summary"><div class="plan-title">Leads Recovered</div><div class="plan-value">{result.total_leads_recovered:,.0f}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="plan-summary"><div class="plan-title">Unmet Demand</div><div class="plan-value">{result.unmet_demand:,.0f}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="plan-summary"><div class="plan-title">Efficiency</div><div class="plan-value">{eff:.1f}</div><div style="font-size:0.8rem;color:#94a3b8">leads/$1K</div></div>', unsafe_allow_html=True)
    
    for w in result.warnings:
        st.warning(f"⚠️ {w}")


def render_cluster_graph(G, builder_master, cluster_id):
    """Render cluster network visualization."""
    if G is None or len(G.nodes) == 0:
        st.warning("No graph data available")
        return
    
    # Filter to selected cluster
    cluster_nodes = builder_master[builder_master["ClusterId"] == cluster_id]["BuilderRegionKey"].tolist()
    subG = G.subgraph(cluster_nodes)
    
    if len(subG.nodes) == 0:
        st.info("No nodes in this cluster")
        return
    
    # Layout
    pos = nx.spring_layout(subG, weight="weight", seed=42, k=2)
    
    # Edges
    edge_x, edge_y = [], []
    for u, v in subG.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.8, color="#94a3b8"),
                            hoverinfo="none", mode="lines")
    
    # Nodes
    node_x = [pos[n][0] for n in subG.nodes()]
    node_y = [pos[n][1] for n in subG.nodes()]
    
    # Color by role
    role_map = builder_master.set_index("BuilderRegionKey")["Role"].to_dict()
    colors = []
    for n in subG.nodes():
        role = role_map.get(n, "mixed")
        if role == "pure_sender":
            colors.append("#22c55e")
        elif role == "pure_receiver":
            colors.append("#3b82f6")
        else:
            colors.append("#f59e0b")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=[n[:15] for n in subG.nodes()], textposition="top center",
        marker=dict(size=12, color=colors, line=dict(width=1, color="white")),
        hovertext=list(subG.nodes()), hoverinfo="text"
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False, height=450,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# MAIN
# =============================================================================
def main():
    st.markdown('<div class="command-header"><div class="command-title">🎯 Network Command Center</div><div class="command-sub">Risk detection • Clustering • Budget optimization</div></div>', unsafe_allow_html=True)
    
    events_full = load_data()
    if events_full is None:
        st.warning("⚠️ Upload events data on Home page first.")
        st.page_link("app.py", label="← Go to Home", icon="🏠")
        return
    
    # ==========================================================================
    # SIDEBAR
    # ==========================================================================
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Date range
        st.subheader("📅 Period")
        date_col, available_months = get_date_info(events_full)
        use_all = st.checkbox("Use All Time", value=False)
        
        if use_all or len(available_months) == 0:
            events_selected = events_full
            period_days = 90
        else:
            start_month, end_month = st.select_slider(
                "Date Range", options=available_months,
                value=(available_months[0], available_months[-1]),
                format_func=lambda x: x.strftime("%b %Y")
            )
            end_date = end_month + pd.offsets.MonthEnd(1)
            mask = (pd.to_datetime(events_full[date_col]) >= start_month) & (pd.to_datetime(events_full[date_col]) <= end_date)
            events_selected = events_full.loc[mask].copy()
            period_days = max((end_date - start_month).days, 1)
        
        st.caption(f"📊 {len(events_selected):,} events")
        st.divider()
        
        # Budget
        st.subheader("💰 Budget")
        budget = st.number_input("Media Budget ($)", 1000, 1000000, 50000, 5000)
        
        st.divider()
        
        # Exclusions
        st.subheader("🚫 Exclusions")
        all_sources = sorted(events_selected["MediaPayer_BuilderRegionKey"].dropna().unique().tolist())
        excluded = st.multiselect("Exclude sources", all_sources, [])
        
        st.divider()
        
        # Clustering params
        st.subheader("🕸️ Clustering")
        resolution = st.slider("Resolution", 0.5, 2.5, 1.5, 0.1)
        max_clusters = st.slider("Max Clusters", 5, 25, 15)
        
        with st.expander("Advanced"):
            min_tr = st.slider("Min Transfer Rate", 0.01, 0.20, 0.02, 0.01)
    
    # ==========================================================================
    # CALCULATIONS
    # ==========================================================================
    with st.spinner("Analyzing..."):
        # Get all builders for velocity coverage
        all_builders = events_full["Dest_BuilderRegionKey"].dropna().unique().tolist()
        
        # 1. Shortfall
        shortfall_df = calculate_shortfalls_v2(
            events_df=events_selected,
            total_events_df=events_full,
            period_days=period_days
        )
        
        # 2. Velocity - pass all builders to ensure coverage
        velocity_df = calculate_velocity_metrics(
            events_selected,
            builder_col="Dest_BuilderRegionKey",
            date_col=date_col,
            all_builders=all_builders
        )
        
        # 3. Clustering
        cluster_results = run_referral_clustering(
            events_selected,
            resolution=resolution,
            target_max_clusters=max_clusters
        )
        builder_master = cluster_results.get("builder_master", pd.DataFrame())
        cluster_summary = cluster_results.get("cluster_summary", pd.DataFrame())
        G = cluster_results.get("graph", nx.Graph())
        edges_clean = cluster_results.get("edges_clean", pd.DataFrame())
        
        # 4. Leverage
        leverage_df = analyze_network_leverage_v2(events_selected, excluded)
        
        # 5. Optimization
        opt_result = run_budget_constrained_optimization(
            shortfall_df, leverage_df, budget, excluded, min_tr
        )
        plan_df = optimization_result_to_dataframe(opt_result)
    
    # Critical builders
    critical_df = shortfall_df[shortfall_df["Risk_Level"].str.contains("Critical", na=False)]
    decel_df = velocity_df[velocity_df["is_decelerating"]] if not velocity_df.empty else pd.DataFrame()
    
    cluster_count = len(cluster_summary) if not cluster_summary.empty else 0
    
    # ==========================================================================
    # ALERTS & KPIS
    # ==========================================================================
    render_critical_alerts(critical_df, decel_df)
    render_kpi_row(shortfall_df, velocity_df, budget, cluster_count)
    
    # ==========================================================================
    # TABS
    # ==========================================================================
    tab_risk, tab_opt, tab_cluster, tab_detail = st.tabs([
        "🔍 Risk Monitor", "🚀 Optimization", "🕸️ Clusters", "📋 Builder Detail"
    ])
    
    # --------------------------------------------------------------------------
    # TAB: RISK
    # --------------------------------------------------------------------------
    with tab_risk:
        st.markdown('<div class="section-header">🚨 Critical Builder Monitor</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns([1.4, 1])
        with c1:
            render_risk_table(shortfall_df, velocity_df)
        with c2:
            st.markdown("**Risk Distribution**")
            render_risk_scatter(shortfall_df)
            
            # Velocity breakdown
            if not velocity_df.empty:
                st.markdown("**Velocity Status**")
                sev_counts = velocity_df["decel_severity"].value_counts()
                fig = px.pie(values=sev_counts.values, names=sev_counts.index, hole=0.4,
                            color=sev_counts.index,
                            color_discrete_map={"Severe": "#7c3aed", "Moderate": "#a855f7", 
                                               "Mild": "#c4b5fd", "None": "#d1d5db", "No Activity": "#f3f4f6", "No Data": "#e5e7eb"})
                fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
    
    # --------------------------------------------------------------------------
    # TAB: OPTIMIZATION
    # --------------------------------------------------------------------------
    with tab_opt:
        st.markdown('<div class="section-header">🚀 Budget-Constrained Optimization</div>', unsafe_allow_html=True)
        
        render_optimization_summary(opt_result, budget)
        
        if not plan_df.empty:
            st.markdown("### 📊 Allocation Plan")
            st.dataframe(
                plan_df.style.format({
                    "Transfer_Rate": "{:.1%}", "Source_CPR": "${:,.0f}", "Effective_CPR": "${:,.0f}",
                    "Leads_Needed": "{:,.0f}", "Leads_Allocated": "{:,.1f}", "Investment": "${:,.0f}",
                    "Source_Surplus": "{:,.0f}"
                }).background_gradient(subset=["Investment"], cmap="Reds"),
                use_container_width=True, hide_index=True, height=400
            )
            
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("📥 Export CSV", plan_df.to_csv(index=False), "plan.csv", "text/csv")
            with c2:
                st.download_button("📥 Export Excel", export_to_excel(plan_df, "plan.xlsx"), "plan.xlsx")
        else:
            st.info("No allocations generated. Check network paths or adjust filters.")
    
    # --------------------------------------------------------------------------
    # TAB: CLUSTERS
    # --------------------------------------------------------------------------
    with tab_cluster:
        st.markdown('<div class="section-header">🕸️ Referral Ecosystem Clusters</div>', unsafe_allow_html=True)
        
        if cluster_summary.empty:
            st.warning("No clusters found. Check data or adjust parameters.")
        else:
            # Cluster selector
            cluster_opts = {int(r.ClusterId): f"Cluster {int(r.ClusterId)} ({int(r.N_builders)} builders)" 
                          for r in cluster_summary.itertuples()}
            sel_cluster = st.selectbox("Select Cluster", list(cluster_opts.keys()), 
                                       format_func=lambda x: cluster_opts[x])
            
            c1, c2 = st.columns([1.5, 1])
            
            with c1:
                st.markdown("**Network Topology**")
                render_cluster_graph(G, builder_master, sel_cluster)
                st.caption("🟢 Sender | 🔵 Receiver | 🟠 Mixed")
            
            with c2:
                # Cluster summary
                st.markdown("**Cluster Summary**")
                st.dataframe(cluster_summary, use_container_width=True, hide_index=True)
                
                # Builders in cluster
                st.markdown("**Builders in Selected Cluster**")
                cluster_builders = builder_master[builder_master["ClusterId"] == sel_cluster]
                st.dataframe(
                    cluster_builders[["BuilderRegionKey", "Role", "Referrals_in", "Referrals_out"]],
                    use_container_width=True, hide_index=True, height=200
                )
    
    # --------------------------------------------------------------------------
    # TAB: BUILDER DETAIL
    # --------------------------------------------------------------------------
    with tab_detail:
        st.markdown('<div class="section-header">📋 Builder Deep Dive</div>', unsafe_allow_html=True)
        
        at_risk = shortfall_df[shortfall_df["Projected_Shortfall"] > 0]["BuilderRegionKey"].tolist()
        all_b = sorted(shortfall_df["BuilderRegionKey"].dropna().unique())
        sorted_opts = at_risk + [b for b in all_b if b not in at_risk]
        
        sel_builder = st.selectbox("Select Builder", sorted_opts)
        
        if sel_builder:
            row = shortfall_df[shortfall_df["BuilderRegionKey"] == sel_builder]
            if not row.empty:
                r = row.iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Target", int(r["LeadTarget"]))
                c2.metric("Projected", int(r["Projected_Total"]), delta=int(r["Net_Gap"]))
                c3.metric("Days Left", int(r["Days_Remaining"]))
                c4.metric("Risk Score", f"{r['Risk_Score']:.1f}")
                
                # Velocity for this builder
                if not velocity_df.empty:
                    vel = velocity_df[velocity_df["BuilderRegionKey"] == sel_builder]
                    if not vel.empty:
                        v = vel.iloc[0]
                        st.markdown(f"**Velocity:** 2D: {v['pace_2d']:.2f} | 7D: {v['pace_7d']:.2f} | 14D: {v['pace_14d']:.2f} | Status: `{v['decel_severity']}`")
            
            # Strategies
            strats = generate_investment_strategies(sel_builder, shortfall_df, leverage_df, events_selected)
            if not strats.empty:
                st.markdown("### Inbound Pathways")
                st.dataframe(strats.style.format({
                    "Transfer_Rate": "{:.1%}", "Base_CPR": "${:,.0f}",
                    "Effective_CPR": "${:,.0f}", "Investment_Required": "${:,.0f}"
                }), use_container_width=True, hide_index=True)
            else:
                st.info("No inbound paths found for this builder.")


if __name__ == "__main__":
    main()