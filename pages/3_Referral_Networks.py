"""
Referral Networks Dashboard - Streamlit Page
Filename: pages/3_Referral_Networks.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go

import sys
from pathlib import Path

root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data_loader import load_events, export_to_excel
from src.normalization import normalize_events
from src.builder_pnl import build_builder_pnl
from src.referral_clusters import run_referral_clustering, compute_network_metrics
from src.utils import fmt_currency, fmt_roas
from src.network_optimization import (
    calculate_shortfalls, 
    analyze_network_leverage, 
    generate_investment_strategies,
    compute_effective_network_cpr
)

st.set_page_config(page_title="Referral Networks", page_icon="🔗", layout="wide")

st.title("🔗 Referral Network Explorer")
st.markdown("Discover referral ecosystems and media efficiency pathways.")


def run_clustering_for_period(events_df, resolution, target_clusters, period_key):
    """Run clustering - period_key ensures recalculation when month changes."""
    return run_referral_clustering(events_df, resolution=resolution, target_max_clusters=target_clusters)


def load_data():
    if 'events_file' not in st.session_state:
        return None
    events = load_events(st.session_state['events_file'])
    if events is None:
        return None
    return normalize_events(events)


def get_available_months(events):
    """Extract available months from events data."""
    months = []
    date_col = None
    
    if "lead_date" in events.columns:
        date_col = "lead_date"
    elif "RefDate" in events.columns:
        date_col = "RefDate"
    
    if date_col:
        dates = pd.to_datetime(events[date_col], errors="coerce")
        month_starts = dates.dt.to_period("M").dt.start_time.dropna().unique()
        months = sorted(month_starts)
    
    return months, date_col


def filter_events_by_month(events, selected_month, date_col):
    """Filter events to a specific month."""
    if selected_month is None or date_col is None:
        return events
    
    dates = pd.to_datetime(events[date_col], errors="coerce")
    month_starts = dates.dt.to_period("M").dt.start_time
    
    return events[month_starts == selected_month].copy()


def compute_cpr_recommendations(edges_df, builders_df):
    """
    Compute Cost Per Referral (CPR) for each builder and generate recommendations.
    CPR = MediaCost / Referrals_out (cost to generate one referral)
    """
    if builders_df.empty or edges_df.empty:
        return pd.DataFrame()
    
    # Calculate referrals sent by each builder
    referrals_out = edges_df.groupby("Origin_builder")["Referrals"].sum().reset_index()
    referrals_out.columns = ["BuilderRegionKey", "Total_Referrals_Sent"]
    
    # Merge with builder data
    recs = builders_df.merge(referrals_out, on="BuilderRegionKey", how="left")
    recs["Total_Referrals_Sent"] = recs["Total_Referrals_Sent"].fillna(0)
    
    # Calculate CPR (Cost Per Referral)
    recs["CPR"] = np.where(
        recs["Total_Referrals_Sent"] > 0,
        recs["MediaCost"] / recs["Total_Referrals_Sent"],
        np.nan
    )
    
    # Only include builders who send referrals and have media cost
    recs = recs[(recs["Total_Referrals_Sent"] > 0) & (recs["MediaCost"] > 0)].copy()
    
    # Sort by CPR (lowest = most efficient)
    recs = recs.sort_values("CPR", ascending=True)
    
    return recs


def render_investment_sankey(strategy_row, focus_builder):
    """
    Visualize specific investment flow: Source -> Focus Builder + Others (Spillover)
    """
    source = strategy_row['Source_Builder']
    target = focus_builder
    
    # Calculate flows
    total_leads = strategy_row['Total_Leads_Generated']
    leads_to_target = total_leads * strategy_row['Transfer_Rate']
    leads_to_others = strategy_row['Excess_Leads']
    
    # Nodes: 0=Source, 1=Target, 2=Others
    labels = [f"{source}<br>(Source)", f"{target}<br>(Target)", "Other Builders<br>(Spillover)"]
    colors = ["#1E88E5", "#43A047", "#FB8C00"] # Blue, Green, Orange
    
    # Links: Source->Target, Source->Others
    sources = [0, 0]
    targets = [1, 2]
    values = [leads_to_target, leads_to_others]
    
    # Custom hover template
    customdata = [
        f"Target Flow<br>Leads: {int(leads_to_target)}<br>Effective CPR: {fmt_currency(strategy_row['Effective_CPR'])}",
        f"Spillover Flow<br>Excess Leads: {int(leads_to_others)}<br>{strategy_row['Spillover_Impact']}"
    ]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=labels, color=colors
        ),
        link=dict(
            source=sources, target=targets, value=values,
            color=['rgba(67, 160, 71, 0.4)', 'rgba(251, 140, 0, 0.4)'],
            customdata=customdata,
            hovertemplate='%{customdata}<extra></extra>'
        )
    )])
    
    fig.update_layout(title_text=f"Investment Impact: ${strategy_row['Investment_Required']:,.0f}", height=250, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def main():
    events_full = load_data()
    
    if events_full is None:
        st.warning("⚠️ Please upload events data on the Home page first.")
        st.page_link("app.py", label="← Go to Home", icon="🏠")
        return
    
    # Get available months
    available_months, date_col = get_available_months(events_full)
    
    # Sidebar
    with st.sidebar:
        st.header("📅 Time Period")
        if available_months:
            month_options = ["All Time"] + [m.strftime("%Y-%m") for m in available_months]
            selected_month_str = st.selectbox("Select Month", options=month_options, index=0)
            
            if selected_month_str == "All Time":
                selected_month = None
                events = events_full.copy()
                period_key = "all_time"
            else:
                selected_month = pd.Timestamp(selected_month_str + "-01")
                events = filter_events_by_month(events_full, selected_month, date_col)
                period_key = selected_month_str
            
            st.metric("Events in Period", f"{len(events):,}")
        else:
            events = events_full.copy()
            selected_month = None
            period_key = "all_time"
        
        st.divider()
        st.header("🚀 Optimization Mode")
        opt_mode = st.radio(
            "Recommendation Engine",
            ["Standard (Lowest CPR)", "Advanced (Network Investment)"],
            help="Standard: Minimizes generic cost per referral.\nAdvanced: Calculates effective cost to close specific lead gaps."
        )
        
        st.divider()
        st.header("🎛️ Clustering")
        resolution = st.slider("Resolution", 0.5, 2.5, 1.5, 0.1)
        target_clusters = st.slider("Max Clusters", 3, 25, 15, 1)
        
        st.divider()
        st.subheader("🎨 Graph")
        show_labels = st.checkbox("Show Labels", False)
        edge_style = st.selectbox("Edge Style", ["Curved Arrows", "Straight Lines", "Curved Lines"])
        node_size_factor = st.slider("Node Size", 0.5, 2.0, 1.0, 0.1)
    
    # Run clustering
    if len(events) < 10:
        st.error(f"⚠️ Only {len(events)} events. Need at least 10.")
        return
        
    with st.spinner(f"🔄 Running clustering..."):
        results = run_clustering_for_period(events, resolution, target_clusters, period_key)
    
    edges_clean = results.get('edges_clean', pd.DataFrame())
    builder_master = results.get('builder_master', pd.DataFrame())
    cluster_summary = results.get('cluster_summary', pd.DataFrame())
    G = results.get('graph', nx.Graph())
    
    # P&L Data
    pnl_recipient = build_builder_pnl(events, lens="recipient", date_basis="lead_date", freq="ALL")
    pnl_recipient = pnl_recipient.drop(columns=["period_start"], errors="ignore")
    builder_master = builder_master.merge(pnl_recipient, on="BuilderRegionKey", how="left")
    builder_master = compute_network_metrics(G, builder_master)

    # ==========================================
    # 💡 INVESTMENT RECOMMENDATIONS
    # ==========================================
    st.header("💡 Investment Recommendations")
    
    cpr_recs = compute_cpr_recommendations(edges_clean, builder_master)
    
    if opt_mode == "Standard (Lowest CPR)":
        st.markdown("### 📊 Generic Efficiency View")
        st.caption("Ranking builders by lowest cost per referral generated, regardless of destination.")
        
        if not cpr_recs.empty:
            cols = st.columns(4)
            for i, (_, row) in enumerate(cpr_recs.head(4).iterrows()):
                cols[i].metric(
                    f"#{i+1} {row['BuilderRegionKey']}",
                    f"${row['CPR']:,.0f} CPR",
                    f"{int(row['Total_Referrals_Sent'])} refs"
                )
            
            with st.expander("View Full Ranking", expanded=True):
                st.dataframe(
                    cpr_recs[['BuilderRegionKey', 'MediaCost', 'Total_Referrals_Sent', 'CPR', 'ROAS']]
                    .style.format({'MediaCost': '${:,.0f}', 'CPR': '${:,.2f}', 'ROAS': '{:.2f}'})
                    .background_gradient(subset=['CPR'], cmap='RdYlGn_r'),
                    use_container_width=True
                )
        else:
            st.info("No data for standard CPR recommendations.")
    
    else:
        # --- ADVANCED MODE: NETWORK INVESTMENT ENGINE ---
        st.markdown("### 🎯 Network Investment Engine")
        st.caption("Identify gap -> Find leverage -> Calculate investment & spillover.")
        
        # 1. Demand Analysis
        shortfall_data = calculate_shortfalls(events)
        
        # Simulation UI
        with st.expander("⚙️ Simulation Controls (Target Override)", expanded=False):
            st.info("Manually override a target to test the engine.")
            builders = sorted(shortfall_data['BuilderRegionKey'].unique())
            sim_target = st.selectbox("Select Builder:", ["(None)"] + builders)
            sim_amount = st.number_input("Additional Leads Needed:", min_value=0, value=50)
            
            if sim_target != "(None)" and sim_amount > 0:
                mask = shortfall_data['BuilderRegionKey'] == sim_target
                # Increase target, which increases shortfall
                shortfall_data.loc[mask, 'LeadTarget'] += sim_amount
                shortfall_data.loc[mask, 'Shortfall'] += sim_amount
                st.success(f"Added {sim_amount} leads to {sim_target}'s target.")
        
        # 2. Leverage Analysis
        leverage_data = analyze_network_leverage(events)
        
        # 3. Focus Selection
        st.markdown("#### 1. Focus Selection (Demand)")
        
        # Filter to those with Shortfall > 0 for relevance
        shortfall_only = shortfall_data[shortfall_data['Shortfall'] > 0].copy()
        
        if shortfall_only.empty:
            st.success("✅ All builders are currently hitting their lead targets!")
            focus_builder = None
        else:
            # Sort by Urgency (Days Remaining) or Shortfall amount
            shortfall_only['Urgency_Score'] = shortfall_only['Shortfall'] * (1 / (shortfall_only['Days_Remaining'].replace(0, 1) + 1))
            
            focus_builder = st.selectbox(
                "Select Focus Builder (Prioritized by Gap & Urgency):",
                options=shortfall_only.sort_values('Shortfall', ascending=False)['BuilderRegionKey'],
                format_func=lambda x: f"{x} (Gap: {int(shortfall_only[shortfall_only['BuilderRegionKey']==x]['Shortfall'].iloc[0])})"
            )
            
            # --- GAP ANALYSIS CONTAINER ---
            st.markdown("---")
            st.subheader(f"⚡ Gap Analysis: {focus_builder}")
            
            # Get specific data
            target_data = shortfall_data[shortfall_data['BuilderRegionKey'] == focus_builder].iloc[0]
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Target", int(target_data['LeadTarget']))
            m2.metric("Actual", int(target_data['Actual_Referrals']))
            m3.metric("Shortfall", int(target_data['Shortfall']), delta="Gap", delta_color="inverse")
            
            days_rem = int(target_data['Days_Remaining']) if pd.notnull(target_data['Days_Remaining']) else None
            m4.metric("Days Remaining", f"{days_rem}" if days_rem is not None else "N/A")
            
            # Get Strategies
            strategies = generate_investment_strategies(focus_builder, shortfall_data, leverage_data, events)
            
            if not strategies.empty:
                st.markdown("#### 2. Best Pathways to Close Gap")
                
                # Highlight Best Option
                best = strategies.iloc[0]
                
                b1, b2, b3 = st.columns(3)
                b1.metric("Best Source", best['Source_Builder'])
                b2.metric("Effective CPR", fmt_currency(best['Effective_CPR']), help=f"Base CPR (${best['Base_CPR']:,.0f}) / Transfer Rate ({best['Transfer_Rate']:.1%})")
                b3.metric("Investment Needed", fmt_currency(best['Investment_Required']))
                
                # Spillover Bonus
                if best['Optimization_Score'] > 0:
                    st.success(f"🔥 **Network Bonus:** Closing this gap also helps: {best['Spillover_Impact']}")
                
                # Data Table
                st.dataframe(
                    strategies[['Source_Builder', 'Transfer_Rate', 'Base_CPR', 'Effective_CPR', 'Investment_Required', 'Spillover_Impact']]
                    .rename(columns={'Source_Builder': 'Source', 'Transfer_Rate': 'Transfer %', 'Base_CPR': 'Base CPR', 'Effective_CPR': 'Effective CPR', 'Investment_Required': 'Investment', 'Spillover_Impact': 'Spillover'})
                    .style.format({
                        'Transfer %': '{:.1%}', 'Base CPR': '${:,.2f}', 'Effective CPR': '${:,.2f}', 'Investment': '${:,.0f}'
                    })
                    .background_gradient(subset=['Effective CPR'], cmap='RdYlGn_r'),
                    use_container_width=True, hide_index=True
                )
                
                # Interactive Visualization
                st.markdown("#### 3. Flow Visualization")
                
                # Allow user to select a strategy to visualize
                selected_strat_source = st.selectbox("Visualize Investment in:", strategies['Source_Builder'])
                selected_strat_row = strategies[strategies['Source_Builder'] == selected_strat_source].iloc[0]
                
                sankey = render_investment_sankey(selected_strat_row, focus_builder)
                if sankey:
                    st.plotly_chart(sankey, use_container_width=True)
                    
            else:
                st.warning("No existing referral pathways found to this builder. Consider establishing new partnerships or direct media.")

    st.markdown("---")
    
    # ==========================================
    # GLOBAL SEARCH & GRAPH (Keep existing)
    # ==========================================
    st.subheader("🕸️ Network Graph Explorer")
    
    # ... (Rest of existing graph code)
    # Cluster Select
    cluster_labels = {int(r.ClusterId): f"Cluster {int(r.ClusterId)} ({int(r.N_builders)} builders)" for r in cluster_summary.itertuples()}
    
    sel_cluster_id = st.selectbox("Select Ecosystem", options=list(cluster_labels.keys()), format_func=lambda x: cluster_labels[x])
    
    # Filter Graph Data
    sub_builders = builder_master[builder_master["ClusterId"] == sel_cluster_id]
    sub_edges = edges_clean[(edges_clean["Cluster_origin"] == sel_cluster_id) & (edges_clean["Cluster_dest"] == sel_cluster_id)]
    
    render_network_graph(sub_builders, sub_edges, G, None, show_labels, edge_style, node_size_factor)


def render_network_graph(builders, edges, G, focus_builder, show_labels, edge_style, node_size_factor):
    if len(G.nodes) == 0:
        st.info("No network edges to display.")
        return
    
    pos = nx.spring_layout(G, weight="weight", seed=42, k=1.5, iterations=50)
    
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    
    # Normalize coords for plotly
    for n in pos:
        pos[n] = ((pos[n][0] - np.mean(xs)) * 2, (pos[n][1] - np.mean(ys)) * 2)
    
    traces = []
    
    for _, row in edges.iterrows():
        u, v = row["Origin_builder"], row["Dest_builder"]
        if u not in pos or v not in pos: continue
        
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None], mode="lines",
            line=dict(width=1, color='#888'), hoverinfo='none'
        ))
    
    node_x, node_y, texts, colors = [], [], [], []
    for node in G.nodes():
        if node not in pos: continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        texts.append(node)
        colors.append('#1f77b4') # Default blue
        
    traces.append(go.Scatter(
        x=node_x, y=node_y, mode="markers+text" if show_labels else "markers",
        text=texts, hoverinfo='text',
        marker=dict(size=10*node_size_factor, color=colors)
    ))
    
    fig = go.Figure(data=traces)
    fig.update_layout(showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0), height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_focus_analysis(focus_builder, builders, edges, cpr_recs):
    # Simplified placeholder for legacy call
    pass


if __name__ == "__main__":
    main()