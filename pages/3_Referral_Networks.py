"""
Referral Networks Dashboard - Streamlit Page
Filename: pages/3_Referral_Networks.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

import sys
from pathlib import Path

root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data_loader import load_events, export_to_excel
from src.normalization import normalize_events
from src.referral_clusters import run_referral_clustering
from src.utils import fmt_currency, fmt_roas
from src.network_optimization import (
    calculate_shortfalls, 
    analyze_network_leverage, 
    generate_global_media_plan,
    analyze_network_health,
    generate_investment_strategies
)

st.set_page_config(page_title="Referral Networks", page_icon="🔗", layout="wide")

# ==========================================
# STYLING
# ==========================================
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #0F172A; }
    .metric-label { font-size: 14px; color: #64748B; margin-bottom: 5px; }
    .priority-critical { color: #dc2626; font-weight: bold; }
    .priority-high { color: #ea580c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🔗 Referral Network & Optimization Engine")
st.markdown("Leverage network effects to close lead gaps efficiently.")


def load_data():
    if 'events_file' not in st.session_state:
        return None
    events = load_events(st.session_state['events_file'])
    if events is None:
        return None
    return normalize_events(events)

def get_date_range_info(events):
    date_col = 'lead_date' if 'lead_date' in events.columns else 'RefDate'
    if date_col not in events.columns:
        return None, [], None
    dates = pd.to_datetime(events[date_col], errors='coerce').dropna()
    if dates.empty:
        return date_col, [], None
    min_date, max_date = dates.min(), dates.max()
    month_starts = pd.date_range(start=min_date.replace(day=1), end=max_date.replace(day=1), freq='MS')
    return date_col, month_starts

def render_interactive_network_plotly(G, selected_cluster_id):
    """
    Render interactive network using Plotly (Standard, no extra deps).
    Highlights the selected cluster.
    """
    # 1. Compute Layout
    # Use spring layout for organic look
    pos = nx.spring_layout(G, seed=42, k=0.5) 

    # 2. Edges
    edge_x = []
    edge_y = []
    
    for u, v, data in G.edges(data=True):
        # Filter: Only show edges connected to selected cluster (to reduce noise)
        u_cluster = G.nodes[u].get('cluster')
        v_cluster = G.nodes[v].get('cluster')
        
        if u_cluster == selected_cluster_id or v_cluster == selected_cluster_id:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # 3. Nodes
    node_x = []
    node_y = []
    node_color = []
    node_text = []
    node_size = []

    for node in G.nodes():
        if node not in pos: continue
        
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        cluster = G.nodes[node].get('cluster', -1)
        role = G.nodes[node].get('Role', 'Unknown')
        
        # Color logic: Highlight selected cluster
        if cluster == selected_cluster_id:
            node_color.append(1) # Highlight
            sz = 20
        else:
            node_color.append(0) # Grey out
            sz = 10
            
        node_size.append(sz)
        node_text.append(f"<b>{node}</b><br>Cluster: {cluster}<br>Role: {role}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            colorscale=[[0, '#CBD5E1'], [1, '#1D4ED8']], # Grey vs Blue
            color=node_color,
            size=node_size,
            line_width=2
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=f"Cluster {selected_cluster_id} Ecosystem",
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
    )
    
    return fig

def main():
    events_full = load_data()
    
    if events_full is None:
        st.warning("⚠️ Please upload events data on the Home page first.")
        st.page_link("app.py", label="← Go to Home", icon="🏠")
        return

    # --- SIDEBAR CONTROLS ---
    date_col, available_months = get_date_range_info(events_full)
    
    with st.sidebar:
        st.header("📅 Time Period")
        use_all_time = st.checkbox("Use All Time", value=True)
        
        if use_all_time or not len(available_months):
            events_selected = events_full
            if date_col and not events_full.empty:
                dates = pd.to_datetime(events_full[date_col], errors='coerce')
                period_days = max((dates.max() - dates.min()).days, 1)
            else:
                period_days = 90
        else:
            start_month, end_month = st.select_slider(
                "Select Analysis Range",
                options=available_months,
                value=(available_months[0], available_months[-1]),
                format_func=lambda x: x.strftime("%b %Y")
            )
            end_date_filter = end_month + pd.offsets.MonthEnd(1)
            mask = (pd.to_datetime(events_full[date_col]) >= start_month) & (pd.to_datetime(events_full[date_col]) <= end_date_filter)
            events_selected = events_full.loc[mask].copy()
            period_days = max((end_date_filter - start_month).days, 1)
            st.info(f"Analyzing {period_days} days.")

        st.divider()
        
        # --- SCENARIO SIMULATOR ---
        with st.expander("🎲 Scenario Simulator", expanded=False):
            st.caption("Adjust to forecast future network states.")
            
            scen_target_mult = st.slider(
                "Target Multiplier", 0.5, 2.0, 1.0, 0.1,
                help="Scale builder targets up or down (e.g. 1.2 = 20% higher targets)"
            )
            
            scen_velocity_mult = st.slider(
                "Velocity Multiplier", 0.5, 2.0, 1.0, 0.1,
                help="Scale current lead pace (e.g. 1.1 = Market heats up by 10%)"
            )
            
            scen_strict_cap = st.checkbox(
                "Strict Capacity Check", value=False,
                help="If checked, only recommend sources that have a projected surplus."
            )
            
            scenario_params = {
                'target_mult': scen_target_mult,
                'velocity_mult': scen_velocity_mult
            }

    # --- CALCULATIONS ---
    # 1. Shortfall Analysis (With Scenario Params)
    shortfall_data = calculate_shortfalls(
        events_df=events_selected,
        targets_df=None, 
        period_days=period_days,
        total_events_df=events_full,
        scenario_params=scenario_params
    )
    
    # 2. Leverage Analysis
    leverage_data = analyze_network_leverage(events_selected)
    
    # 3. Global Plan Generation (With Strict Capacity Param)
    media_plan = generate_global_media_plan(shortfall_data, leverage_data, strict_capacity=scen_strict_cap)
    
    # 4. Network Health
    health_data = analyze_network_health(events_selected)

    # --- TABS ---
    tab_plan, tab_explorer, tab_health, tab_details = st.tabs([
        "🚀 Optimization Plan", 
        "🕸️ Network Explorer", 
        "🏥 Network Health",
        "🔍 Builder Deep Dive"
    ])

    # ----------------------------------------
    # TAB 1: OPTIMIZATION PLAN
    # ----------------------------------------
    with tab_plan:
        st.header("Strategic Media Allocation")
        
        # Top Metrics
        total_shortfall = shortfall_data['Projected_Shortfall'].sum()
        builders_at_risk = shortfall_data[shortfall_data['Projected_Shortfall'] > 0]['BuilderRegionKey'].nunique()
        total_surplus = shortfall_data['Projected_Surplus'].sum()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Projected Shortfall", f"{int(total_shortfall):,}", help="Based on scenario settings")
        c2.metric("Builders At Risk", f"{builders_at_risk}")
        c3.metric("Available Surplus", f"{int(total_surplus):,}")
        
        if scen_target_mult != 1.0 or scen_velocity_mult != 1.0:
            st.info(f"⚡ Scenario Active: Targets x{scen_target_mult}, Velocity x{scen_velocity_mult}")

        st.divider()
        
        # Media Plan Table
        if media_plan.empty:
            st.success("✅ No interventions required. All builders on track.")
        else:
            st.markdown("### 📋 Recommended Interventions")
            st.dataframe(
                media_plan.style.format({
                    "Est_Investment": "${:,.0f}",
                    "Effective_CPR": "${:,.0f}",
                    "Gap_Leads": "{:,.0f}"
                }).background_gradient(subset=['Est_Investment'], cmap="Reds"),
                use_container_width=True,
                hide_index=True
            )
            
            # Risk Matrix
            if not shortfall_data.empty:
                risk_df = shortfall_data[shortfall_data['Projected_Shortfall'] > 0].copy()
                if not risk_df.empty:
                    fig = px.scatter(
                        risk_df,
                        x="Days_Remaining",
                        y="Projected_Shortfall",
                        size="Risk_Score",
                        color="Risk_Score",
                        hover_name="BuilderRegionKey",
                        color_continuous_scale="RdYlGn_r",
                        title="Risk Matrix (Urgency vs Gap)"
                    )
                    fig.add_vline(x=30, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------
    # TAB 2: NETWORK EXPLORER
    # ----------------------------------------
    with tab_explorer:
        st.subheader("Interactive Ecosystem Map")
        
        c1, c2 = st.columns(2)
        with c1:
            resolution = st.slider("Clustering Resolution", 0.5, 2.5, 1.5, 0.1)
        with c2:
            target_clusters = st.slider("Max Clusters", 3, 25, 15, 1)
        
        # Run Clustering
        with st.spinner("Analyzing network topology..."):
            results = run_referral_clustering(events_selected, resolution=resolution, target_max_clusters=target_clusters)
            
        G = results.get('graph', nx.Graph())
        cluster_summary = results.get('cluster_summary', pd.DataFrame())
        
        if G.nodes:
            # Dropdown for cluster selection
            cluster_labels = {int(r.ClusterId): f"Cluster {int(r.ClusterId)} ({int(r.N_builders)} builders)" for r in cluster_summary.itertuples()}
            sel_cluster_id = st.selectbox("Select Ecosystem to Visualize", options=list(cluster_labels.keys()), format_func=lambda x: cluster_labels[x])
            
            # Add cluster info to graph nodes for visualization
            partition = results['builder_master'].set_index('BuilderRegionKey')['ClusterId'].to_dict()
            nx.set_node_attributes(G, partition, 'cluster')
            
            # Render Plotly (Replacing PyVis/AGraph)
            fig_net = render_interactive_network_plotly(G, sel_cluster_id)
            st.plotly_chart(fig_net, use_container_width=True)
        else:
            st.warning("No network connections found in this period.")

    # ----------------------------------------
    # TAB 3: NETWORK HEALTH
    # ----------------------------------------
    with tab_health:
        st.header("🏥 Network Health Monitor")
        st.markdown("Diagnose broken pathways and identify zombie nodes.")
        
        if health_data.empty:
            st.info("No health data available.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Nodes", len(health_data))
            c2.metric("Zombie Nodes", len(health_data[health_data['Role'].str.contains("Zombie")]))
            c3.metric("Healthy Hubs", len(health_data[health_data['Role'].str.contains("Hub")]))
            c4.metric("Avg Ratio (Give/Take)", f"{health_data['Ratio_Give_Take'].mean():.2f}")
            
            st.dataframe(
                health_data.sort_values('Leads_Received', ascending=False)
                .style.format({'Ratio_Give_Take': '{:.2f}'})
                .background_gradient(subset=['Ratio_Give_Take'], cmap="coolwarm"),
                use_container_width=True,
                hide_index=True
            )

    # ----------------------------------------
    # TAB 4: BUILDER DEEP DIVE
    # ----------------------------------------
    with tab_details:
        st.header("Single Builder Analysis")
        
        # Select Builder
        builders = sorted(shortfall_data['BuilderRegionKey'].unique())
        sel_builder = st.selectbox("Select Builder", builders)
        
        if sel_builder:
            row = shortfall_data[shortfall_data['BuilderRegionKey'] == sel_builder].iloc[0]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Lead Target", int(row['LeadTarget']))
            c2.metric("Projected Total", int(row['Projected_Total']), delta=int(row['Net_Gap']))
            c3.metric("Risk Score", int(row['Risk_Score']))
            
            st.divider()
            
            # Strategies
            strats = generate_investment_strategies(sel_builder, shortfall_data, leverage_data, events_selected)
            
            if not strats.empty:
                st.subheader("Inbound Pathways")
                st.dataframe(strats.style.format({'Effective_CPR': '${:,.0f}', 'Investment_Required': '${:,.0f}'}))
            else:
                st.info("No inbound referral history found.")

if __name__ == "__main__":
    main()