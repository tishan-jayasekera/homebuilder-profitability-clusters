"""
Referral Networks Dashboard - Network Intelligence Engine
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

from src.data_loader import load_events
from src.normalization import normalize_events
from src.referral_clusters import run_referral_clustering
from src.network_optimization import (
    calculate_shortfalls, 
    analyze_network_leverage, 
    optimize_campaign_spend,
    analyze_network_health
)

st.set_page_config(page_title="Network Intelligence", page_icon="🕸️", layout="wide")

# ==========================================
# STYLING & INIT
# ==========================================
st.markdown("""
<style>
    .kpi-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .kpi-label { font-size: 12px; color: #64748B; font-weight: 600; text-transform: uppercase; }
    .kpi-value { font-size: 20px; font-weight: 700; color: #0F172A; }
    .kpi-sub { font-size: 12px; color: #64748B; }
    .add-btn { width: 100%; }
</style>
""", unsafe_allow_html=True)

if 'campaign_targets' not in st.session_state:
    st.session_state['campaign_targets'] = []

# ==========================================
# DATA LOADING
# ==========================================
def load_data():
    if 'events_file' not in st.session_state:
        return None
    events = load_events(st.session_state['events_file'])
    if events is None:
        return None
    return normalize_events(events)

# ==========================================
# VISUALIZATION ENGINE
# ==========================================
def render_ego_network(G, focus_node, all_nodes_metrics=None):
    """
    Renders a directed Ego Graph for the focus node.
    - One-Way flows: Thin grey lines
    - Two-Way flows: Thick blue lines
    """
    # 1. Extract Subgraph
    if focus_node and focus_node in G:
        # Radius 1 includes direct neighbors
        sub_G = nx.ego_graph(G, focus_node, radius=1)
    else:
        # Fallback to main graph (maybe simplified if too large)
        sub_G = G

    # 2. Layout
    pos = nx.spring_layout(sub_G, seed=42, k=0.5)

    # 3. Edges (Categorized)
    edge_traces = []
    
    for u, v, data in sub_G.edges(data=True):
        weight = data.get('weight', 1)
        
        # Check Reciprocity
        is_reciprocal = sub_G.has_edge(v, u)
        
        if is_reciprocal:
            color = '#3B82F6' # Blue
            width = 3 + (np.log(weight) * 0.5)
            opacity = 0.8
        else:
            color = '#94A3B8' # Grey
            width = 1
            opacity = 0.5

        # Straight line for simplicity/performance in Streamlit
        # (Curvature requires spline logic which is heavy for Plotly lines unless using annotations)
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=width, color=color),
            opacity=opacity,
            hoverinfo='text',
            text=f"{u} → {v} ({int(weight)} leads)",
            mode='lines'
        ))

    # 4. Nodes
    node_x = []
    node_y = []
    node_color = []
    node_size = []
    node_text = []
    
    for node in sub_G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Size by Risk if available, else default
        risk = 0
        if all_nodes_metrics is not None and node in all_nodes_metrics.index:
            risk = all_nodes_metrics.loc[node, 'Risk_Score']
        
        # Highlight Focus Node
        if node == focus_node:
            node_color.append('#EF4444') # Red for focus
            node_size.append(30)
        else:
            # Color by Role if available in graph attributes
            role = sub_G.nodes[node].get('Role', 'Unknown')
            if 'Hub' in role:
                node_color.append('#10B981') # Green
            elif 'Zombie' in role:
                node_color.append('#64748B') # Grey
            else:
                node_color.append('#3B82F6') # Blue
            
            node_size.append(15 + (risk/10))

        node_text.append(f"<b>{node}</b><br>Risk: {risk:.0f}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=2, color='white')
        )
    )

    # 5. Figure Construction
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title="",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0,l=0,r=0,t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=500
        )
    )
    
    return fig

# ==========================================
# MAIN APP
# ==========================================
def main():
    events_full = load_data()
    
    if events_full is None:
        st.warning("⚠️ Please upload events data on the Home page first.")
        return

    # --- SIDEBAR: CAMPAIGN CART ---
    with st.sidebar:
        st.header("🛒 Campaign Plan")
        
        cart = st.session_state['campaign_targets']
        
        if not cart:
            st.info("Select builders to add them to your optimization plan.")
        else:
            st.success(f"{len(cart)} Builders Selected")
            for b in cart:
                st.caption(f"• {b}")
                
            if st.button("Clear Cart", type="secondary"):
                st.session_state['campaign_targets'] = []
                st.rerun()

        st.divider()
        
        # Global Scenario Settings
        st.subheader("⚙️ Scenario")
        scen_v_mult = st.slider("Market Velocity", 0.5, 1.5, 1.0, 0.1)
        scen_strict = st.checkbox("Strict Capacity Check", value=True)
        
        if cart:
            st.divider()
            if st.button("✨ Generate Plan", type="primary", use_container_width=True):
                st.session_state['show_plan'] = True
    
    # --- MAIN UI ---
    st.title("Network Intelligence Engine")
    
    # 1. Global Calculations
    # Note: We calculate metrics globally first to populate search & graphs
    with st.spinner("Analyzing ecosystem..."):
        shortfall_df = calculate_shortfalls(
            events_full, 
            scenario_params={'velocity_mult': scen_v_mult}
        )
        leverage_df = analyze_network_leverage(events_full)
        health_df = analyze_network_health(events_full)
        
        # Run Clustering for Graph
        cluster_res = run_referral_clustering(events_full)
        G = cluster_res.get('graph', nx.Graph())
        
        # Enrich Graph with Health Roles
        if not health_df.empty:
            role_dict = health_df.set_index('Builder')['Role'].to_dict()
            nx.set_node_attributes(G, role_dict, 'Role')

    # 2. Search Interface
    all_builders = sorted(shortfall_df['BuilderRegionKey'].unique())
    
    c1, c2 = st.columns([3, 1])
    with c1:
        selected_builder = st.selectbox(
            "🔍 Find a Builder", 
            options=["None"] + all_builders,
            index=0
        )
    
    # 3. Context & Visualization
    if selected_builder != "None":
        
        # --- Builder Context Row ---
        b_metrics = shortfall_df[shortfall_df['BuilderRegionKey'] == selected_builder].iloc[0]
        
        # Safe Integer Conversion
        tgt = int(b_metrics['LeadTarget']) if pd.notna(b_metrics['LeadTarget']) else 0
        proj = int(b_metrics['Projected_Total']) if pd.notna(b_metrics['Projected_Total']) else 0
        gap = int(b_metrics['Net_Gap']) if pd.notna(b_metrics['Net_Gap']) else 0
        risk = int(b_metrics['Risk_Score']) if pd.notna(b_metrics['Risk_Score']) else 0
        
        col_kpi1, col_kpi2, col_kpi3, col_act = st.columns([1, 1, 1, 1.5])
        
        with col_kpi1:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-label">Lead Target</div>
                <div class="kpi-value">{tgt}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_kpi2:
            color = "#EF4444" if gap < 0 else "#10B981"
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-label">Proj. Gap</div>
                <div class="kpi-value" style="color: {color}">{gap:+}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_kpi3:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-label">Risk Score</div>
                <div class="kpi-value">{risk}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_act:
            st.markdown("<br>", unsafe_allow_html=True)
            if selected_builder not in st.session_state['campaign_targets']:
                if st.button("➕ Add to Campaign", use_container_width=True):
                    st.session_state['campaign_targets'].append(selected_builder)
                    st.rerun()
            else:
                if st.button("❌ Remove from Campaign", use_container_width=True):
                    st.session_state['campaign_targets'].remove(selected_builder)
                    st.rerun()

        # --- The Ecosystem Map (Ego Graph) ---
        st.subheader(f"Ecosystem: {selected_builder}")
        
        viz_col, info_col = st.columns([2.5, 1])
        
        with viz_col:
            # Pass metrics for sizing
            metrics_idx = shortfall_df.set_index('BuilderRegionKey')
            fig = render_ego_network(G, selected_builder, metrics_idx)
            st.plotly_chart(fig, use_container_width=True)
            
        with info_col:
            st.markdown("**Leverage Opportunities**")
            # Show top sources for this builder
            sources = leverage_df[leverage_df['Dest_BuilderRegionKey'] == selected_builder].sort_values('eCPR')
            
            if not sources.empty:
                for _, s in sources.head(4).iterrows():
                    st.info(f"**{s['MediaPayer_BuilderRegionKey']}**\n\nCPR: ${s['eCPR']:.0f} • TR: {s['Transfer_Rate']:.0%}")
            else:
                st.warning("No existing inbound flow found.")

    else:
        # Default View (Global)
        st.info("Select a builder above to explore their specific network ecosystem and risks.")
        
        # Show Cluster Map (Simplified)
        if G.number_of_nodes() > 0:
            st.subheader("Global Cluster Map")
            # Just render the whole graph (lightweight version)
            fig_global = render_ego_network(G, None)
            st.plotly_chart(fig_global, use_container_width=True)
        else:
            st.warning("No network data available.")

    # 4. Campaign Planner (Bottom Section)
    if st.session_state.get('show_plan', False) and st.session_state['campaign_targets']:
        st.divider()
        st.header("📋 Campaign Optimization Plan")
        
        plan_df = optimize_campaign_spend(
            st.session_state['campaign_targets'],
            shortfall_df,
            leverage_df,
            strict_capacity=scen_strict
        )
        
        if not plan_df.empty:
            st.dataframe(
                plan_df.style.format({
                    "Est_Investment": "${:,.0f}",
                    "Effective_CPR": "${:,.0f}",
                    "Gap_Leads": "{:,.0f}"
                }).background_gradient(subset=['Est_Investment'], cmap="Greens"),
                use_container_width=True,
                hide_index=True
            )
            
            total_inv = plan_df['Est_Investment'].sum()
            st.metric("Total Campaign Investment", f"${total_inv:,.0f}")
            
        else:
            st.warning("Optimization yielded no results. Check builder connectivity.")

if __name__ == "__main__":
    main()