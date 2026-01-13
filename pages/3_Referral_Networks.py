"""
Network Intelligence Engine - Unified Referral Network Dashboard
Improved: Clear clustering, operational traceability, simple campaign cart
"""
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data_loader import load_events, export_to_excel
from src.normalization import normalize_events
from src.referral_clusters import run_referral_clustering
from src.utils import fmt_currency
from src.network_optimization import (
    calculate_shortfalls, analyze_network_leverage, generate_targeted_media_plan,
    analyze_network_health, get_builder_referral_history, get_cluster_summary,
    calculate_campaign_summary
)

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Network Intelligence", page_icon="🔗", layout="wide")

# ==========================================
# CLUSTER COLORS - 15 distinct colors
# ==========================================
CLUSTER_COLORS = [
    '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
    '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
    '#14b8a6', '#e11d48', '#0ea5e9', '#a855f7', '#22c55e'
]

def get_cluster_color(cluster_id):
    return CLUSTER_COLORS[(cluster_id - 1) % len(CLUSTER_COLORS)]

# ==========================================
# SESSION STATE
# ==========================================
if 'campaign_targets' not in st.session_state:
    st.session_state.campaign_targets = set()
if 'selected_builder' not in st.session_state:
    st.session_state.selected_builder = None
if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = None

# ==========================================
# DATA LOADING
# ==========================================
@st.cache_data(show_spinner=False)
def load_and_process():
    if 'events_file' not in st.session_state:
        return None
    events = load_events(st.session_state['events_file'])
    return normalize_events(events) if events is not None else None

def get_all_builders(events_df):
    builders = set()
    for col in ['Dest_BuilderRegionKey', 'MediaPayer_BuilderRegionKey']:
        if col in events_df.columns:
            builders.update(events_df[col].dropna().unique())
    return sorted(builders)

# ==========================================
# NETWORK VISUALIZATION
# ==========================================
def render_cluster_map(G, pos, partition_dict, cluster_summary_df, selected_cluster=None, selected_builder=None):
    """Render network with clear cluster coloring."""
    fig = go.Figure()
    
    # Draw edges first
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        # Determine visibility based on selection
        u_cluster = partition_dict.get(u, 0)
        v_cluster = partition_dict.get(v, 0)
        
        if selected_cluster and u_cluster != selected_cluster and v_cluster != selected_cluster:
            opacity = 0.05
            color = '#e2e8f0'
        elif selected_cluster:
            # Highlight edges within or connected to selected cluster
            if u_cluster == selected_cluster and v_cluster == selected_cluster:
                opacity = 0.8
                color = get_cluster_color(selected_cluster)
            else:
                opacity = 0.3
                color = '#94a3b8'
        else:
            opacity = 0.4
            color = '#94a3b8'
        
        weight = data.get('weight', 1)
        width = 0.5 + min(weight / 10, 2)
        
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines', hoverinfo='skip',
            line=dict(color=color, width=width),
            opacity=opacity, showlegend=False
        ))
    
    # Draw nodes by cluster
    clusters = set(partition_dict.values())
    
    for cluster_id in sorted(clusters):
        cluster_nodes = [n for n, c in partition_dict.items() if c == cluster_id and n in pos]
        if not cluster_nodes:
            continue
        
        color = get_cluster_color(cluster_id)
        
        # Determine if this cluster is dimmed
        if selected_cluster and cluster_id != selected_cluster:
            color = '#e2e8f0'
            opacity = 0.3
        else:
            opacity = 1.0
        
        node_x = [pos[n][0] for n in cluster_nodes]
        node_y = [pos[n][1] for n in cluster_nodes]
        
        # Size based on degree
        sizes = [12 + min(G.degree(n, weight='weight') / 5, 20) for n in cluster_nodes]
        
        # Highlight selected builder
        if selected_builder in cluster_nodes:
            idx = cluster_nodes.index(selected_builder)
            sizes[idx] = 35
        
        hover_text = [f"<b>{n}</b><br>Cluster {cluster_id}" for n in cluster_nodes]
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode='markers',
            marker=dict(
                size=sizes, color=color, opacity=opacity,
                line=dict(width=2, color='white')
            ),
            text=hover_text, hoverinfo='text',
            name=f"Cluster {cluster_id}",
            showlegend=not selected_cluster or cluster_id == selected_cluster
        ))
    
    fig.update_layout(
        height=500, margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor='#fafafa', plot_bgcolor='#fafafa',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='closest'
    )
    
    return fig

# ==========================================
# REFERRAL HISTORY (TRACEABILITY)
# ==========================================
def render_referral_history(history: dict, shortfall_row: pd.Series):
    """Render operational referral history for a builder."""
    
    st.markdown(f"### 📋 {history['builder']}")
    
    # Status banner
    if shortfall_row is not None and not shortfall_row.empty:
        gap = shortfall_row.get('Net_Gap', 0)
        risk = shortfall_row.get('Risk_Score', 0)
        if gap >= 0:
            st.success(f"✅ **On Track** — Projected surplus of {int(gap)} leads")
        elif risk > 50:
            st.error(f"🔴 **Critical** — {int(abs(gap))} lead shortfall, Risk Score: {int(risk)}")
        elif risk > 25:
            st.warning(f"🟠 **At Risk** — {int(abs(gap))} lead shortfall, Risk Score: {int(risk)}")
        else:
            st.info(f"🟡 **Monitor** — {int(abs(gap))} lead shortfall")
    
    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Referrals Received", f"{history['total_received']:,}")
    c2.metric("Referrals Sent", f"{history['total_sent']:,}")
    c3.metric("Unique Sources", history['unique_sources'])
    c4.metric("Unique Destinations", history['unique_destinations'])
    
    # Monthly trend
    st.markdown("#### 📈 Monthly Referrals Received")
    monthly = history.get('monthly_trend', pd.DataFrame())
    
    if not monthly.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly['month'], y=monthly['referrals'],
            marker_color='#3b82f6'
        ))
        fig.update_layout(
            height=200, margin=dict(l=10, r=10, t=10, b=30),
            xaxis_title='', yaxis_title='Referrals',
            plot_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No monthly data available")
    
    # Top sources
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔗 Top Referral Sources")
        st.caption("Who sends leads to this builder")
        sources = history.get('top_sources', pd.DataFrame())
        if not sources.empty:
            st.dataframe(
                sources.head(5).style.format({'Media Value': '${:,.0f}'}),
                hide_index=True, use_container_width=True
            )
        else:
            st.caption("No inbound referrals")
    
    with col2:
        st.markdown("#### 📤 Top Destinations")
        st.caption("Who this builder sends leads to")
        dests = history.get('top_destinations', pd.DataFrame())
        if not dests.empty:
            st.dataframe(dests.head(5), hide_index=True, use_container_width=True)
        else:
            st.caption("No outbound referrals")

# ==========================================
# CAMPAIGN CART (SIMPLIFIED)
# ==========================================
def render_campaign_sidebar(all_builders, shortfall_df):
    """Render simplified campaign cart in sidebar."""
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🛒 Campaign Cart")
    
    targets = st.session_state.campaign_targets
    
    if not targets:
        st.sidebar.info("No builders selected yet")
    else:
        st.sidebar.markdown(f"**{len(targets)} builder(s) selected**")
        
        # List with remove buttons
        for t in list(targets):
            col1, col2 = st.sidebar.columns([4, 1])
            
            # Get risk status
            row = shortfall_df[shortfall_df['BuilderRegionKey'] == t]
            if not row.empty:
                risk = row['Risk_Score'].iloc[0]
                icon = "🔴" if risk > 50 else ("🟠" if risk > 25 else "🟢")
            else:
                icon = "⚪"
            
            col1.markdown(f"{icon} {t[:20]}")
            if col2.button("✕", key=f"rm_{t}"):
                st.session_state.campaign_targets.discard(t)
                st.rerun()
        
        # Clear all
        if st.sidebar.button("🗑️ Clear All", use_container_width=True):
            st.session_state.campaign_targets = set()
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # Quick summary
        at_risk = sum(1 for t in targets if not shortfall_df[
            (shortfall_df['BuilderRegionKey'] == t) & (shortfall_df['Risk_Score'] > 25)
        ].empty)
        
        st.sidebar.metric("At Risk", f"{at_risk} / {len(targets)}")

# ==========================================
# MAIN
# ==========================================
def main():
    st.title("🔗 Network Intelligence Engine")
    st.caption("Understand referral flows, identify leverage points, plan campaigns")
    
    events = load_and_process()
    
    if events is None:
        st.warning("⚠️ Upload events data on the Home page first.")
        st.page_link("app.py", label="← Go to Home", icon="🏠")
        return
    
    # ==========================================
    # SIDEBAR CONTROLS
    # ==========================================
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Date filter
        date_col = 'lead_date' if 'lead_date' in events.columns else 'RefDate'
        dates = pd.to_datetime(events[date_col], errors='coerce').dropna()
        
        if not dates.empty:
            min_d, max_d = dates.min().date(), dates.max().date()
            date_range = st.date_input("Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
            
            if len(date_range) == 2:
                mask = (pd.to_datetime(events[date_col]) >= pd.Timestamp(date_range[0])) & \
                       (pd.to_datetime(events[date_col]) <= pd.Timestamp(date_range[1]))
                events_filtered = events[mask].copy()
                period_days = (date_range[1] - date_range[0]).days
            else:
                events_filtered = events.copy()
                period_days = 90
        else:
            events_filtered = events.copy()
            period_days = 90
        
        st.markdown("### 🎯 Scenario")
        target_mult = st.slider("Target ×", 0.5, 2.0, 1.0, 0.1)
        velocity_mult = st.slider("Velocity ×", 0.5, 2.0, 1.0, 0.1)
        
        scenario = {'target_mult': target_mult, 'velocity_mult': velocity_mult}
    
    # ==========================================
    # COMPUTE ANALYTICS
    # ==========================================
    with st.spinner("Analyzing network..."):
        shortfall_df = calculate_shortfalls(events_filtered, period_days=period_days, total_events_df=events, scenario_params=scenario)
        leverage_df = analyze_network_leverage(events_filtered)
        health_df = analyze_network_health(events_filtered)
        
        # Clustering
        cluster_results = run_referral_clustering(events_filtered, resolution=1.5, target_max_clusters=12)
        G = cluster_results.get('graph', nx.Graph())
        builder_master = cluster_results.get('builder_master', pd.DataFrame())
        cluster_summary = cluster_results.get('cluster_summary', pd.DataFrame())
        
        if not builder_master.empty:
            partition_dict = builder_master.set_index('BuilderRegionKey')['ClusterId'].to_dict()
        else:
            partition_dict = {}
    
    all_builders = get_all_builders(events_filtered)
    
    # Render campaign cart in sidebar
    render_campaign_sidebar(all_builders, shortfall_df)
    
    # ==========================================
    # SEARCH BAR
    # ==========================================
    st.markdown("### 🔍 Find a Builder")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        selected = st.selectbox(
            "Search by name",
            options=[""] + all_builders,
            format_func=lambda x: "Type to search..." if x == "" else x,
            key="search_builder",
            label_visibility="collapsed"
        )
    
    with col2:
        add_disabled = not selected or selected in st.session_state.campaign_targets
        if st.button("➕ Add to Cart", disabled=add_disabled, use_container_width=True):
            st.session_state.campaign_targets.add(selected)
            st.rerun()
    
    if selected:
        st.session_state.selected_builder = selected
        # Auto-select cluster
        if selected in partition_dict:
            st.session_state.selected_cluster = partition_dict[selected]
    
    st.markdown("---")
    
    # ==========================================
    # TWO-COLUMN LAYOUT: MAP + DETAILS
    # ==========================================
    map_col, detail_col = st.columns([3, 2])
    
    with map_col:
        st.markdown("### 🗺️ Ecosystem Map")
        
        # Cluster selector
        if not cluster_summary.empty:
            cluster_options = ["All Clusters"] + [f"Cluster {int(c)} ({int(n)} builders)" 
                for c, n in zip(cluster_summary['ClusterId'], cluster_summary['N_builders'])]
            
            cluster_choice = st.selectbox("Focus on cluster", cluster_options, key="cluster_select")
            
            if cluster_choice != "All Clusters":
                st.session_state.selected_cluster = int(cluster_choice.split()[1])
            else:
                st.session_state.selected_cluster = None
        
        # Render map
        if G.number_of_nodes() > 0:
            pos = nx.spring_layout(G, seed=42, k=1.0, iterations=50)
            fig = render_cluster_map(
                G, pos, partition_dict, cluster_summary,
                selected_cluster=st.session_state.selected_cluster,
                selected_builder=st.session_state.selected_builder
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster legend
            if not cluster_summary.empty:
                st.caption("**Clusters:** " + " • ".join([
                    f"<span style='color:{get_cluster_color(int(c))}'>●</span> {int(c)}" 
                    for c in cluster_summary['ClusterId'].head(8)
                ]), unsafe_allow_html=True)
        else:
            st.info("No network connections in selected period")
        
        # Cluster summary table
        if st.session_state.selected_cluster and not cluster_summary.empty:
            st.markdown(f"#### Cluster {st.session_state.selected_cluster} Summary")
            c_sum = get_cluster_summary(st.session_state.selected_cluster, builder_master, leverage_df)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Members", c_sum['member_count'])
            c2.metric("Internal Refs", f"{c_sum['internal_referrals']:,}")
            c3.metric("External Inbound", f"{c_sum['external_inbound']:,}")
            
            # List members with add-to-cart
            with st.expander(f"View {c_sum['member_count']} Members"):
                for m in c_sum.get('members', [])[:20]:
                    cols = st.columns([4, 1])
                    cols[0].write(m)
                    if cols[1].button("➕", key=f"add_{m}"):
                        st.session_state.campaign_targets.add(m)
                        st.rerun()
    
    with detail_col:
        if st.session_state.selected_builder:
            # Get history
            history = get_builder_referral_history(events_filtered, st.session_state.selected_builder)
            shortfall_row = shortfall_df[shortfall_df['BuilderRegionKey'] == st.session_state.selected_builder]
            
            if not shortfall_row.empty:
                shortfall_row = shortfall_row.iloc[0]
            else:
                shortfall_row = None
            
            render_referral_history(history, shortfall_row)
            
            # Leverage paths
            st.markdown("#### 🎯 Leverage Paths")
            st.caption("Best sources to scale media for this builder")
            
            paths = leverage_df[leverage_df['Dest_BuilderRegionKey'] == st.session_state.selected_builder].copy()
            if not paths.empty:
                paths = paths.sort_values('eCPR').head(5)
                paths_display = paths[['MediaPayer_BuilderRegionKey', 'Referrals_to_Target', 'Transfer_Rate', 'eCPR']].copy()
                paths_display.columns = ['Source', 'Historical Refs', 'Transfer Rate', 'eCPR']
                
                st.dataframe(
                    paths_display.style.format({
                        'Transfer Rate': '{:.1%}',
                        'eCPR': '${:,.0f}'
                    }),
                    hide_index=True, use_container_width=True
                )
            else:
                st.caption("No historical referral paths found")
        else:
            st.markdown("### 📊 Builder Details")
            st.info("Select a builder above to see their referral history, sources, and leverage paths.")
            
            # Show quick stats instead
            st.markdown("#### Network Overview")
            c1, c2 = st.columns(2)
            c1.metric("Total Builders", len(all_builders))
            c2.metric("Clusters", len(cluster_summary) if not cluster_summary.empty else 0)
            
            at_risk = len(shortfall_df[shortfall_df['Risk_Score'] > 25])
            c3, c4 = st.columns(2)
            c3.metric("At Risk", at_risk)
            c4.metric("Connections", G.number_of_edges())
    
    st.markdown("---")
    
    # ==========================================
    # CAMPAIGN PLANNER
    # ==========================================
    st.markdown("## 🚀 Generate Campaign Plan")
    
    targets = st.session_state.campaign_targets
    
    if not targets:
        st.info("👆 Add builders to your cart using the search bar or cluster explorer above, then generate an optimized media plan.")
    else:
        st.markdown(f"**Selected Targets:** {len(targets)} builders")
        
        # Budget option
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            use_cap = st.checkbox("Set budget cap")
        with col2:
            budget_cap = st.number_input("Max Budget ($)", 10000, 1000000, 100000, step=10000, disabled=not use_cap)
        with col3:
            generate = st.button("⚡ Generate Plan", type="primary", use_container_width=True)
        
        if generate:
            with st.spinner("Optimizing..."):
                plan_df = generate_targeted_media_plan(
                    target_builders=list(targets),
                    shortfall_df=shortfall_df,
                    leverage_df=leverage_df,
                    budget_cap=budget_cap if use_cap else None
                )
                st.session_state.campaign_plan = plan_df
        
        # Display results
        if 'campaign_plan' in st.session_state and not st.session_state.campaign_plan.empty:
            plan = st.session_state.campaign_plan
            summary = calculate_campaign_summary(plan)
            
            # Summary
            st.markdown("### 📊 Plan Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Budget", fmt_currency(summary['total_budget']))
            m2.metric("Projected Leads", f"{summary['total_leads']:,.0f}")
            m3.metric("Blended CPR", fmt_currency(summary['avg_cpr']))
            m4.metric("Sources to Scale", summary['sources_used'])
            
            # Table
            st.markdown("### 📋 Detailed Plan")
            
            display = plan[['Target_Builder', 'Status', 'Gap_Leads', 'Recommended_Source', 
                           'Budget_Allocation', 'Projected_Leads', 'Effective_CPR', 'Action']].copy()
            
            st.dataframe(
                display.style.format({
                    'Gap_Leads': '{:,.0f}',
                    'Budget_Allocation': '${:,.0f}',
                    'Projected_Leads': '{:,.0f}',
                    'Effective_CPR': '${:,.0f}'
                }),
                hide_index=True, use_container_width=True, height=400
            )
            
            # Download
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download CSV",
                    plan.to_csv(index=False),
                    "campaign_plan.csv",
                    "text/csv",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    "📥 Download Excel",
                    export_to_excel(plan, "campaign_plan.xlsx"),
                    "campaign_plan.xlsx",
                    use_container_width=True
                )
    
    # Footer
    st.markdown("---")
    st.caption(f"Network Intelligence Engine • {len(all_builders):,} builders • {G.number_of_edges():,} referral connections")

if __name__ == "__main__":
    main()