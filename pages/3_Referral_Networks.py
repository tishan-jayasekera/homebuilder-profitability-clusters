"""
Network Intelligence Engine - Unified Referral Network Dashboard
Single-page workflow: Search → Visualize → Plan → Export
"""
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from io import BytesIO

root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data_loader import load_events, export_to_excel
from src.normalization import normalize_events
from src.referral_clusters import run_referral_clustering
from src.utils import fmt_currency
from src.network_optimization import (
    calculate_shortfalls, analyze_network_leverage, generate_targeted_media_plan,
    analyze_network_health, generate_investment_strategies, build_flow_matrix,
    get_builder_ego_network, calculate_campaign_summary
)

# ==========================================
# PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(page_title="Network Intelligence Engine", page_icon="🔗", layout="wide")

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, .stDeployButton { display: none; }

:root {
    --ink: #0f172a; --muted: #64748b; --card: #ffffff;
    --line: #e2e8f0; --accent: #3b82f6; --success: #10b981;
    --warning: #f59e0b; --danger: #ef4444;
}

.page-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 1.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
    color: white;
}
.page-title { font-size: 1.8rem; font-weight: 700; margin: 0; }
.page-subtitle { font-size: 0.95rem; color: #94a3b8; margin: 0.25rem 0 0 0; }

.search-container {
    background: var(--card); border: 2px solid var(--line);
    border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1rem;
}

.metric-row { display: flex; gap: 1rem; margin: 1rem 0; }
.metric-card {
    flex: 1; background: var(--card); border: 1px solid var(--line);
    border-radius: 10px; padding: 1rem; text-align: center;
}
.metric-value { font-size: 1.5rem; font-weight: 700; color: var(--ink); }
.metric-label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }

.health-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid var(--line); border-radius: 12px; padding: 1.25rem;
}
.health-title { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); margin-bottom: 0.5rem; }
.health-value { font-size: 1.1rem; font-weight: 600; color: var(--ink); }

.campaign-panel {
    background: #fefce8; border: 2px solid #fde047;
    border-radius: 12px; padding: 1rem; margin: 1rem 0;
}
.campaign-title { font-weight: 600; color: #854d0e; margin-bottom: 0.5rem; }

.status-critical { color: #dc2626; font-weight: 600; }
.status-high { color: #ea580c; font-weight: 600; }
.status-medium { color: #ca8a04; font-weight: 600; }
.status-ok { color: #16a34a; font-weight: 600; }

.builder-chip {
    display: inline-block; background: #dbeafe; color: #1e40af;
    padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem;
    margin: 0.15rem; font-weight: 500;
}
.builder-chip-remove {
    cursor: pointer; margin-left: 0.5rem; color: #3b82f6;
}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if 'campaign_targets' not in st.session_state:
    st.session_state.campaign_targets = []
if 'selected_builder' not in st.session_state:
    st.session_state.selected_builder = None
if 'ego_depth' not in st.session_state:
    st.session_state.ego_depth = 1

# ==========================================
# DATA LOADING
# ==========================================
@st.cache_data(show_spinner=False)
def load_and_process_data():
    if 'events_file' not in st.session_state:
        return None
    events = load_events(st.session_state['events_file'])
    return normalize_events(events) if events is not None else None

def get_all_builders(events_df):
    builders = set()
    for col in ['Dest_BuilderRegionKey', 'MediaPayer_BuilderRegionKey', 'Origin_BuilderRegionKey']:
        if col in events_df.columns:
            builders.update(events_df[col].dropna().unique())
    return sorted(builders)

# ==========================================
# VISUALIZATION - CURVED EDGE NETWORK
# ==========================================
def create_bezier_edge(x0, y0, x1, y1, curvature=0.2):
    """Create Bezier curve control points for curved edges."""
    mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    ctrl_x = mid_x - dy * curvature
    ctrl_y = mid_y + dx * curvature
    
    t = np.linspace(0, 1, 20)
    bx = (1-t)**2 * x0 + 2*(1-t)*t * ctrl_x + t**2 * x1
    by = (1-t)**2 * y0 + 2*(1-t)*t * ctrl_y + t**2 * y1
    return bx.tolist(), by.tolist()

def create_arrow_marker(x0, y0, x1, y1, size=0.02):
    """Create arrowhead coordinates."""
    dx, dy = x1 - x0, y1 - y0
    length = np.sqrt(dx**2 + dy**2)
    if length == 0:
        return [], []
    dx, dy = dx/length, dy/length
    
    # Arrow tip slightly before target
    tip_x, tip_y = x1 - dx * 0.03, y1 - dy * 0.03
    left_x = tip_x - dx * size - dy * size * 0.5
    left_y = tip_y - dy * size + dx * size * 0.5
    right_x = tip_x - dx * size + dy * size * 0.5
    right_y = tip_y - dy * size - dx * size * 0.5
    
    return [left_x, tip_x, right_x, None], [left_y, tip_y, right_y, None]

def render_ecosystem_map(
    G, pos, flows_df, selected_builder=None, highlight_nodes=None, 
    show_labels=True, height=550
):
    """Render advanced network visualization with curved directed edges."""
    fig = go.Figure()
    
    if highlight_nodes is None:
        highlight_nodes = set(G.nodes())
    
    # Precompute reciprocal pairs
    reciprocal_set = set()
    if not flows_df.empty:
        for _, row in flows_df.iterrows():
            if row.get('is_reciprocal', False):
                reciprocal_set.add((row['source'], row['target']))
                reciprocal_set.add((row['target'], row['source']))
    
    # Draw edges
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data.get('weight', 1)
        
        # Determine if in highlight set
        in_focus = (u in highlight_nodes) and (v in highlight_nodes)
        is_reciprocal = (u, v) in reciprocal_set
        
        # Edge styling
        if not in_focus:
            color, width, opacity = '#e2e8f0', 0.5, 0.3
        elif is_reciprocal:
            color, width, opacity = '#3b82f6', 2 + min(weight/5, 3), 0.9
        else:
            color, width, opacity = '#94a3b8', 1 + min(weight/10, 2), 0.6
        
        # Curved edge
        bx, by = create_bezier_edge(x0, y0, x1, y1, curvature=0.15 if is_reciprocal else 0.1)
        
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode='lines', hoverinfo='skip',
            line=dict(color=color, width=width), opacity=opacity,
            showlegend=False
        ))
        
        # Arrow for direction (only for focused edges)
        if in_focus:
            ax, ay = create_arrow_marker(bx[-3], by[-3], bx[-1], by[-1], size=0.025)
            fig.add_trace(go.Scatter(
                x=ax, y=ay, mode='lines', fill='toself',
                fillcolor=color, line=dict(color=color, width=1),
                hoverinfo='skip', showlegend=False, opacity=opacity
            ))
    
    # Draw nodes
    node_x, node_y, node_color, node_size, node_text, node_line = [], [], [], [], [], []
    
    for node in G.nodes():
        if node not in pos:
            continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        in_highlight = node in highlight_nodes
        is_selected = node == selected_builder
        
        if is_selected:
            color, size, lw = '#dc2626', 28, 4
        elif in_highlight:
            color, size, lw = '#3b82f6', 18, 2
        else:
            color, size, lw = '#cbd5e1', 10, 1
        
        node_color.append(color)
        node_size.append(size)
        node_line.append(lw)
        
        # Hover text
        deg = G.degree(node, weight='weight')
        node_text.append(f"<b>{node}</b><br>Connections: {deg}")
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text' if show_labels else 'markers',
        marker=dict(size=node_size, color=node_color, line=dict(width=node_line, color='white')),
        text=[n[:12] + '...' if len(n) > 12 else n for n in G.nodes()] if show_labels else None,
        textposition='top center', textfont=dict(size=8, color='#475569'),
        hovertext=node_text, hoverinfo='text', showlegend=False
    ))
    
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='white', plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode='closest'
    )
    
    return fig

# ==========================================
# CAMPAIGN CART MANAGEMENT
# ==========================================
def add_to_campaign(builder):
    if builder and builder not in st.session_state.campaign_targets:
        st.session_state.campaign_targets.append(builder)

def remove_from_campaign(builder):
    if builder in st.session_state.campaign_targets:
        st.session_state.campaign_targets.remove(builder)

def clear_campaign():
    st.session_state.campaign_targets = []

def render_campaign_cart():
    """Render the campaign planning cart."""
    targets = st.session_state.campaign_targets
    
    if not targets:
        st.info("🛒 **Campaign Cart Empty** — Search and add builders to start planning.")
        return
    
    st.markdown(f"""
    <div class="campaign-panel">
        <div class="campaign-title">🎯 Campaign Targets ({len(targets)} builders)</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Render chips
    cols = st.columns(min(len(targets), 5))
    for i, t in enumerate(targets):
        with cols[i % 5]:
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"**{t[:15]}**" + ("..." if len(t) > 15 else ""))
            if col2.button("✕", key=f"rm_{t}", help=f"Remove {t}"):
                remove_from_campaign(t)
                st.rerun()
    
    # Clear all
    if st.button("🗑️ Clear All", key="clear_cart"):
        clear_campaign()
        st.rerun()

# ==========================================
# BUILDER HEALTH CARD
# ==========================================
def render_builder_health_card(builder, shortfall_df, leverage_df, health_df):
    """Render contextual health card for selected builder."""
    if not builder:
        return
    
    st.markdown(f"### 📊 {builder}")
    
    # Shortfall metrics
    row = shortfall_df[shortfall_df['BuilderRegionKey'] == builder]
    if row.empty:
        st.warning("No shortfall data for this builder.")
        return
    
    r = row.iloc[0]
    target = int(r.get('LeadTarget', 0)) if pd.notna(r.get('LeadTarget')) else 0
    actual = int(r.get('Actual_Referrals', 0)) if pd.notna(r.get('Actual_Referrals')) else 0
    projected = int(r.get('Projected_Total', 0)) if pd.notna(r.get('Projected_Total')) else 0
    gap = int(r.get('Net_Gap', 0)) if pd.notna(r.get('Net_Gap')) else 0
    risk = int(r.get('Risk_Score', 0)) if pd.notna(r.get('Risk_Score')) else 0
    days = int(r.get('Days_Remaining', 0)) if pd.notna(r.get('Days_Remaining')) else 0
    
    # Status
    if gap >= 0:
        status_class, status_text = 'status-ok', '✅ On Track'
    elif risk > 50:
        status_class, status_text = 'status-critical', '🔴 Critical'
    elif risk > 25:
        status_class, status_text = 'status-high', '🟠 At Risk'
    else:
        status_class, status_text = 'status-medium', '🟡 Monitor'
    
    st.markdown(f'<span class="{status_class}">{status_text}</span>', unsafe_allow_html=True)
    
    # Metrics grid
    c1, c2, c3 = st.columns(3)
    c1.metric("Target", f"{target:,}")
    c2.metric("Projected", f"{projected:,}", delta=f"{gap:+,}")
    c3.metric("Days Left", days)
    
    c4, c5, c6 = st.columns(3)
    c4.metric("Risk Score", risk)
    c5.metric("Actual YTD", f"{actual:,}")
    velocity = r.get('Velocity_LeadsPerDay', 0)
    c6.metric("Velocity", f"{velocity:.1f}/day" if pd.notna(velocity) else "N/A")
    
    # Leverage paths
    st.markdown("#### 🔗 Inbound Leverage Paths")
    paths = leverage_df[leverage_df['Dest_BuilderRegionKey'] == builder].copy()
    
    if paths.empty:
        st.info("No historical referral paths found.")
    else:
        paths = paths.sort_values('eCPR').head(5)
        for _, p in paths.iterrows():
            source = p['MediaPayer_BuilderRegionKey']
            tr = p['Transfer_Rate']
            ecpr = p['eCPR']
            
            st.markdown(f"""
            **{source[:25]}** → Transfer Rate: `{tr:.1%}` | eCPR: `${ecpr:,.0f}`
            """)
    
    # Health diagnosis
    if not health_df.empty:
        h_row = health_df[health_df['Builder'] == builder]
        if not h_row.empty:
            role = h_row['Role'].iloc[0]
            ratio = h_row['Ratio_Give_Take'].iloc[0]
            st.markdown(f"**Network Role:** {role} (Give/Take Ratio: {ratio:.2f})")

# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    # Header
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">🔗 Network Intelligence Engine</h1>
        <p class="page-subtitle">Search → Visualize → Plan → Execute</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    events = load_and_process_data()
    
    if events is None:
        st.warning("⚠️ Please upload events data on the Home page first.")
        st.page_link("app.py", label="← Go to Home", icon="🏠")
        return
    
    # ==========================================
    # SIDEBAR - CONTROLS & SETTINGS
    # ==========================================
    with st.sidebar:
        st.markdown("### ⚙️ Analysis Settings")
        
        # Time period
        date_col = 'lead_date' if 'lead_date' in events.columns else 'RefDate'
        if date_col in events.columns:
            dates = pd.to_datetime(events[date_col], errors='coerce').dropna()
            if not dates.empty:
                min_date, max_date = dates.min().date(), dates.max().date()
                date_range = st.date_input(
                    "Analysis Period",
                    value=(min_date, max_date),
                    min_value=min_date, max_value=max_date
                )
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
        else:
            events_filtered = events.copy()
            period_days = 90
        
        st.divider()
        
        # Scenario parameters
        st.markdown("### 🎲 Scenario Modeling")
        target_mult = st.slider("Target Multiplier", 0.5, 2.0, 1.0, 0.1)
        velocity_mult = st.slider("Velocity Multiplier", 0.5, 2.0, 1.0, 0.1)
        
        scenario_params = {'target_mult': target_mult, 'velocity_mult': velocity_mult}
        
        st.divider()
        
        # Visualization settings
        st.markdown("### 🎨 Visualization")
        show_labels = st.checkbox("Show Node Labels", value=True)
        ego_depth = st.radio("Ego Network Depth", [1, 2], horizontal=True)
        st.session_state.ego_depth = ego_depth
        
        st.divider()
        
        # Budget cap for optimization
        st.markdown("### 💰 Budget Settings")
        use_budget_cap = st.checkbox("Set Budget Cap", value=False)
        budget_cap = st.number_input("Max Budget ($)", 10000, 1000000, 100000, 10000) if use_budget_cap else None
    
    # ==========================================
    # COMPUTE ANALYTICS
    # ==========================================
    with st.spinner("Computing network analytics..."):
        shortfall_df = calculate_shortfalls(
            events_df=events_filtered, period_days=period_days,
            total_events_df=events, scenario_params=scenario_params
        )
        leverage_df = analyze_network_leverage(events_filtered)
        health_df = analyze_network_health(events_filtered)
        flows_df, reciprocal_pairs = build_flow_matrix(events_filtered)
        
        # Run clustering for network structure
        cluster_results = run_referral_clustering(events_filtered, resolution=1.5, target_max_clusters=15)
        G = cluster_results.get('graph', nx.Graph())
        partition = cluster_results.get('builder_master', pd.DataFrame())
        if not partition.empty:
            partition_dict = partition.set_index('BuilderRegionKey')['ClusterId'].to_dict()
        else:
            partition_dict = {}
    
    # ==========================================
    # SEARCH & SELECTION INTERFACE
    # ==========================================
    st.markdown("## 🔍 Find a Builder")
    
    all_builders = get_all_builders(events_filtered)
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        selected = st.selectbox(
            "Search builders by name",
            options=[""] + all_builders,
            format_func=lambda x: "Type to search..." if x == "" else x,
            key="builder_search"
        )
        if selected:
            st.session_state.selected_builder = selected
    
    with col2:
        if st.button("➕ Add to Campaign", disabled=not selected, use_container_width=True):
            add_to_campaign(selected)
            st.rerun()
    
    with col3:
        quick_filter = st.selectbox(
            "Quick Filter",
            ["All", "At Risk", "Healthy Hubs", "Zombies"],
            key="quick_filter"
        )
    
    # Apply quick filter
    if quick_filter != "All" and not health_df.empty:
        if quick_filter == "At Risk":
            at_risk = shortfall_df[shortfall_df['Risk_Score'] > 25]['BuilderRegionKey'].tolist()
            st.info(f"Showing {len(at_risk)} builders at risk")
        elif quick_filter == "Healthy Hubs":
            hubs = health_df[health_df['Role'].str.contains('Hub')]['Builder'].tolist()
            st.info(f"Showing {len(hubs)} healthy hubs")
        elif quick_filter == "Zombies":
            zombies = health_df[health_df['Role'].str.contains('Zombie')]['Builder'].tolist()
            st.info(f"Showing {len(zombies)} zombie nodes")
    
    st.divider()
    
    # ==========================================
    # MAIN CONTENT AREA - TWO COLUMNS
    # ==========================================
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        st.markdown("## 🕸️ Ecosystem Map")
        
        if G.number_of_nodes() == 0:
            st.warning("No network connections found in the selected period.")
        else:
            # Compute layout
            pos = nx.spring_layout(G, seed=42, k=0.8, iterations=50)
            
            # Determine highlight set
            highlight_nodes = None
            if st.session_state.selected_builder:
                ego = get_builder_ego_network(
                    events_filtered, 
                    st.session_state.selected_builder,
                    depth=st.session_state.ego_depth
                )
                highlight_nodes = set([n['id'] for n in ego['nodes']])
            
            # Render map
            fig = render_ecosystem_map(
                G, pos, flows_df,
                selected_builder=st.session_state.selected_builder,
                highlight_nodes=highlight_nodes,
                show_labels=show_labels,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Legend
            st.markdown("""
            <div style="font-size: 0.8rem; color: #64748b;">
            <b>Legend:</b> 
            <span style="color: #3b82f6;">●</span> Blue = Reciprocal flow | 
            <span style="color: #94a3b8;">●</span> Grey = One-way flow | 
            <span style="color: #dc2626;">●</span> Red = Selected builder
            </div>
            """, unsafe_allow_html=True)
    
    with right_col:
        # Builder health card OR campaign cart
        if st.session_state.selected_builder:
            render_builder_health_card(
                st.session_state.selected_builder,
                shortfall_df, leverage_df, health_df
            )
        else:
            st.markdown("### 📋 Select a Builder")
            st.info("Use the search above to explore a builder's network position and leverage paths.")
        
        st.divider()
        
        # Campaign cart (always visible)
        st.markdown("### 🛒 Campaign Cart")
        render_campaign_cart()
    
    st.divider()
    
    # ==========================================
    # CAMPAIGN PLANNER
    # ==========================================
    st.markdown("## 🚀 Campaign Planner")
    
    targets = st.session_state.campaign_targets
    
    if not targets:
        st.info("Add builders to your campaign cart above, then generate an optimized media plan.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("⚡ Generate Optimized Plan", type="primary", use_container_width=True):
                with st.spinner("Optimizing media allocation..."):
                    plan_df = generate_targeted_media_plan(
                        target_builders=targets,
                        shortfall_df=shortfall_df,
                        leverage_df=leverage_df,
                        budget_cap=budget_cap
                    )
                    st.session_state.campaign_plan = plan_df
        
        with col2:
            if 'campaign_plan' in st.session_state and not st.session_state.campaign_plan.empty:
                # Export button
                plan_csv = st.session_state.campaign_plan.to_csv(index=False)
                st.download_button(
                    "📥 Download Plan (CSV)",
                    plan_csv,
                    "campaign_media_plan.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        # Display plan results
        if 'campaign_plan' in st.session_state:
            plan = st.session_state.campaign_plan
            
            if plan.empty:
                st.success("✅ All selected builders are on track! No intervention needed.")
            else:
                # Summary metrics
                summary = calculate_campaign_summary(plan)
                
                st.markdown("### 📊 Plan Summary")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Budget", fmt_currency(summary['total_budget']))
                m2.metric("Projected Leads", f"{summary['total_leads']:,.0f}")
                m3.metric("Avg CPR", fmt_currency(summary['avg_cpr']))
                m4.metric("Sources Used", summary['sources_used'])
                
                # Detailed plan table
                st.markdown("### 📋 Detailed Allocation")
                
                display_cols = [
                    'Target_Builder', 'Status', 'Recommended_Source',
                    'Budget_Allocation', 'Projected_Leads', 'Effective_CPR', 'Strategy'
                ]
                display_cols = [c for c in display_cols if c in plan.columns]
                
                st.dataframe(
                    plan[display_cols].style.format({
                        'Budget_Allocation': '${:,.0f}',
                        'Projected_Leads': '{:,.0f}',
                        'Effective_CPR': '${:,.0f}'
                    }),
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                
                # Visualization - Budget by source
                if 'Recommended_Source' in plan.columns and 'Budget_Allocation' in plan.columns:
                    source_totals = plan.groupby('Recommended_Source')['Budget_Allocation'].sum().sort_values(ascending=True)
                    source_totals = source_totals[source_totals > 0].tail(10)
                    
                    if not source_totals.empty:
                        st.markdown("### 💰 Budget by Source")
                        fig = go.Figure(go.Bar(
                            x=source_totals.values,
                            y=source_totals.index,
                            orientation='h',
                            marker_color='#3b82f6'
                        ))
                        fig.update_layout(
                            height=300,
                            margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="Budget ($)",
                            yaxis_title=""
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================
    # NETWORK HEALTH OVERVIEW (COLLAPSIBLE)
    # ==========================================
    with st.expander("📈 Network Health Overview", expanded=False):
        if health_df.empty:
            st.info("No health data available.")
        else:
            # Summary
            total = len(health_df)
            zombies = len(health_df[health_df['Role'].str.contains('Zombie')])
            hubs = len(health_df[health_df['Role'].str.contains('Hub')])
            feeders = len(health_df[health_df['Role'].str.contains('Feeder')])
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Nodes", total)
            c2.metric("🧟 Zombies", zombies, delta=f"{zombies/total:.0%} of network")
            c3.metric("🔄 Healthy Hubs", hubs)
            c4.metric("📡 Feeders", feeders)
            
            # Role distribution chart
            role_counts = health_df['Role'].value_counts()
            fig = go.Figure(go.Pie(
                labels=role_counts.index,
                values=role_counts.values,
                hole=0.4,
                marker_colors=['#ef4444', '#3b82f6', '#10b981', '#94a3b8']
            ))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.dataframe(
                health_df.sort_values('Leads_Received', ascending=False).head(20),
                use_container_width=True,
                hide_index=True
            )
    
    # ==========================================
    # FOOTER
    # ==========================================
    st.divider()
    st.caption(f"Network Intelligence Engine • {len(all_builders):,} builders • {G.number_of_edges():,} connections")

if __name__ == "__main__":
    main()