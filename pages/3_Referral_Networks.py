"""
Network Intelligence Engine - Unified Referral Network Dashboard
Fixed: session state, column references, deprecation warnings, improved map
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
from src.referral_clusters import run_referral_clustering
from src.utils import fmt_currency
from src.network_optimization import (
    calculate_shortfalls, analyze_network_leverage, generate_targeted_media_plan,
    analyze_network_health, get_builder_referral_history, get_cluster_summary,
    calculate_campaign_summary
)

st.set_page_config(page_title="Network Intelligence", page_icon="🔗", layout="wide")

# ==========================================
# CLUSTER COLORS
# ==========================================
CLUSTER_COLORS = [
    '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
    '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
    '#14b8a6', '#e11d48', '#0ea5e9', '#a855f7', '#22c55e'
]

def get_cluster_color(cluster_id):
    return CLUSTER_COLORS[(cluster_id - 1) % len(CLUSTER_COLORS)]

# ==========================================
# SESSION STATE - Use lists for JSON serialization
# ==========================================
if 'campaign_targets' not in st.session_state:
    st.session_state.campaign_targets = []
if 'selected_builder' not in st.session_state:
    st.session_state.selected_builder = None
if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = None

def add_to_cart(builder):
    if builder and builder not in st.session_state.campaign_targets:
        st.session_state.campaign_targets.append(builder)

def remove_from_cart(builder):
    if builder in st.session_state.campaign_targets:
        st.session_state.campaign_targets.remove(builder)

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
# FLOW ANALYSIS
# ==========================================
def analyze_flows(events_df, focus_builder=None):
    """Analyze directed flows and classify as two-way, inbound-only, outbound-only."""
    refs = events_df[events_df['is_referral'] == True].copy()
    if refs.empty:
        return pd.DataFrame(), {}
    
    # Aggregate flows
    flows = refs.groupby(['MediaPayer_BuilderRegionKey', 'Dest_BuilderRegionKey']).agg(
        count=('LeadId', 'count'),
        value=('MediaCost_referral_event', 'sum')
    ).reset_index()
    flows.columns = ['source', 'target', 'count', 'value']
    
    # Detect reciprocal pairs
    flow_set = set(zip(flows['source'], flows['target']))
    
    def classify_flow(row):
        s, t = row['source'], row['target']
        reverse_exists = (t, s) in flow_set
        if reverse_exists:
            return 'two_way'
        return 'one_way'
    
    flows['flow_type'] = flows.apply(classify_flow, axis=1)
    
    # If focus builder, classify relative to them
    if focus_builder:
        def classify_relative(row):
            s, t = row['source'], row['target']
            if row['flow_type'] == 'two_way':
                return 'two_way'
            elif t == focus_builder:
                return 'inbound'
            elif s == focus_builder:
                return 'outbound'
            else:
                return 'other'
        flows['relative_type'] = flows.apply(classify_relative, axis=1)
    else:
        flows['relative_type'] = flows['flow_type']
    
    # Summary stats
    summary = {
        'total_flows': len(flows),
        'two_way': len(flows[flows['flow_type'] == 'two_way']),
        'one_way': len(flows[flows['flow_type'] == 'one_way'])
    }
    
    return flows, summary

# ==========================================
# IMPROVED NETWORK MAP WITH ARROWS
# ==========================================
def render_network_map(G, pos, partition_dict, flows_df, selected_cluster=None, selected_builder=None):
    """Render network with clear directional arrows and flow type coloring."""
    fig = go.Figure()
    
    # Flow type colors
    COLORS = {
        'two_way': '#3b82f6',      # Blue - strong partnership
        'inbound': '#10b981',      # Green - receiving
        'outbound': '#f59e0b',     # Orange - sending
        'other': '#cbd5e1',        # Grey - not connected to focus
        'dimmed': '#e2e8f0'        # Light grey - outside cluster
    }
    
    # Build flow lookup
    flow_lookup = {}
    if not flows_df.empty:
        for _, row in flows_df.iterrows():
            key = (row['source'], row['target'])
            flow_lookup[key] = {
                'count': row['count'],
                'type': row.get('relative_type', 'other')
            }
    
    # Draw edges with arrows
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data.get('weight', 1)
        
        # Get flow info
        flow_info = flow_lookup.get((u, v), {'count': weight, 'type': 'other'})
        flow_type = flow_info['type']
        
        # Determine visibility based on cluster selection
        u_cluster = partition_dict.get(u, 0)
        v_cluster = partition_dict.get(v, 0)
        
        in_focus = True
        if selected_cluster:
            in_focus = (u_cluster == selected_cluster or v_cluster == selected_cluster)
        
        if not in_focus:
            color = COLORS['dimmed']
            opacity = 0.1
            width = 0.5
        else:
            color = COLORS.get(flow_type, COLORS['other'])
            opacity = 0.7 if flow_type != 'other' else 0.3
            width = 1 + min(weight / 8, 3)
        
        # Calculate arrow direction
        dx, dy = x1 - x0, y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            continue
        
        # Normalize
        udx, udy = dx / length, dy / length
        
        # Shorten line to make room for arrowhead
        arrow_len = 0.03
        end_x = x1 - udx * arrow_len * 1.5
        end_y = y1 - udy * arrow_len * 1.5
        
        # Draw edge line
        fig.add_trace(go.Scatter(
            x=[x0, end_x], y=[y0, end_y],
            mode='lines', hoverinfo='skip',
            line=dict(color=color, width=width),
            opacity=opacity, showlegend=False
        ))
        
        # Draw arrowhead (triangle) if in focus
        if in_focus and opacity > 0.2:
            # Arrowhead size based on weight
            head_size = 0.015 + min(weight / 200, 0.015)
            
            # Perpendicular vector
            px, py = -udy, udx
            
            # Arrow points
            tip_x, tip_y = x1 - udx * 0.02, y1 - udy * 0.02
            left_x = tip_x - udx * head_size - px * head_size * 0.6
            left_y = tip_y - udy * head_size - py * head_size * 0.6
            right_x = tip_x - udx * head_size + px * head_size * 0.6
            right_y = tip_y - udy * head_size + py * head_size * 0.6
            
            fig.add_trace(go.Scatter(
                x=[left_x, tip_x, right_x, left_x],
                y=[left_y, tip_y, right_y, left_y],
                mode='lines', fill='toself',
                fillcolor=color, line=dict(color=color, width=1),
                opacity=opacity, hoverinfo='skip', showlegend=False
            ))
    
    # Draw nodes
    for cluster_id in set(partition_dict.values()):
        cluster_nodes = [n for n, c in partition_dict.items() if c == cluster_id and n in pos]
        if not cluster_nodes:
            continue
        
        # Cluster visibility
        if selected_cluster and cluster_id != selected_cluster:
            node_color = '#e2e8f0'
            opacity = 0.3
        else:
            node_color = get_cluster_color(cluster_id)
            opacity = 1.0
        
        node_x = [pos[n][0] for n in cluster_nodes]
        node_y = [pos[n][1] for n in cluster_nodes]
        
        # Sizes and colors
        sizes = []
        colors = []
        borders = []
        
        for n in cluster_nodes:
            deg = G.degree(n, weight='weight')
            base_size = 14 + min(deg / 3, 18)
            
            if n == selected_builder:
                sizes.append(40)
                colors.append('#dc2626')  # Red for selected
                borders.append(4)
            else:
                sizes.append(base_size)
                colors.append(node_color)
                borders.append(2)
        
        hover_text = []
        for n in cluster_nodes:
            inbound = sum(1 for s, t in G.edges() if t == n)
            outbound = sum(1 for s, t in G.edges() if s == n)
            hover_text.append(f"<b>{n}</b><br>Cluster {cluster_id}<br>In: {inbound} | Out: {outbound}")
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode='markers',
            marker=dict(size=sizes, color=colors, opacity=opacity, line=dict(width=borders, color='white')),
            text=hover_text, hoverinfo='text',
            name=f"Cluster {cluster_id}" if not selected_cluster or cluster_id == selected_cluster else "",
            showlegend=not selected_cluster or cluster_id == selected_cluster
        ))
    
    # Legend for flow types
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=COLORS['two_way']), name='↔ Two-Way'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=COLORS['inbound']), name='→ Inbound'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=COLORS['outbound']), name='← Outbound'))
    
    fig.update_layout(
        height=520, margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor='#fafafa', plot_bgcolor='#fafafa',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='y'),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, font=dict(size=11)),
        hovermode='closest'
    )
    
    return fig

# ==========================================
# REFERRAL HISTORY
# ==========================================
def render_referral_history(history, shortfall_row):
    """Render operational referral history for a builder."""
    st.markdown(f"### 📋 {history['builder']}")
    
    # Status banner
    if shortfall_row is not None:
        gap = shortfall_row.get('Net_Gap', 0)
        risk = shortfall_row.get('Risk_Score', 0)
        gap = 0 if pd.isna(gap) else gap
        risk = 0 if pd.isna(risk) else risk
        
        if gap >= 0:
            st.success(f"✅ **On Track** — Surplus of {int(gap)} leads projected")
        elif risk > 50:
            st.error(f"🔴 **Critical** — {int(abs(gap))} lead shortfall | Risk: {int(risk)}")
        elif risk > 25:
            st.warning(f"🟠 **At Risk** — {int(abs(gap))} lead shortfall | Risk: {int(risk)}")
        else:
            st.info(f"🟡 **Monitor** — {int(abs(gap))} lead shortfall")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Received", f"{history['total_received']:,}")
    c2.metric("Sent", f"{history['total_sent']:,}")
    c3.metric("Sources", history['unique_sources'])
    c4.metric("Destinations", history['unique_destinations'])
    
    # Monthly trend
    monthly = history.get('monthly_trend', pd.DataFrame())
    if not monthly.empty:
        st.markdown("#### 📈 Monthly Inbound Trend")
        fig = go.Figure(go.Bar(x=monthly['month'], y=monthly['referrals'], marker_color='#3b82f6'))
        fig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=30), plot_bgcolor='white', xaxis_title='', yaxis_title='')
        st.plotly_chart(fig, width='stretch')
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔗 Top Sources")
        sources = history.get('top_sources', pd.DataFrame())
        if not sources.empty:
            st.dataframe(sources.head(5).style.format({'Media Value': '${:,.0f}'}), hide_index=True, width='stretch')
        else:
            st.caption("No inbound referrals")
    
    with col2:
        st.markdown("#### 📤 Top Destinations")
        dests = history.get('top_destinations', pd.DataFrame())
        if not dests.empty:
            st.dataframe(dests.head(5), hide_index=True, width='stretch')
        else:
            st.caption("No outbound referrals")

# ==========================================
# CAMPAIGN CART
# ==========================================
def render_campaign_sidebar(shortfall_df):
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🛒 Campaign Cart")
    
    targets = st.session_state.campaign_targets
    
    if not targets:
        st.sidebar.info("No builders selected")
        return
    
    st.sidebar.markdown(f"**{len(targets)} builder(s)**")
    
    for t in list(targets):
        row = shortfall_df[shortfall_df['BuilderRegionKey'] == t]
        if not row.empty:
            risk = row['Risk_Score'].iloc[0]
            risk = 0 if pd.isna(risk) else risk
            icon = "🔴" if risk > 50 else ("🟠" if risk > 25 else "🟢")
        else:
            icon = "⚪"
        
        col1, col2 = st.sidebar.columns([4, 1])
        col1.markdown(f"{icon} {t[:18]}")
        if col2.button("✕", key=f"rm_{t}"):
            remove_from_cart(t)
            st.rerun()
    
    if st.sidebar.button("🗑️ Clear All"):
        st.session_state.campaign_targets = []
        st.rerun()
    
    at_risk = sum(1 for t in targets if not shortfall_df[(shortfall_df['BuilderRegionKey'] == t) & (shortfall_df['Risk_Score'] > 25)].empty)
    st.sidebar.metric("At Risk", f"{at_risk} / {len(targets)}")

# ==========================================
# MAIN
# ==========================================
def main():
    st.title("🔗 Network Intelligence Engine")
    st.caption("Understand referral flows • Identify leverage • Plan campaigns")
    
    events = load_and_process()
    if events is None:
        st.warning("⚠️ Upload events data on the Home page first.")
        st.page_link("app.py", label="← Go to Home", icon="🏠")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        date_col = 'lead_date' if 'lead_date' in events.columns else 'RefDate'
        dates = pd.to_datetime(events[date_col], errors='coerce').dropna()
        
        if not dates.empty:
            min_d, max_d = dates.min().date(), dates.max().date()
            date_range = st.date_input("Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
            if len(date_range) == 2:
                mask = (pd.to_datetime(events[date_col]) >= pd.Timestamp(date_range[0])) & (pd.to_datetime(events[date_col]) <= pd.Timestamp(date_range[1]))
                events_filtered = events[mask].copy()
                period_days = (date_range[1] - date_range[0]).days
            else:
                events_filtered, period_days = events.copy(), 90
        else:
            events_filtered, period_days = events.copy(), 90
        
        st.markdown("### 🎯 Scenario")
        target_mult = st.slider("Target ×", 0.5, 2.0, 1.0, 0.1)
        velocity_mult = st.slider("Velocity ×", 0.5, 2.0, 1.0, 0.1)
        scenario = {'target_mult': target_mult, 'velocity_mult': velocity_mult}
    
    # Analytics
    with st.spinner("Analyzing network..."):
        shortfall_df = calculate_shortfalls(events_filtered, period_days=period_days, total_events_df=events, scenario_params=scenario)
        leverage_df = analyze_network_leverage(events_filtered)
        health_df = analyze_network_health(events_filtered)
        
        cluster_results = run_referral_clustering(events_filtered, resolution=1.5, target_max_clusters=12)
        G = cluster_results.get('graph', nx.Graph())
        builder_master = cluster_results.get('builder_master', pd.DataFrame())
        cluster_summary = cluster_results.get('cluster_summary', pd.DataFrame())
        partition_dict = builder_master.set_index('BuilderRegionKey')['ClusterId'].to_dict() if not builder_master.empty else {}
        
        flows_df, flow_summary = analyze_flows(events_filtered, st.session_state.selected_builder)
    
    all_builders = get_all_builders(events_filtered)
    render_campaign_sidebar(shortfall_df)
    
    # Search
    st.markdown("### 🔍 Find a Builder")
    col1, col2 = st.columns([4, 1])
    with col1:
        selected = st.selectbox("Search by name", options=[""] + all_builders, format_func=lambda x: "Type to search..." if x == "" else x, key="search_builder", label_visibility="collapsed")
    with col2:
        add_disabled = not selected or selected in st.session_state.campaign_targets
        if st.button("➕ Add", disabled=add_disabled):
            add_to_cart(selected)
            st.rerun()
    
    if selected:
        st.session_state.selected_builder = selected
        if selected in partition_dict:
            st.session_state.selected_cluster = partition_dict[selected]
        # Re-analyze flows with focus
        flows_df, _ = analyze_flows(events_filtered, selected)
    
    st.markdown("---")
    
    # Two columns
    map_col, detail_col = st.columns([3, 2])
    
    with map_col:
        st.markdown("### 🗺️ Referral Ecosystem")
        
        if not cluster_summary.empty:
            cluster_opts = ["All Clusters"] + [f"Cluster {int(c)} ({int(n)} members)" for c, n in zip(cluster_summary['ClusterId'], cluster_summary['N_builders'])]
            cluster_choice = st.selectbox("Focus", cluster_opts, key="cluster_sel")
            st.session_state.selected_cluster = int(cluster_choice.split()[1]) if cluster_choice != "All Clusters" else None
        
        if G.number_of_nodes() > 0:
            pos = nx.spring_layout(G, seed=42, k=1.2, iterations=60)
            fig = render_network_map(G, pos, partition_dict, flows_df, st.session_state.selected_cluster, st.session_state.selected_builder)
            st.plotly_chart(fig, width='stretch')
            
            # Flow summary
            if st.session_state.selected_builder:
                builder_flows = flows_df[(flows_df['source'] == st.session_state.selected_builder) | (flows_df['target'] == st.session_state.selected_builder)]
                two_way = len(builder_flows[builder_flows['relative_type'] == 'two_way']) // 2
                inbound = len(builder_flows[builder_flows['relative_type'] == 'inbound'])
                outbound = len(builder_flows[builder_flows['relative_type'] == 'outbound'])
                st.caption(f"**{st.session_state.selected_builder}**: {two_way} two-way partnerships | {inbound} inbound sources | {outbound} outbound destinations")
        else:
            st.info("No network connections in selected period")
        
        # Cluster members
        if st.session_state.selected_cluster and not cluster_summary.empty:
            c_sum = get_cluster_summary(st.session_state.selected_cluster, builder_master, leverage_df)
            st.markdown(f"#### Cluster {st.session_state.selected_cluster}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Members", c_sum['member_count'])
            c2.metric("Internal Refs", f"{c_sum['internal_referrals']:,}")
            c3.metric("External In", f"{c_sum['external_inbound']:,}")
            
            with st.expander(f"View Members ({c_sum['member_count']})"):
                for m in c_sum.get('members', [])[:20]:
                    cols = st.columns([4, 1])
                    cols[0].write(m)
                    if cols[1].button("➕", key=f"add_{m}"):
                        add_to_cart(m)
                        st.rerun()
    
    with detail_col:
        if st.session_state.selected_builder:
            history = get_builder_referral_history(events_filtered, st.session_state.selected_builder)
            row = shortfall_df[shortfall_df['BuilderRegionKey'] == st.session_state.selected_builder]
            shortfall_row = row.iloc[0].to_dict() if not row.empty else None
            render_referral_history(history, shortfall_row)
            
            st.markdown("#### 🎯 Leverage Paths")
            paths = leverage_df[leverage_df['Dest_BuilderRegionKey'] == st.session_state.selected_builder].copy()
            if not paths.empty:
                paths = paths.sort_values('eCPR').head(5)
                disp = paths[['MediaPayer_BuilderRegionKey', 'Referrals_to_Target', 'Transfer_Rate', 'eCPR']].copy()
                disp.columns = ['Source', 'Hist. Refs', 'TR', 'eCPR']
                st.dataframe(disp.style.format({'TR': '{:.1%}', 'eCPR': '${:,.0f}'}), hide_index=True, width='stretch')
            else:
                st.caption("No historical paths")
        else:
            st.markdown("### 📊 Network Overview")
            st.info("Select a builder to see details")
            c1, c2 = st.columns(2)
            c1.metric("Builders", len(all_builders))
            c2.metric("Clusters", len(cluster_summary) if not cluster_summary.empty else 0)
            c3, c4 = st.columns(2)
            c3.metric("At Risk", len(shortfall_df[shortfall_df['Risk_Score'] > 25]))
            c4.metric("Connections", G.number_of_edges())
    
    st.markdown("---")
    
    # Campaign Planner
    st.markdown("## 🚀 Generate Campaign Plan")
    targets = st.session_state.campaign_targets
    
    if not targets:
        st.info("👆 Add builders to your cart, then generate an optimized media plan.")
    else:
        st.markdown(f"**{len(targets)} targets selected**")
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            use_cap = st.checkbox("Budget cap")
        with col2:
            budget_cap = st.number_input("Max ($)", 10000, 1000000, 100000, step=10000, disabled=not use_cap)
        with col3:
            generate = st.button("⚡ Generate", type="primary")
        
        if generate:
            with st.spinner("Optimizing..."):
                plan_df = generate_targeted_media_plan(list(targets), shortfall_df, leverage_df, budget_cap if use_cap else None)
                st.session_state.campaign_plan = plan_df
        
        if 'campaign_plan' in st.session_state and not st.session_state.campaign_plan.empty:
            plan = st.session_state.campaign_plan
            summary = calculate_campaign_summary(plan)
            
            st.markdown("### 📊 Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Budget", fmt_currency(summary['total_budget']))
            m2.metric("Leads", f"{summary['total_leads']:,.0f}")
            m3.metric("Blended CPR", fmt_currency(summary['avg_cpr']))
            m4.metric("Sources", summary['sources_used'])
            
            st.markdown("### 📋 Plan")
            
            # Only include columns that exist
            avail_cols = plan.columns.tolist()
            display_cols = [c for c in ['Target_Builder', 'Status', 'Gap_Leads', 'Recommended_Source', 'Budget_Allocation', 'Projected_Leads', 'Effective_CPR'] if c in avail_cols]
            
            fmt_dict = {}
            if 'Gap_Leads' in display_cols:
                fmt_dict['Gap_Leads'] = '{:,.0f}'
            if 'Budget_Allocation' in display_cols:
                fmt_dict['Budget_Allocation'] = '${:,.0f}'
            if 'Projected_Leads' in display_cols:
                fmt_dict['Projected_Leads'] = '{:,.0f}'
            if 'Effective_CPR' in display_cols:
                fmt_dict['Effective_CPR'] = '${:,.0f}'
            
            st.dataframe(plan[display_cols].style.format(fmt_dict), hide_index=True, width='stretch', height=350)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 CSV", plan.to_csv(index=False), "campaign_plan.csv", "text/csv")
            with col2:
                st.download_button("📥 Excel", export_to_excel(plan, "plan.xlsx"), "campaign_plan.xlsx")
    
    st.markdown("---")
    st.caption(f"Network Intelligence • {len(all_builders):,} builders • {G.number_of_edges():,} connections")

if __name__ == "__main__":
    main()