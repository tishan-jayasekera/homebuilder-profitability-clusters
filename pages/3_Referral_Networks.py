"""
Network Intelligence Engine
Commercial focus: Who sends you leads? Who do you send to? What are the strongest partnerships?
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
# SESSION STATE
# ==========================================
if 'campaign_targets' not in st.session_state:
    st.session_state.campaign_targets = []
if 'selected_builder' not in st.session_state:
    st.session_state.selected_builder = None

def add_to_cart(builder):
    if builder and builder not in st.session_state.campaign_targets:
        st.session_state.campaign_targets.append(builder)

def remove_from_cart(builder):
    if builder in st.session_state.campaign_targets:
        st.session_state.campaign_targets.remove(builder)

# ==========================================
# DATA
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
def get_builder_connections(events_df, builder):
    """Get direct connections for a builder with flow classification."""
    refs = events_df[events_df['is_referral'] == True].copy()
    if refs.empty:
        return {'inbound': [], 'outbound': [], 'two_way': []}
    
    # Inbound: others -> builder
    inbound_df = refs[refs['Dest_BuilderRegionKey'] == builder].groupby('MediaPayer_BuilderRegionKey').agg(
        count=('LeadId', 'count'),
        value=('MediaCost_referral_event', 'sum')
    ).reset_index()
    inbound_df.columns = ['partner', 'refs_in', 'value_in']
    
    # Outbound: builder -> others
    outbound_df = refs[refs['MediaPayer_BuilderRegionKey'] == builder].groupby('Dest_BuilderRegionKey').agg(
        count=('LeadId', 'count')
    ).reset_index()
    outbound_df.columns = ['partner', 'refs_out']
    
    # Merge to find two-way
    merged = pd.merge(inbound_df, outbound_df, on='partner', how='outer').fillna(0)
    
    two_way = merged[(merged['refs_in'] > 0) & (merged['refs_out'] > 0)]['partner'].tolist()
    inbound_only = merged[(merged['refs_in'] > 0) & (merged['refs_out'] == 0)]['partner'].tolist()
    outbound_only = merged[(merged['refs_in'] == 0) & (merged['refs_out'] > 0)]['partner'].tolist()
    
    # Build detailed lists
    result = {
        'two_way': [],
        'inbound': [],
        'outbound': []
    }
    
    for _, row in merged.iterrows():
        partner = row['partner']
        if partner in two_way:
            result['two_way'].append({'partner': partner, 'in': int(row['refs_in']), 'out': int(row['refs_out'])})
        elif partner in inbound_only:
            result['inbound'].append({'partner': partner, 'count': int(row['refs_in']), 'value': row['value_in']})
        elif partner in outbound_only:
            result['outbound'].append({'partner': partner, 'count': int(row['refs_out'])})
    
    # Sort by volume
    result['two_way'] = sorted(result['two_way'], key=lambda x: x['in'] + x['out'], reverse=True)
    result['inbound'] = sorted(result['inbound'], key=lambda x: x['count'], reverse=True)
    result['outbound'] = sorted(result['outbound'], key=lambda x: x['count'], reverse=True)
    
    return result

# ==========================================
# NETWORK MAP - FULL GRAPH WITH HIGHLIGHTED CONNECTIONS
# ==========================================
def render_network_map(G, pos, connections, selected_builder=None):
    """
    Render full network graph.
    - When no builder selected: show all nodes/edges in neutral colors
    - When builder selected: highlight their connections, dim everything else
    """
    fig = go.Figure()
    
    # Colors
    COL_TWO_WAY = '#2563eb'    # Blue - mutual partnerships
    COL_INBOUND = '#16a34a'    # Green - sends TO selected
    COL_OUTBOUND = '#f59e0b'   # Orange - receives FROM selected
    COL_SELECTED = '#dc2626'   # Red - the selected builder
    COL_MUTED = '#e5e7eb'      # Light grey - background
    COL_NODE_DEFAULT = '#6366f1'  # Purple - default node color
    
    # Build connection sets for selected builder
    two_way_set = set()
    inbound_set = set()
    outbound_set = set()
    
    if selected_builder and connections:
        two_way_set = {c['partner'] for c in connections.get('two_way', [])}
        inbound_set = {c['partner'] for c in connections.get('inbound', [])}
        outbound_set = {c['partner'] for c in connections.get('outbound', [])}
    
    connected_nodes = two_way_set | inbound_set | outbound_set
    if selected_builder:
        connected_nodes.add(selected_builder)
    
    # Helper to draw arrow
    def draw_edge(x0, y0, x1, y1, color, width, opacity, show_arrow=True):
        dx, dy = x1 - x0, y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        if length < 0.01:
            return
        udx, udy = dx / length, dy / length
        
        # Shorten line for node margins
        margin = 0.04
        sx, sy = x0 + udx * margin, y0 + udy * margin
        ex, ey = x1 - udx * margin, y1 - udy * margin
        
        # Draw line
        fig.add_trace(go.Scatter(
            x=[sx, ex], y=[sy, ey], mode='lines',
            line=dict(color=color, width=width),
            opacity=opacity, hoverinfo='skip', showlegend=False
        ))
        
        # Draw arrowhead
        if show_arrow and opacity > 0.3:
            head = 0.025
            px, py = -udy, udx
            tip_x, tip_y = ex, ey
            l_x = tip_x - udx * head - px * head * 0.6
            l_y = tip_y - udy * head - py * head * 0.6
            r_x = tip_x - udx * head + px * head * 0.6
            r_y = tip_y - udy * head + py * head * 0.6
            
            fig.add_trace(go.Scatter(
                x=[l_x, tip_x, r_x], y=[l_y, tip_y, r_y],
                mode='lines', fill='toself', fillcolor=color,
                line=dict(color=color, width=0.5),
                opacity=opacity, hoverinfo='skip', showlegend=False
            ))
    
    # Draw all edges
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data.get('weight', 1)
        
        if selected_builder:
            # Determine edge type relative to selected builder
            if u == selected_builder and v in two_way_set:
                # Two-way: draw both directions
                draw_edge(x0, y0, x1, y1, COL_TWO_WAY, 2 + min(weight/5, 3), 0.9)
            elif v == selected_builder and u in two_way_set:
                draw_edge(x0, y0, x1, y1, COL_TWO_WAY, 2 + min(weight/5, 3), 0.9)
            elif u in inbound_set and v == selected_builder:
                # Inbound: partner -> selected
                draw_edge(x0, y0, x1, y1, COL_INBOUND, 1.5 + min(weight/8, 2.5), 0.85)
            elif u == selected_builder and v in outbound_set:
                # Outbound: selected -> partner
                draw_edge(x0, y0, x1, y1, COL_OUTBOUND, 1.5 + min(weight/8, 2.5), 0.85)
            else:
                # Background edge
                draw_edge(x0, y0, x1, y1, COL_MUTED, 0.5, 0.15, show_arrow=False)
        else:
            # No selection - show all edges muted
            draw_edge(x0, y0, x1, y1, '#9ca3af', 0.5 + min(weight/15, 1.5), 0.4, show_arrow=False)
    
    # Draw nodes
    all_nodes = list(G.nodes())
    
    if selected_builder:
        # Background nodes (not connected)
        bg_nodes = [n for n in all_nodes if n not in connected_nodes and n in pos]
        if bg_nodes:
            fig.add_trace(go.Scatter(
                x=[pos[n][0] for n in bg_nodes],
                y=[pos[n][1] for n in bg_nodes],
                mode='markers',
                marker=dict(size=8, color=COL_MUTED, opacity=0.3, line=dict(width=0)),
                hovertext=[f"{n}" for n in bg_nodes],
                hoverinfo='text', showlegend=False
            ))
        
        # Two-way partners
        tw_nodes = [n for n in two_way_set if n in pos]
        if tw_nodes:
            sizes = [18 + min(G.degree(n, weight='weight')/3, 15) for n in tw_nodes]
            fig.add_trace(go.Scatter(
                x=[pos[n][0] for n in tw_nodes],
                y=[pos[n][1] for n in tw_nodes],
                mode='markers',
                marker=dict(size=sizes, color=COL_TWO_WAY, line=dict(width=2, color='white')),
                hovertext=[f"<b>{n}</b><br>↔ Two-way partner" for n in tw_nodes],
                hoverinfo='text', name='↔ Two-Way', showlegend=True
            ))
        
        # Inbound sources
        in_nodes = [n for n in inbound_set if n in pos]
        if in_nodes:
            sizes = [16 + min(G.degree(n, weight='weight')/4, 12) for n in in_nodes]
            fig.add_trace(go.Scatter(
                x=[pos[n][0] for n in in_nodes],
                y=[pos[n][1] for n in in_nodes],
                mode='markers',
                marker=dict(size=sizes, color=COL_INBOUND, line=dict(width=2, color='white')),
                hovertext=[f"<b>{n}</b><br>→ Sends you leads" for n in in_nodes],
                hoverinfo='text', name='→ Inbound', showlegend=True
            ))
        
        # Outbound destinations
        out_nodes = [n for n in outbound_set if n in pos]
        if out_nodes:
            sizes = [16 + min(G.degree(n, weight='weight')/4, 12) for n in out_nodes]
            fig.add_trace(go.Scatter(
                x=[pos[n][0] for n in out_nodes],
                y=[pos[n][1] for n in out_nodes],
                mode='markers',
                marker=dict(size=sizes, color=COL_OUTBOUND, line=dict(width=2, color='white')),
                hovertext=[f"<b>{n}</b><br>← You send leads" for n in out_nodes],
                hoverinfo='text', name='← Outbound', showlegend=True
            ))
        
        # Selected builder (on top)
        if selected_builder in pos:
            fig.add_trace(go.Scatter(
                x=[pos[selected_builder][0]],
                y=[pos[selected_builder][1]],
                mode='markers+text',
                marker=dict(size=35, color=COL_SELECTED, line=dict(width=3, color='white')),
                text=[selected_builder[:12]],
                textposition='top center',
                textfont=dict(size=10, color='#1f2937', family='Arial'),
                hovertext=f"<b>{selected_builder}</b><br>(Selected)",
                hoverinfo='text', showlegend=False
            ))
    else:
        # No selection - show all nodes by degree
        degrees = dict(G.degree(weight='weight'))
        nodes_in_pos = [n for n in all_nodes if n in pos]
        sizes = [10 + min(degrees.get(n, 0) / 4, 20) for n in nodes_in_pos]
        
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nodes_in_pos],
            y=[pos[n][1] for n in nodes_in_pos],
            mode='markers',
            marker=dict(size=sizes, color=COL_NODE_DEFAULT, opacity=0.7, line=dict(width=1, color='white')),
            hovertext=[f"<b>{n}</b><br>Connections: {degrees.get(n, 0)}" for n in nodes_in_pos],
            hoverinfo='text', showlegend=False
        ))
    
    fig.update_layout(
        height=480,
        margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='y'),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='center', x=0.5, font=dict(size=10)),
        hovermode='closest'
    )
    
    return fig

# ==========================================
# BUILDER DETAIL PANEL
# ==========================================
def render_builder_panel(builder, connections, shortfall_df, leverage_df, events_filtered):
    """Render commercial intelligence panel for selected builder."""
    
    # Status
    row = shortfall_df[shortfall_df['BuilderRegionKey'] == builder]
    if not row.empty:
        r = row.iloc[0]
        gap = r.get('Net_Gap', 0)
        risk = r.get('Risk_Score', 0)
        target = r.get('LeadTarget', 0)
        actual = r.get('Actual_Referrals', 0)
        gap = 0 if pd.isna(gap) else gap
        risk = 0 if pd.isna(risk) else risk
        
        if gap >= 0:
            st.success(f"✅ **On Track** — {int(gap)} lead surplus projected")
        elif risk > 50:
            st.error(f"🔴 **Critical** — {int(abs(gap))} lead gap | Risk: {int(risk)}")
        elif risk > 25:
            st.warning(f"🟠 **At Risk** — {int(abs(gap))} lead gap | Risk: {int(risk)}")
        else:
            st.info(f"🟡 **Monitor** — {int(abs(gap))} lead gap")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Target", f"{int(target):,}" if not pd.isna(target) else "—")
        c2.metric("Actual", f"{int(actual):,}" if not pd.isna(actual) else "—")
        c3.metric("Gap", f"{int(gap):+,}" if not pd.isna(gap) else "—")
    
    st.markdown("---")
    
    # Connection summary
    n_two = len(connections['two_way'])
    n_in = len(connections['inbound'])
    n_out = len(connections['outbound'])
    
    st.markdown("#### Referral Relationships")
    c1, c2, c3 = st.columns(3)
    c1.metric("↔ Two-Way", n_two, help="Strong partnerships with mutual referrals")
    c2.metric("→ Inbound", n_in, help="Partners who send you leads")
    c3.metric("← Outbound", n_out, help="Partners you send leads to")
    
    # Top sources (for scaling media)
    st.markdown("#### 🎯 Best Sources to Scale")
    st.caption("Partners who send you leads, ranked by efficiency (lowest eCPR first)")
    
    paths = leverage_df[leverage_df['Dest_BuilderRegionKey'] == builder].copy()
    if not paths.empty:
        paths = paths.sort_values('eCPR').head(5)
        disp = paths[['MediaPayer_BuilderRegionKey', 'Referrals_to_Target', 'Transfer_Rate', 'eCPR']].copy()
        disp.columns = ['Source', 'Historical Refs', 'Transfer Rate', 'Cost/Lead']
        st.dataframe(
            disp.style.format({'Transfer Rate': '{:.1%}', 'Cost/Lead': '${:,.0f}'}),
            hide_index=True, width='stretch', height=200
        )
    else:
        st.caption("No historical referral sources found")
    
    # Two-way partnerships (valuable relationships)
    if connections['two_way']:
        st.markdown("#### 🤝 Two-Way Partnerships")
        st.caption("Mutual referral relationships — your strongest network ties")
        tw_df = pd.DataFrame(connections['two_way'])
        tw_df.columns = ['Partner', 'Refs Received', 'Refs Sent']
        tw_df['Net'] = tw_df['Refs Received'] - tw_df['Refs Sent']
        st.dataframe(tw_df.head(5), hide_index=True, width='stretch')

# ==========================================
# CAMPAIGN SIDEBAR
# ==========================================
def render_sidebar_cart(shortfall_df):
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🛒 Campaign Cart")
    
    targets = st.session_state.campaign_targets
    
    if not targets:
        st.sidebar.caption("Add builders to plan a campaign")
        return
    
    st.sidebar.markdown(f"**{len(targets)} selected**")
    
    for t in list(targets):
        row = shortfall_df[shortfall_df['BuilderRegionKey'] == t]
        if not row.empty:
            risk = row['Risk_Score'].iloc[0]
            risk = 0 if pd.isna(risk) else risk
            icon = "🔴" if risk > 50 else ("🟠" if risk > 25 else "🟢")
        else:
            icon = "⚪"
        
        col1, col2 = st.sidebar.columns([5, 1])
        col1.markdown(f"{icon} {t[:20]}")
        if col2.button("✕", key=f"rm_{t}"):
            remove_from_cart(t)
            st.rerun()
    
    if st.sidebar.button("Clear All"):
        st.session_state.campaign_targets = []
        st.rerun()

# ==========================================
# MAIN
# ==========================================
def main():
    st.title("🔗 Network Intelligence")
    st.caption("Who sends you leads? Who do you send leads to? Find leverage to close gaps.")
    
    events = load_and_process()
    if events is None:
        st.warning("⚠️ Upload events data on the Home page first.")
        st.page_link("app.py", label="← Go to Home", icon="🏠")
        return
    
    # Sidebar settings
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
        
        st.markdown("### Scenario")
        target_mult = st.slider("Target ×", 0.5, 2.0, 1.0, 0.1)
        velocity_mult = st.slider("Velocity ×", 0.5, 2.0, 1.0, 0.1)
        scenario = {'target_mult': target_mult, 'velocity_mult': velocity_mult}
    
    # Compute
    with st.spinner("Analyzing..."):
        shortfall_df = calculate_shortfalls(events_filtered, period_days=period_days, total_events_df=events, scenario_params=scenario)
        leverage_df = analyze_network_leverage(events_filtered)
        
        cluster_results = run_referral_clustering(events_filtered, resolution=1.5, target_max_clusters=12)
        G = cluster_results.get('graph', nx.Graph())
    
    all_builders = get_all_builders(events_filtered)
    render_sidebar_cart(shortfall_df)
    
    # Search
    st.markdown("### 🔍 Select a Builder")
    col1, col2 = st.columns([5, 1])
    with col1:
        selected = st.selectbox(
            "Search", options=[""] + all_builders,
            format_func=lambda x: "Type to search..." if x == "" else x,
            key="search", label_visibility="collapsed"
        )
    with col2:
        disabled = not selected or selected in st.session_state.campaign_targets
        if st.button("➕ Add", disabled=disabled, help="Add to campaign cart"):
            add_to_cart(selected)
            st.rerun()
    
    if selected:
        st.session_state.selected_builder = selected
    
    st.markdown("---")
    
    # Main content
    if G.number_of_nodes() == 0:
        st.info("No referral connections found in the selected period.")
        return
    
    # Compute layout once
    pos = nx.spring_layout(G, seed=42, k=1.0, iterations=50)
    
    builder = st.session_state.selected_builder
    connections = get_builder_connections(events_filtered, builder) if builder else None
    
    if builder:
        st.markdown(f"## {builder}")
    else:
        st.markdown("### Network Overview")
        st.caption("Select a builder above to see their referral relationships.")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        fig = render_network_map(G, pos, connections, builder)
        st.plotly_chart(fig, width='stretch')
        
        if builder and connections:
            n_tw = len(connections['two_way'])
            n_in = len(connections['inbound'])
            n_out = len(connections['outbound'])
            st.caption(f"**{builder}**: {n_tw} two-way | {n_in} inbound sources | {n_out} outbound destinations")
    
    with col2:
        if builder:
            render_builder_panel(builder, connections, shortfall_df, leverage_df, events_filtered)
        else:
            st.markdown("#### Quick Stats")
            st.metric("Total Builders", len(all_builders))
            st.metric("Connections", G.number_of_edges())
            
            at_risk = len(shortfall_df[shortfall_df['Risk_Score'] > 25])
            st.metric("Builders At Risk", at_risk)
            
            st.markdown("#### Top Lead Receivers")
            refs = events_filtered[events_filtered['is_referral'] == True]
            if not refs.empty:
                top_recv = refs['Dest_BuilderRegionKey'].value_counts().head(5)
                for b, c in top_recv.items():
                    cols = st.columns([4, 1])
                    cols[0].markdown(f"{b[:22]}: **{c}**")
                    if cols[1].button("→", key=f"go_{b}", help="View this builder"):
                        st.session_state.selected_builder = b
                        st.rerun()
    
    st.markdown("---")
    
    # Campaign planner
    st.markdown("## 🚀 Campaign Planner")
    targets = st.session_state.campaign_targets
    
    if not targets:
        st.info("Add builders to your cart using the ➕ button, then generate an optimized media plan.")
    else:
        st.markdown(f"**{len(targets)} targets:** " + ", ".join([t[:15] for t in targets[:5]]) + ("..." if len(targets) > 5 else ""))
        
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            use_cap = st.checkbox("Set budget cap")
        with col2:
            budget_cap = st.number_input("Max budget ($)", 10000, 1000000, 100000, step=10000, disabled=not use_cap)
        with col3:
            generate = st.button("⚡ Generate", type="primary")
        
        if generate:
            with st.spinner("Optimizing..."):
                plan_df = generate_targeted_media_plan(list(targets), shortfall_df, leverage_df, budget_cap if use_cap else None)
                st.session_state.campaign_plan = plan_df
        
        if 'campaign_plan' in st.session_state and not st.session_state.campaign_plan.empty:
            plan = st.session_state.campaign_plan
            summary = calculate_campaign_summary(plan)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Budget", fmt_currency(summary['total_budget']))
            m2.metric("Projected Leads", f"{summary['total_leads']:,.0f}")
            m3.metric("Blended CPR", fmt_currency(summary['avg_cpr']))
            m4.metric("Sources", summary['sources_used'])
            
            avail = plan.columns.tolist()
            cols = [c for c in ['Target_Builder', 'Status', 'Gap_Leads', 'Recommended_Source', 'Budget_Allocation', 'Projected_Leads', 'Effective_CPR'] if c in avail]
            
            fmt = {}
            if 'Gap_Leads' in cols: fmt['Gap_Leads'] = '{:,.0f}'
            if 'Budget_Allocation' in cols: fmt['Budget_Allocation'] = '${:,.0f}'
            if 'Projected_Leads' in cols: fmt['Projected_Leads'] = '{:,.0f}'
            if 'Effective_CPR' in cols: fmt['Effective_CPR'] = '${:,.0f}'
            
            st.dataframe(plan[cols].style.format(fmt), hide_index=True, width='stretch', height=300)
            
            col1, col2 = st.columns(2)
            col1.download_button("📥 Download CSV", plan.to_csv(index=False), "campaign_plan.csv", "text/csv")
            col2.download_button("📥 Download Excel", export_to_excel(plan, "plan.xlsx"), "campaign_plan.xlsx")
    
    st.caption(f"Network Intelligence • {len(all_builders)} builders • {G.number_of_edges()} connections")

if __name__ == "__main__":
    main()