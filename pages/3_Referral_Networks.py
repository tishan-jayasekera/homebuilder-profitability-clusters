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
# NETWORK MAP - CLEAN, COMMERCIAL FOCUS
# ==========================================
def render_ego_network(connections, builder, leverage_df):
    """Render focused ego network showing only direct connections with clear arrows."""
    
    two_way = [c['partner'] for c in connections['two_way']]
    inbound = [c['partner'] for c in connections['inbound']]
    outbound = [c['partner'] for c in connections['outbound']]
    
    all_partners = two_way + inbound + outbound
    if not all_partners:
        return None
    
    # Layout: center builder, partners in a circle
    n = len(all_partners)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 1.0
    
    pos = {builder: (0, 0)}
    for i, partner in enumerate(all_partners):
        pos[partner] = (radius * np.cos(angles[i]), radius * np.sin(angles[i]))
    
    fig = go.Figure()
    
    # Colors
    COL_TWO_WAY = '#2563eb'   # Blue
    COL_INBOUND = '#16a34a'   # Green  
    COL_OUTBOUND = '#ea580c'  # Orange
    COL_CENTER = '#dc2626'    # Red
    
    def draw_arrow(x0, y0, x1, y1, color, width, label=None, bidirectional=False):
        """Draw edge with arrowhead."""
        dx, dy = x1 - x0, y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            return
        udx, udy = dx / length, dy / length
        
        # Shorten to not overlap nodes
        margin = 0.08
        sx, sy = x0 + udx * margin, y0 + udy * margin
        ex, ey = x1 - udx * margin, y1 - udy * margin
        
        # Line
        fig.add_trace(go.Scatter(
            x=[sx, ex], y=[sy, ey], mode='lines',
            line=dict(color=color, width=width),
            hoverinfo='skip', showlegend=False
        ))
        
        # Arrowhead at end
        head = 0.04
        px, py = -udy, udx  # perpendicular
        tip_x, tip_y = ex, ey
        l_x, l_y = tip_x - udx * head - px * head * 0.5, tip_y - udy * head - py * head * 0.5
        r_x, r_y = tip_x - udx * head + px * head * 0.5, tip_y - udy * head + py * head * 0.5
        
        fig.add_trace(go.Scatter(
            x=[l_x, tip_x, r_x], y=[l_y, tip_y, r_y],
            mode='lines', fill='toself', fillcolor=color,
            line=dict(color=color, width=1),
            hoverinfo='skip', showlegend=False
        ))
        
        # If bidirectional, add arrow at start too
        if bidirectional:
            tip_x, tip_y = sx, sy
            l_x, l_y = tip_x + udx * head - px * head * 0.5, tip_y + udy * head - py * head * 0.5
            r_x, r_y = tip_x + udx * head + px * head * 0.5, tip_y + udy * head + py * head * 0.5
            
            fig.add_trace(go.Scatter(
                x=[l_x, tip_x, r_x], y=[l_y, tip_y, r_y],
                mode='lines', fill='toself', fillcolor=color,
                line=dict(color=color, width=1),
                hoverinfo='skip', showlegend=False
            ))
    
    # Draw edges
    for c in connections['two_way']:
        p = c['partner']
        x0, y0 = pos[builder]
        x1, y1 = pos[p]
        w = 2 + min((c['in'] + c['out']) / 10, 4)
        draw_arrow(x0, y0, x1, y1, COL_TWO_WAY, w, bidirectional=True)
    
    for c in connections['inbound']:
        p = c['partner']
        x0, y0 = pos[p]  # From partner
        x1, y1 = pos[builder]  # To builder
        w = 1.5 + min(c['count'] / 10, 3)
        draw_arrow(x0, y0, x1, y1, COL_INBOUND, w)
    
    for c in connections['outbound']:
        p = c['partner']
        x0, y0 = pos[builder]  # From builder
        x1, y1 = pos[p]  # To partner
        w = 1.5 + min(c['count'] / 10, 3)
        draw_arrow(x0, y0, x1, y1, COL_OUTBOUND, w)
    
    # Draw nodes
    # Partners - Two-way
    if two_way:
        fig.add_trace(go.Scatter(
            x=[pos[p][0] for p in two_way],
            y=[pos[p][1] for p in two_way],
            mode='markers+text',
            marker=dict(size=28, color=COL_TWO_WAY, line=dict(width=2, color='white')),
            text=[p[:12] for p in two_way],
            textposition='bottom center',
            textfont=dict(size=9, color='#374151'),
            hovertext=[f"<b>{p}</b><br>↔ Two-way partner" for p in two_way],
            hoverinfo='text',
            name='↔ Two-Way',
            showlegend=True
        ))
    
    # Partners - Inbound only
    if inbound:
        fig.add_trace(go.Scatter(
            x=[pos[p][0] for p in inbound],
            y=[pos[p][1] for p in inbound],
            mode='markers+text',
            marker=dict(size=24, color=COL_INBOUND, line=dict(width=2, color='white')),
            text=[p[:12] for p in inbound],
            textposition='bottom center',
            textfont=dict(size=9, color='#374151'),
            hovertext=[f"<b>{p}</b><br>→ Sends you leads" for p in inbound],
            hoverinfo='text',
            name='→ Sends You Leads',
            showlegend=True
        ))
    
    # Partners - Outbound only
    if outbound:
        fig.add_trace(go.Scatter(
            x=[pos[p][0] for p in outbound],
            y=[pos[p][1] for p in outbound],
            mode='markers+text',
            marker=dict(size=24, color=COL_OUTBOUND, line=dict(width=2, color='white')),
            text=[p[:12] for p in outbound],
            textposition='bottom center',
            textfont=dict(size=9, color='#374151'),
            hovertext=[f"<b>{p}</b><br>← You send them leads" for p in outbound],
            hoverinfo='text',
            name='← You Send Leads',
            showlegend=True
        ))
    
    # Center node (selected builder)
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers+text',
        marker=dict(size=45, color=COL_CENTER, line=dict(width=3, color='white')),
        text=[builder[:15]],
        textposition='middle center',
        textfont=dict(size=10, color='white', family='Arial Black'),
        hovertext=f"<b>{builder}</b><br>(Selected)",
        hoverinfo='text',
        showlegend=False
    ))
    
    fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5], scaleanchor='y'),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, font=dict(size=11)),
        hovermode='closest'
    )
    
    return fig

def render_overview_network(G, pos, top_n=50):
    """Render simplified overview when no builder selected. Show top connected nodes only."""
    
    # Get top nodes by degree
    degrees = dict(G.degree(weight='weight'))
    top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:top_n]
    
    fig = go.Figure()
    
    # Draw edges (only between top nodes)
    for u, v, data in G.edges(data=True):
        if u not in top_nodes or v not in top_nodes:
            continue
        if u not in pos or v not in pos:
            continue
        
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w = data.get('weight', 1)
        
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines', hoverinfo='skip',
            line=dict(color='#d1d5db', width=0.5 + min(w / 20, 1.5)),
            opacity=0.5, showlegend=False
        ))
    
    # Draw nodes
    node_x = [pos[n][0] for n in top_nodes if n in pos]
    node_y = [pos[n][1] for n in top_nodes if n in pos]
    node_size = [12 + min(degrees[n] / 5, 25) for n in top_nodes if n in pos]
    node_text = [f"<b>{n}</b><br>Connections: {degrees[n]}" for n in top_nodes if n in pos]
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers',
        marker=dict(size=node_size, color='#6366f1', line=dict(width=1.5, color='white'), opacity=0.8),
        text=node_text, hoverinfo='text',
        showlegend=False
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='#fafafa',
        plot_bgcolor='#fafafa',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
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
    if st.session_state.selected_builder:
        builder = st.session_state.selected_builder
        connections = get_builder_connections(events_filtered, builder)
        
        st.markdown(f"## {builder}")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("#### Referral Network")
            fig = render_ego_network(connections, builder, leverage_df)
            if fig:
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No direct referral connections found for this builder.")
        
        with col2:
            render_builder_panel(builder, connections, shortfall_df, leverage_df, events_filtered)
    
    else:
        # Overview mode
        st.markdown("### Network Overview")
        st.caption("Select a builder above to see their referral relationships and leverage paths.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if G.number_of_nodes() > 0:
                pos = nx.spring_layout(G, seed=42, k=1.0, iterations=50)
                fig = render_overview_network(G, pos, top_n=40)
                st.plotly_chart(fig, width='stretch')
                st.caption(f"Showing top 40 most connected builders. {G.number_of_nodes()} total in network.")
            else:
                st.info("No referral connections in selected period.")
        
        with col2:
            st.markdown("#### Quick Stats")
            st.metric("Total Builders", len(all_builders))
            st.metric("Network Connections", G.number_of_edges())
            
            at_risk = len(shortfall_df[shortfall_df['Risk_Score'] > 25])
            st.metric("Builders At Risk", at_risk)
            
            # Top receivers
            st.markdown("#### Top Lead Receivers")
            refs = events_filtered[events_filtered['is_referral'] == True]
            if not refs.empty:
                top_recv = refs['Dest_BuilderRegionKey'].value_counts().head(5)
                for b, c in top_recv.items():
                    st.markdown(f"**{b[:25]}**: {c}")
    
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