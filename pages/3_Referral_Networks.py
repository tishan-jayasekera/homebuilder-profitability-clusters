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
    calculate_campaign_summary, analyze_campaign_network, simulate_campaign_spend
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
# CAMPAIGN PLANNER FUNCTIONS
# ==========================================
def render_campaign_network(targets, sources, flows, G, pos):
    """Render campaign-specific network showing targets, sources, and leverage flows."""
    fig = go.Figure()
    
    target_set = set(targets)
    source_set = set(s['source'] for s in sources)
    
    COL_TARGET = '#dc2626'
    COL_SOURCE = '#16a34a'
    COL_FLOW = '#3b82f6'
    COL_MUTED = '#e5e7eb'
    
    # Background edges
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines', line=dict(color=COL_MUTED, width=0.3),
            opacity=0.1, hoverinfo='skip', showlegend=False
        ))
    
    # Background nodes
    bg_nodes = [n for n in G.nodes() if n not in target_set and n not in source_set and n in pos]
    if bg_nodes:
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in bg_nodes], y=[pos[n][1] for n in bg_nodes],
            mode='markers', marker=dict(size=6, color=COL_MUTED, opacity=0.2),
            hoverinfo='skip', showlegend=False
        ))
    
    # Leverage flows
    for flow in flows:
        src, tgt = flow['source'], flow['target']
        if src not in pos or tgt not in pos:
            continue
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        width = 1.5 + min(flow['refs'] / 5, 4)
        
        dx, dy = x1 - x0, y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        if length < 0.01:
            continue
        udx, udy = dx / length, dy / length
        margin = 0.05
        sx, sy = x0 + udx * margin, y0 + udy * margin
        ex, ey = x1 - udx * margin, y1 - udy * margin
        
        fig.add_trace(go.Scatter(
            x=[sx, ex], y=[sy, ey], mode='lines',
            line=dict(color=COL_FLOW, width=width),
            opacity=0.7, hoverinfo='skip', showlegend=False
        ))
        
        # Arrow
        head = 0.03
        px, py = -udy, udx
        l_x = ex - udx * head - px * head * 0.6
        l_y = ey - udy * head - py * head * 0.6
        r_x = ex - udx * head + px * head * 0.6
        r_y = ey - udy * head + py * head * 0.6
        fig.add_trace(go.Scatter(
            x=[l_x, ex, r_x], y=[l_y, ey, r_y],
            mode='lines', fill='toself', fillcolor=COL_FLOW,
            line=dict(color=COL_FLOW, width=0.5),
            opacity=0.7, hoverinfo='skip', showlegend=False
        ))
    
    # Source nodes
    source_nodes = [s['source'] for s in sources if s['source'] in pos]
    if source_nodes:
        sizes = [20 + min(s['total_refs_sent'] / 3, 15) for s in sources if s['source'] in pos]
        hover = [f"<b>{s['source']}</b><br>Refs: {s['total_refs_sent']}<br>To targets: {s['refs_to_targets']} ({s['target_rate']:.0%})<br>eCPR: ${s['effective_cpr']:,.0f}" for s in sources if s['source'] in pos]
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in source_nodes], y=[pos[n][1] for n in source_nodes],
            mode='markers', marker=dict(size=sizes, color=COL_SOURCE, line=dict(width=2, color='white')),
            hovertext=hover, hoverinfo='text', name='📡 Sources', showlegend=True
        ))
    
    # Target nodes
    target_nodes = [t for t in targets if t in pos]
    if target_nodes:
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in target_nodes], y=[pos[n][1] for n in target_nodes],
            mode='markers+text', marker=dict(size=28, color=COL_TARGET, line=dict(width=2, color='white')),
            text=[t[:10] for t in target_nodes], textposition='top center', textfont=dict(size=9),
            hovertext=[f"<b>{t}</b><br>🎯 Target" for t in target_nodes],
            hoverinfo='text', name='🎯 Targets', showlegend=True
        ))
    
    fig.update_layout(
        height=420, margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor='white', plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='y'),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='center', x=0.5, font=dict(size=10)),
        hovermode='closest'
    )
    return fig


def render_spend_waterfall(simulation):
    """Render Sankey showing spend -> leads -> targets vs leakage."""
    summary = simulation['summary']
    
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color='white', width=0.5),
            label=[
                f"Budget<br>${summary['total_spent']:,.0f}",
                f"Leads<br>{summary['total_leads_generated']:,.0f}",
                f"To Targets<br>{summary['leads_to_targets']:,.0f}",
                f"Leaked<br>{summary['leads_leaked']:,.0f}"
            ],
            color=['#6366f1', '#3b82f6', '#16a34a', '#f59e0b']
        ),
        link=dict(
            source=[0, 1, 1], target=[1, 2, 3],
            value=[summary['total_leads_generated'], summary['leads_to_targets'], summary['leads_leaked']],
            color=['rgba(99, 102, 241, 0.4)', 'rgba(22, 163, 74, 0.4)', 'rgba(249, 115, 22, 0.4)']
        )
    ))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), font=dict(size=11))
    return fig


def render_campaign_planner(targets, shortfall_df, leverage_df, G, pos):
    """Render the full campaign planner with network visualization and traceability."""
    st.markdown("## 🚀 Campaign Planner")
    
    if not targets:
        st.info("👆 Add builders to your cart using the ➕ button to plan a campaign.")
        return
    
    campaign_analysis = analyze_campaign_network(targets, leverage_df, shortfall_df)
    sources = campaign_analysis['sources']
    flows = campaign_analysis['flows']
    stats = campaign_analysis['stats']
    
    if not sources:
        st.warning("No historical referral paths found to these targets.")
        return
    
    st.markdown(f"### {len(targets)} Target Builders")
    
    c1, c2, c3, c4 = st.columns(4)
    total_shortfall = sum(
        shortfall_df[shortfall_df['BuilderRegionKey'] == t]['Projected_Shortfall'].iloc[0]
        for t in targets
        if not shortfall_df[shortfall_df['BuilderRegionKey'] == t].empty
        and not pd.isna(shortfall_df[shortfall_df['BuilderRegionKey'] == t]['Projected_Shortfall'].iloc[0])
    )
    c1.metric("Lead Shortfall", f"{int(total_shortfall):,}")
    c2.metric("Sources Available", stats['num_sources'])
    c3.metric("Target Capture", f"{stats['target_capture_rate']:.0%}")
    c4.metric("Leakage Risk", f"{1 - stats['target_capture_rate']:.0%}")
    
    st.markdown("---")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("#### 🗺️ Leverage Network")
        st.caption("Green = Sources | Red = Targets | Blue = Referral flows")
        fig = render_campaign_network(targets, sources, flows, G, pos)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.markdown("#### 💰 Budget Simulation")
        budget = st.number_input("Campaign Budget ($)", 5000, 1000000, 50000, step=5000)
        
        if st.button("🔄 Simulate", type="primary"):
            sim = simulate_campaign_spend(targets, budget, sources, shortfall_df)
            st.session_state.campaign_simulation = sim
        
        if 'campaign_simulation' in st.session_state:
            sim = st.session_state.campaign_simulation
            summary = sim['summary']
            
            r1, r2 = st.columns(2)
            r1.metric("Leads to Targets", f"{summary['leads_to_targets']:,.0f}")
            r2.metric("Effective CPR", f"${summary['effective_cpr']:,.0f}")
            r3, r4 = st.columns(2)
            r3.metric("Gap Covered", f"{summary['coverage_pct']:.0%}")
            r4.metric("Leakage", f"{summary['leakage_pct']:.0%}")
            
            st.progress(min(summary['coverage_pct'], 1.0))
            st.caption(f"{summary['shortfall_covered']:,.0f} / {summary['target_shortfall']:,.0f} gap covered")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📡 Sources", "💰 Allocation", "🔍 Leakage"])
    
    with tab1:
        st.caption("Ranked by effective CPR (cost to get a lead to your targets)")
        source_df = pd.DataFrame(sources)
        if not source_df.empty:
            source_df.columns = ['Source', 'Total Refs', 'To Targets', 'To Others', 'Target Rate', 'Leakage', 'Base CPR', 'Eff CPR']
            st.dataframe(source_df.style.format({
                'Target Rate': '{:.0%}', 'Leakage': '{:.0%}', 'Base CPR': '${:,.0f}', 'Eff CPR': '${:,.0f}'
            }).background_gradient(subset=['Target Rate'], cmap='Greens'), hide_index=True, width='stretch', height=280)
    
    with tab2:
        if 'campaign_simulation' in st.session_state:
            sim = st.session_state.campaign_simulation
            alloc_df = pd.DataFrame(sim['allocations'])
            if not alloc_df.empty:
                fig = render_spend_waterfall(sim)
                st.plotly_chart(fig, width='stretch')
                alloc_df.columns = ['Source', 'Budget', 'Base CPR', 'Eff CPR', 'Total Leads', 'To Targets', 'Leaked', 'Target Rate', 'Leads/$1K']
                st.dataframe(alloc_df[['Source', 'Budget', 'To Targets', 'Leaked', 'Eff CPR']].style.format({
                    'Budget': '${:,.0f}', 'To Targets': '{:,.0f}', 'Leaked': '{:,.0f}', 'Eff CPR': '${:,.0f}'
                }), hide_index=True, width='stretch')
        else:
            st.info("Run simulation first")
    
    with tab3:
        leakage = campaign_analysis['leakage']
        if leakage:
            st.caption("Where else your sources send leads (not to your targets)")
            leak_df = pd.DataFrame(leakage)
            fig = go.Figure(go.Bar(y=leak_df['destination'].head(8), x=leak_df['refs'].head(8), orientation='h', marker_color='#f59e0b'))
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange='reversed'), plot_bgcolor='white')
            st.plotly_chart(fig, width='stretch')
            st.caption("💡 Consider adding these to your targets to capture more value")
        else:
            st.success("No significant leakage!")
    
    if 'campaign_simulation' in st.session_state:
        st.markdown("---")
        sim = st.session_state.campaign_simulation
        export_data = [{'Source': a['source'], 'Budget': a['budget'], 'Eff_CPR': a['effective_cpr'], 'Leads_to_Targets': a['leads_to_targets'], 'Leaked': a['leads_leaked']} for a in sim['allocations']]
        export_df = pd.DataFrame(export_data)
        col1, col2 = st.columns(2)
        col1.download_button("📥 CSV", export_df.to_csv(index=False), "campaign.csv", "text/csv")
        col2.download_button("📥 Excel", export_to_excel(export_df, "campaign.xlsx"), "campaign.xlsx")

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
    
    # Compute layout once (store for campaign planner)
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
    render_campaign_planner(st.session_state.campaign_targets, shortfall_df, leverage_df, G, pos)
    
    st.markdown("---")
    
    # Campaign planner
    render_campaign_planner(st.session_state.campaign_targets, shortfall_df, leverage_df, G, pos)


def render_campaign_network(targets, sources, flows, G, pos):
    """Render campaign-specific network showing targets, sources, and leverage flows."""
    fig = go.Figure()
    
    target_set = set(targets)
    source_set = set(s['source'] for s in sources)
    
    # Colors
    COL_TARGET = '#dc2626'    # Red - targets we want to fill
    COL_SOURCE = '#16a34a'    # Green - sources we'll leverage
    COL_FLOW = '#3b82f6'      # Blue - leverage flows
    COL_LEAKAGE = '#f59e0b'   # Orange - leakage flows
    COL_MUTED = '#e5e7eb'
    
    # Draw background edges (muted)
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines', line=dict(color=COL_MUTED, width=0.3),
            opacity=0.1, hoverinfo='skip', showlegend=False
        ))
    
    # Draw background nodes
    bg_nodes = [n for n in G.nodes() if n not in target_set and n not in source_set and n in pos]
    if bg_nodes:
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in bg_nodes],
            y=[pos[n][1] for n in bg_nodes],
            mode='markers',
            marker=dict(size=6, color=COL_MUTED, opacity=0.2),
            hoverinfo='skip', showlegend=False
        ))
    
    # Draw leverage flows (source -> target)
    for flow in flows:
        src, tgt = flow['source'], flow['target']
        if src not in pos or tgt not in pos:
            continue
        
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        
        # Line thickness based on referral count
        width = 1.5 + min(flow['refs'] / 5, 4)
        
        # Draw curved line
        dx, dy = x1 - x0, y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        if length < 0.01:
            continue
        udx, udy = dx / length, dy / length
        
        margin = 0.05
        sx, sy = x0 + udx * margin, y0 + udy * margin
        ex, ey = x1 - udx * margin, y1 - udy * margin
        
        fig.add_trace(go.Scatter(
            x=[sx, ex], y=[sy, ey],
            mode='lines', line=dict(color=COL_FLOW, width=width),
            opacity=0.7, hoverinfo='skip', showlegend=False
        ))
        
        # Arrowhead
        head = 0.03
        px, py = -udy, udx
        tip_x, tip_y = ex, ey
        l_x = tip_x - udx * head - px * head * 0.6
        l_y = tip_y - udy * head - py * head * 0.6
        r_x = tip_x - udx * head + px * head * 0.6
        r_y = tip_y - udy * head + py * head * 0.6
        
        fig.add_trace(go.Scatter(
            x=[l_x, tip_x, r_x], y=[l_y, tip_y, r_y],
            mode='lines', fill='toself', fillcolor=COL_FLOW,
            line=dict(color=COL_FLOW, width=0.5),
            opacity=0.7, hoverinfo='skip', showlegend=False
        ))
    
    # Draw source nodes
    source_nodes = [s['source'] for s in sources if s['source'] in pos]
    if source_nodes:
        sizes = [20 + min(s['total_refs_sent'] / 3, 15) for s in sources if s['source'] in pos]
        hover = [f"<b>{s['source']}</b><br>Refs sent: {s['total_refs_sent']}<br>To targets: {s['refs_to_targets']} ({s['target_rate']:.0%})<br>eCPR: ${s['effective_cpr']:,.0f}" for s in sources if s['source'] in pos]
        
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in source_nodes],
            y=[pos[n][1] for n in source_nodes],
            mode='markers',
            marker=dict(size=sizes, color=COL_SOURCE, line=dict(width=2, color='white')),
            hovertext=hover, hoverinfo='text',
            name='📡 Sources (scale media)', showlegend=True
        ))
    
    # Draw target nodes
    target_nodes = [t for t in targets if t in pos]
    if target_nodes:
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in target_nodes],
            y=[pos[n][1] for n in target_nodes],
            mode='markers+text',
            marker=dict(size=28, color=COL_TARGET, line=dict(width=2, color='white')),
            text=[t[:10] for t in target_nodes],
            textposition='top center',
            textfont=dict(size=9, color='#1f2937'),
            hovertext=[f"<b>{t}</b><br>🎯 Campaign Target" for t in target_nodes],
            hoverinfo='text',
            name='🎯 Targets (need leads)', showlegend=True
        ))
    
    fig.update_layout(
        height=420,
        margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor='white', plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='y'),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='center', x=0.5, font=dict(size=10)),
        hovermode='closest'
    )
    
    return fig


def render_spend_waterfall(simulation):
    """Render waterfall showing spend -> leads -> targets vs leakage."""
    summary = simulation['summary']
    
    labels = ['Budget', 'Total Leads', 'To Targets', 'Leaked']
    values = [
        summary['total_spent'],
        summary['total_leads_generated'],
        summary['leads_to_targets'],
        -summary['leads_leaked']
    ]
    
    # Normalize for display (leads vs dollars)
    # Show as a flow: Budget -> Leads Generated -> Split (Targets vs Leakage)
    fig = go.Figure()
    
    # Sankey-style visualization
    fig.add_trace(go.Sankey(
        node=dict(
            pad=15, thickness=20,
            line=dict(color='white', width=0.5),
            label=[
                f"Budget<br>${summary['total_spent']:,.0f}",
                f"Leads Generated<br>{summary['total_leads_generated']:,.0f}",
                f"To Targets<br>{summary['leads_to_targets']:,.0f}",
                f"Leaked<br>{summary['leads_leaked']:,.0f}"
            ],
            color=['#6366f1', '#3b82f6', '#16a34a', '#f59e0b']
        ),
        link=dict(
            source=[0, 1, 1],
            target=[1, 2, 3],
            value=[
                summary['total_leads_generated'],
                summary['leads_to_targets'],
                summary['leads_leaked']
            ],
            color=['rgba(99, 102, 241, 0.4)', 'rgba(22, 163, 74, 0.4)', 'rgba(249, 115, 22, 0.4)']
        )
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(size=11)
    )
    
    return fig


def render_campaign_planner(targets, shortfall_df, leverage_df, G, pos):
    """Render the full campaign planner with network visualization and traceability."""
    
    st.markdown("## 🚀 Campaign Planner")
    
    if not targets:
        st.info("👆 Add builders to your cart using the ➕ button to plan a campaign.")
        return
    
    # Analyze campaign network
    campaign_analysis = analyze_campaign_network(targets, leverage_df, shortfall_df)
    sources = campaign_analysis['sources']
    flows = campaign_analysis['flows']
    stats = campaign_analysis['stats']
    
    if not sources:
        st.warning("No historical referral paths found to these targets. Consider establishing new partnerships.")
        return
    
    # Header metrics
    st.markdown(f"### Campaign: {len(targets)} Target Builders")
    
    c1, c2, c3, c4 = st.columns(4)
    
    total_shortfall = sum(
        shortfall_df[shortfall_df['BuilderRegionKey'] == t]['Projected_Shortfall'].iloc[0]
        for t in targets
        if not shortfall_df[shortfall_df['BuilderRegionKey'] == t].empty
        and not pd.isna(shortfall_df[shortfall_df['BuilderRegionKey'] == t]['Projected_Shortfall'].iloc[0])
    )
    
    c1.metric("Lead Shortfall", f"{int(total_shortfall):,}", help="Total leads needed across targets")
    c2.metric("Available Sources", stats['num_sources'], help="Builders who historically send to your targets")
    c3.metric("Target Capture Rate", f"{stats['target_capture_rate']:.0%}", help="% of source referrals that go to your targets")
    c4.metric("Leakage Risk", f"{1 - stats['target_capture_rate']:.0%}", help="% that goes to non-targets")
    
    st.markdown("---")
    
    # Two columns: Network + Controls
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("#### 🗺️ Leverage Network")
        st.caption("Green = Sources to scale | Red = Your targets | Blue arrows = Referral flows")
        
        fig = render_campaign_network(targets, sources, flows, G, pos)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.markdown("#### ⚙️ Budget Simulation")
        
        budget = st.number_input("Campaign Budget ($)", min_value=5000, max_value=1000000, value=50000, step=5000)
        
        if st.button("🔄 Simulate Spend", type="primary"):
            simulation = simulate_campaign_spend(targets, budget, sources, shortfall_df)
            st.session_state.campaign_simulation = simulation
        
        if 'campaign_simulation' in st.session_state:
            sim = st.session_state.campaign_simulation
            summary = sim['summary']
            
            st.markdown("##### Results")
            
            # Key outcomes
            r1, r2 = st.columns(2)
            r1.metric("Leads to Targets", f"{summary['leads_to_targets']:,.0f}")
            r2.metric("Effective CPR", f"${summary['effective_cpr']:,.0f}")
            
            r3, r4 = st.columns(2)
            r3.metric("Shortfall Covered", f"{summary['coverage_pct']:.0%}")
            r4.metric("Leakage", f"{summary['leakage_pct']:.0%}")
            
            # Progress bar for coverage
            st.markdown("##### Gap Coverage")
            st.progress(min(summary['coverage_pct'], 1.0))
            st.caption(f"{summary['shortfall_covered']:,.0f} of {summary['target_shortfall']:,.0f} lead gap covered")
    
    st.markdown("---")
    
    # Detailed tables
    tab1, tab2, tab3 = st.tabs(["📡 Source Analysis", "💰 Spend Allocation", "🔍 Leakage"])
    
    with tab1:
        st.markdown("#### Source Efficiency Ranking")
        st.caption("Sources ranked by effective CPR (cost to get a lead to YOUR targets)")
        
        source_df = pd.DataFrame(sources)
        if not source_df.empty:
            source_df = source_df.rename(columns={
                'source': 'Source Builder',
                'total_refs_sent': 'Total Refs',
                'refs_to_targets': 'To Targets',
                'refs_to_others': 'To Others',
                'target_rate': 'Target Rate',
                'leakage_rate': 'Leakage',
                'base_cpr': 'Base CPR',
                'effective_cpr': 'Effective CPR'
            })
            
            st.dataframe(
                source_df.style.format({
                    'Target Rate': '{:.1%}',
                    'Leakage': '{:.1%}',
                    'Base CPR': '${:,.0f}',
                    'Effective CPR': '${:,.0f}'
                }).background_gradient(subset=['Target Rate'], cmap='Greens')
                .background_gradient(subset=['Effective CPR'], cmap='Reds_r'),
                hide_index=True, width='stretch', height=300
            )
    
    with tab2:
        if 'campaign_simulation' in st.session_state:
            sim = st.session_state.campaign_simulation
            alloc_df = pd.DataFrame(sim['allocations'])
            
            if not alloc_df.empty:
                st.markdown("#### Budget Allocation by Source")
                st.caption("How the budget flows through sources to generate leads")
                
                # Waterfall
                fig = render_spend_waterfall(sim)
                st.plotly_chart(fig, width='stretch')
                
                st.markdown("##### Detailed Allocation")
                alloc_df = alloc_df.rename(columns={
                    'source': 'Source',
                    'budget': 'Budget',
                    'total_leads': 'Total Leads',
                    'leads_to_targets': 'To Targets',
                    'leads_leaked': 'Leaked',
                    'target_rate': 'Target Rate',
                    'effective_cpr': 'Eff. CPR',
                    'efficiency': 'Leads/$1K'
                })
                
                display_cols = ['Source', 'Budget', 'Total Leads', 'To Targets', 'Leaked', 'Target Rate', 'Eff. CPR', 'Leads/$1K']
                display_cols = [c for c in display_cols if c in alloc_df.columns]
                
                st.dataframe(
                    alloc_df[display_cols].style.format({
                        'Budget': '${:,.0f}',
                        'Total Leads': '{:,.0f}',
                        'To Targets': '{:,.0f}',
                        'Leaked': '{:,.0f}',
                        'Target Rate': '{:.0%}',
                        'Eff. CPR': '${:,.0f}',
                        'Leads/$1K': '{:.1f}'
                    }),
                    hide_index=True, width='stretch'
                )
            else:
                st.info("Run simulation to see allocation details")
        else:
            st.info("Run budget simulation to see allocation details")
    
    with tab3:
        st.markdown("#### Leakage Analysis")
        st.caption("Where else do your sources send leads? (Referrals not going to your targets)")
        
        leakage = campaign_analysis['leakage']
        if leakage:
            leak_df = pd.DataFrame(leakage)
            leak_df = leak_df.rename(columns={'destination': 'Destination (non-target)', 'refs': 'Referrals'})
            
            # Bar chart
            fig = go.Figure(go.Bar(
                y=leak_df['Destination (non-target)'].head(10),
                x=leak_df['Referrals'].head(10),
                orientation='h',
                marker_color='#f59e0b'
            ))
            fig.update_layout(
                height=300, margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title='Referrals (leakage)',
                yaxis=dict(autorange='reversed'),
                plot_bgcolor='white'
            )
            st.plotly_chart(fig, width='stretch')
            
            st.caption("💡 **Tip**: Consider adding high-leakage destinations to your target list to capture more value from the same spend.")
        else:
            st.success("No significant leakage detected!")
    
    st.markdown("---")
    
    # Export
    st.markdown("#### 📥 Export Campaign Plan")
    
    if 'campaign_simulation' in st.session_state:
        sim = st.session_state.campaign_simulation
        
        # Build export dataframe
        export_data = []
        for alloc in sim['allocations']:
            export_data.append({
                'Source': alloc['source'],
                'Budget_Allocation': alloc['budget'],
                'Base_CPR': alloc['base_cpr'],
                'Effective_CPR': alloc['effective_cpr'],
                'Total_Leads_Generated': alloc['total_leads'],
                'Leads_to_Targets': alloc['leads_to_targets'],
                'Leads_Leaked': alloc['leads_leaked'],
                'Target_Rate': alloc['target_rate']
            })
        
        export_df = pd.DataFrame(export_data)
        
        # Add summary row
        summary_row = {
            'Source': 'TOTAL',
            'Budget_Allocation': sim['summary']['total_spent'],
            'Base_CPR': '',
            'Effective_CPR': sim['summary']['effective_cpr'],
            'Total_Leads_Generated': sim['summary']['total_leads_generated'],
            'Leads_to_Targets': sim['summary']['leads_to_targets'],
            'Leads_Leaked': sim['summary']['leads_leaked'],
            'Target_Rate': 1 - sim['summary']['leakage_pct']
        }
        export_df = pd.concat([export_df, pd.DataFrame([summary_row])], ignore_index=True)
        
        col1, col2 = st.columns(2)
        col1.download_button("📥 Download CSV", export_df.to_csv(index=False), "campaign_plan.csv", "text/csv")
        col2.download_button("📥 Download Excel", export_to_excel(export_df, "plan.xlsx"), "campaign_plan.xlsx")
    else:
        st.caption("Run budget simulation to enable export")

if __name__ == "__main__":
    main()