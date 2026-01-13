"""
Network Command Center - v2.0
A radically improved referral network intelligence platform.
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
from collections import defaultdict

root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data_loader import load_events, export_to_excel
from src.normalization import normalize_events
from src.referral_clusters import run_referral_clustering
from src.utils import fmt_currency
from src.builder_pnl import build_builder_pnl
from src.network_optimization import (
    calculate_shortfalls, analyze_network_leverage,
    simulate_campaign_spend, analyze_campaign_network
)

st.set_page_config(
    page_title="Network Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# DESIGN SYSTEM
# ============================================================================
THEME = {
    "bg_dark": "#0a0a0f",
    "bg_card": "#12121a",
    "bg_elevated": "#1a1a24",
    "border": "#2a2a3a",
    "text_primary": "#ffffff",
    "text_secondary": "#8b8b9e",
    "text_muted": "#5a5a6e",
    "accent_blue": "#3b82f6",
    "accent_purple": "#8b5cf6",
    "accent_emerald": "#10b981",
    "accent_amber": "#f59e0b",
    "accent_red": "#ef4444",
    "accent_cyan": "#06b6d4",
    "gradient_start": "#3b82f6",
    "gradient_end": "#8b5cf6",
}

STYLES = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
    --bg-dark: {THEME["bg_dark"]};
    --bg-card: {THEME["bg_card"]};
    --bg-elevated: {THEME["bg_elevated"]};
    --border: {THEME["border"]};
    --text-primary: {THEME["text_primary"]};
    --text-secondary: {THEME["text_secondary"]};
    --text-muted: {THEME["text_muted"]};
    --accent-blue: {THEME["accent_blue"]};
    --accent-purple: {THEME["accent_purple"]};
    --accent-emerald: {THEME["accent_emerald"]};
    --accent-amber: {THEME["accent_amber"]};
    --accent-red: {THEME["accent_red"]};
}}

.stApp {{
    background: linear-gradient(135deg, var(--bg-dark) 0%, #0f0f18 100%);
}}

.stApp > header {{ display: none; }}

section[data-testid="stSidebar"] {{
    background: var(--bg-card);
    border-right: 1px solid var(--border);
}}

/* Command Bar */
.command-bar {{
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    backdrop-filter: blur(10px);
}}

.command-title {{
    font-family: 'Inter', sans-serif;
    font-size: 1.75rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}}

.command-subtitle {{
    color: var(--text-secondary);
    font-size: 0.875rem;
    margin-top: 0.25rem;
}}

/* Stat Cards */
.stat-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}}

.stat-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    position: relative;
    overflow: hidden;
}}

.stat-card::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
}}

.stat-label {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
    font-weight: 600;
}}

.stat-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
}}

.stat-delta {{
    font-size: 0.75rem;
    margin-top: 0.25rem;
}}

.stat-delta.positive {{ color: var(--accent-emerald); }}
.stat-delta.negative {{ color: var(--accent-red); }}

/* Glass Card */
.glass-card {{
    background: rgba(18, 18, 26, 0.8);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}}

.glass-card-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
}}

.glass-card-title {{
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

/* Builder Chip */
.builder-chip {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    font-size: 0.8rem;
    color: var(--text-primary);
    margin: 0.25rem;
    transition: all 0.2s;
}}

.builder-chip:hover {{
    border-color: var(--accent-blue);
    background: rgba(59, 130, 246, 0.1);
}}

.builder-chip .dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
}}

.builder-chip .dot.critical {{ background: var(--accent-red); }}
.builder-chip .dot.warning {{ background: var(--accent-amber); }}
.builder-chip .dot.healthy {{ background: var(--accent-emerald); }}

/* Risk Badge */
.risk-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.35rem 0.75rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

.risk-badge.critical {{
    background: rgba(239, 68, 68, 0.15);
    color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 0.3);
}}

.risk-badge.warning {{
    background: rgba(245, 158, 11, 0.15);
    color: #fcd34d;
    border: 1px solid rgba(245, 158, 11, 0.3);
}}

.risk-badge.healthy {{
    background: rgba(16, 185, 129, 0.15);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.3);
}}

/* Flow Table */
.flow-table {{
    width: 100%;
    border-collapse: collapse;
}}

.flow-table th {{
    text-align: left;
    padding: 0.75rem;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
    font-weight: 600;
}}

.flow-table td {{
    padding: 0.75rem;
    font-size: 0.85rem;
    color: var(--text-primary);
    border-bottom: 1px solid rgba(42, 42, 58, 0.5);
}}

.flow-table tr:hover td {{
    background: rgba(59, 130, 246, 0.05);
}}

/* Progress Ring */
.progress-ring {{
    position: relative;
    width: 80px;
    height: 80px;
}}

/* Insight Box */
.insight-box {{
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1));
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
}}

.insight-box .icon {{
    font-size: 1.25rem;
    margin-right: 0.5rem;
}}

.insight-box .text {{
    color: var(--text-primary);
    font-size: 0.9rem;
    line-height: 1.5;
}}

/* Action Button */
.action-btn {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1.25rem;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    border: none;
}}

.action-btn.primary {{
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    color: white;
}}

.action-btn.secondary {{
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    color: var(--text-primary);
}}

/* Tabs Override */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.5rem;
    background: var(--bg-card);
    padding: 0.5rem;
    border-radius: 12px;
    border: 1px solid var(--border);
}}

.stTabs [data-baseweb="tab"] {{
    background: transparent;
    border-radius: 8px;
    color: var(--text-secondary);
    font-weight: 500;
    padding: 0.75rem 1.25rem;
}}

.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    color: white !important;
}}

/* Metric Override */
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
}}

[data-testid="stMetricLabel"] {{
    color: var(--text-muted);
}}

/* Dataframe Override */
.stDataFrame {{
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}}

/* Hide Streamlit Elements */
#MainMenu, footer, .stDeployButton {{ display: none; }}

/* Animations */
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.5; }}
}}

.pulse {{ animation: pulse 2s infinite; }}

@keyframes slideIn {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.slide-in {{ animation: slideIn 0.3s ease-out; }}
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================
defaults = {
    'targets': [],
    'focus_builder': None,
    'compare_builders': [],
    'view_mode': 'overview',
    'simulation_results': None,
    'path_results': None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================================
# DATA LAYER
# ============================================================================
@st.cache_data(show_spinner=False)
def load_data():
    if 'events_file' not in st.session_state:
        return None
    events = load_events(st.session_state['events_file'])
    return normalize_events(events) if events is not None else None

@st.cache_data(show_spinner=False)
def process_network(_events, date_range=None):
    """Process events into network structures."""
    df = _events.copy()
    
    if date_range and len(date_range) == 2:
        mask = (df['lead_date'] >= pd.Timestamp(date_range[0])) & \
               (df['lead_date'] <= pd.Timestamp(date_range[1]))
        df = df[mask]
    
    # Build core analytics
    pnl = build_builder_pnl(df, lens='recipient', freq='ALL')
    shortfalls = calculate_shortfalls(df, period_days=90)
    leverage = analyze_network_leverage(df)
    clusters = run_referral_clustering(df, target_max_clusters=12)
    
    # Enrich builder master
    builder_master = clusters.get('builder_master', pd.DataFrame())
    if not builder_master.empty and 'BuilderRegionKey' in pnl.columns:
        builder_master = builder_master.merge(
            pnl[['BuilderRegionKey', 'Profit', 'ROAS', 'MediaCost', 'Revenue']],
            on='BuilderRegionKey', how='left'
        ).fillna(0)
    
    # Monthly trends
    df['month'] = pd.to_datetime(df['lead_date']).dt.to_period('M').dt.start_time
    monthly = df[df['is_referral'] == True].groupby(['month', 'Dest_BuilderRegionKey']).size().reset_index(name='refs')
    
    return {
        'events': df,
        'pnl': pnl,
        'shortfalls': shortfalls,
        'leverage': leverage,
        'clusters': clusters,
        'builder_master': builder_master,
        'edges': clusters.get('edges_clean', pd.DataFrame()),
        'graph': clusters.get('graph', nx.Graph()),
        'monthly': monthly,
    }

def get_builder_stats(builder, data):
    """Get comprehensive stats for a builder."""
    bm = data['builder_master']
    sf = data['shortfalls']
    edges = data['edges']
    
    row = bm[bm['BuilderRegionKey'] == builder]
    sf_row = sf[sf['BuilderRegionKey'] == builder]
    
    if row.empty:
        return None
    
    r = row.iloc[0]
    
    # Inbound/Outbound from edges
    inbound = edges[edges['Dest_builder'] == builder]['Referrals'].sum() if not edges.empty else 0
    outbound = edges[edges['Origin_builder'] == builder]['Referrals'].sum() if not edges.empty else 0
    
    return {
        'builder': builder,
        'cluster': int(r.get('ClusterId', 0)),
        'profit': float(r.get('Profit', 0)),
        'revenue': float(r.get('Revenue', 0)),
        'media_cost': float(r.get('MediaCost', 0)),
        'roas': float(r.get('ROAS', 0)),
        'refs_in': int(r.get('Referrals_in', 0)),
        'refs_out': int(r.get('Referrals_out', 0)),
        'role': r.get('Role', 'unknown'),
        'shortfall': float(sf_row['Projected_Shortfall'].iloc[0]) if not sf_row.empty else 0,
        'risk_score': float(sf_row['Risk_Score'].iloc[0]) if not sf_row.empty else 0,
        'velocity': float(sf_row['Velocity_LeadsPerDay'].iloc[0]) if not sf_row.empty else 0,
    }

def find_optimal_paths(source, target, G, leverage_df, top_k=5):
    """Find optimal referral paths between two builders."""
    if source not in G or target not in G:
        return []
    
    paths = []
    try:
        for path in nx.all_simple_paths(G, source, target, cutoff=3):
            # Calculate path metrics
            total_weight = sum(G[path[i]][path[i+1]].get('weight', 1) for i in range(len(path)-1))
            hops = len(path) - 1
            
            # Estimate efficiency
            efficiency = total_weight / hops if hops > 0 else 0
            
            paths.append({
                'path': path,
                'hops': hops,
                'volume': total_weight,
                'efficiency': efficiency,
            })
    except nx.NetworkXNoPath:
        pass
    
    return sorted(paths, key=lambda x: -x['efficiency'])[:top_k]

# ============================================================================
# VISUALIZATION COMPONENTS
# ============================================================================
def render_stat_card(label, value, delta=None, delta_type=None, icon=""):
    delta_html = ""
    if delta is not None:
        cls = "positive" if delta_type == "positive" else "negative" if delta_type == "negative" else ""
        sign = "+" if delta > 0 else ""
        delta_html = f'<div class="stat-delta {cls}">{sign}{delta}</div>'
    
    return f"""
    <div class="stat-card slide-in">
        <div class="stat-label">{icon} {label}</div>
        <div class="stat-value">{value}</div>
        {delta_html}
    </div>
    """

def render_network_viz(G, pos, builder_master, focus=None, targets=None):
    """Render advanced network visualization."""
    fig = go.Figure()
    
    COLORS = {
        'node_default': '#4a4a5a',
        'node_focus': '#3b82f6',
        'node_target': '#10b981',
        'node_source': '#f59e0b',
        'edge_default': '#2a2a3a',
        'edge_active': '#3b82f6',
    }
    
    cluster_colors = px.colors.qualitative.Set3
    cluster_map = builder_master.set_index('BuilderRegionKey')['ClusterId'].to_dict() if not builder_master.empty else {}
    profit_map = builder_master.set_index('BuilderRegionKey')['Profit'].to_dict() if 'Profit' in builder_master.columns else {}
    
    targets = targets or []
    
    # Edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=0.5, color=COLORS['edge_default']),
        hoverinfo='skip', showlegend=False
    ))
    
    # Nodes
    node_x, node_y, node_color, node_size, node_text = [], [], [], [], []
    degrees = dict(G.degree(weight='weight'))
    max_deg = max(degrees.values()) if degrees else 1
    
    for node in G.nodes():
        if node not in pos:
            continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        deg = degrees.get(node, 0)
        size = 10 + (deg / max_deg) * 30
        
        # Color logic
        if node == focus:
            color = COLORS['node_focus']
            size += 10
        elif node in targets:
            color = COLORS['node_target']
            size += 5
        else:
            cid = cluster_map.get(node, 0)
            color = cluster_colors[cid % len(cluster_colors)]
        
        profit = profit_map.get(node, 0)
        node_color.append(color)
        node_size.append(size)
        node_text.append(f"<b>{node}</b><br>Cluster: {cluster_map.get(node, '?')}<br>Profit: ${profit:,.0f}<br>Volume: {deg:,.0f}")
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers',
        marker=dict(size=node_size, color=node_color, line=dict(width=1, color='white')),
        text=node_text, hoverinfo='text', showlegend=False
    ))
    
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode='closest'
    )
    
    return fig

def render_flow_sankey(sources, targets, leverage_df):
    """Render Sankey flow diagram."""
    if leverage_df.empty:
        return None
    
    # Filter to relevant flows
    flows = leverage_df[
        leverage_df['MediaPayer_BuilderRegionKey'].isin(sources) |
        leverage_df['Dest_BuilderRegionKey'].isin(targets)
    ].copy()
    
    if flows.empty:
        return None
    
    # Build labels
    all_nodes = list(set(flows['MediaPayer_BuilderRegionKey'].tolist() + flows['Dest_BuilderRegionKey'].tolist()))
    node_idx = {n: i for i, n in enumerate(all_nodes)}
    
    source_idx = [node_idx[s] for s in flows['MediaPayer_BuilderRegionKey']]
    target_idx = [node_idx[t] for t in flows['Dest_BuilderRegionKey']]
    values = flows['Referrals_to_Target'].tolist()
    
    # Colors
    node_colors = ['#3b82f6' if n in sources else '#10b981' if n in targets else '#4a4a5a' for n in all_nodes]
    
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20,
            line=dict(color='#1a1a24', width=1),
            label=all_nodes,
            color=node_colors
        ),
        link=dict(
            source=source_idx, target=target_idx, value=values,
            color='rgba(59, 130, 246, 0.3)'
        )
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8b8b9e', size=10)
    )
    
    return fig

def render_trend_sparkline(monthly_data, builder):
    """Render sparkline trend for a builder."""
    df = monthly_data[monthly_data['Dest_BuilderRegionKey'] == builder].sort_values('month')
    if df.empty:
        return None
    
    fig = go.Figure(go.Scatter(
        x=df['month'], y=df['refs'],
        mode='lines', fill='tozeroy',
        line=dict(color='#3b82f6', width=2),
        fillcolor='rgba(59, 130, 246, 0.2)'
    ))
    
    fig.update_layout(
        height=60, width=150,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False
    )
    
    return fig

def render_cluster_heatmap(builder_master, edges):
    """Render cluster-to-cluster flow heatmap."""
    if edges.empty or builder_master.empty:
        return None
    
    cluster_map = builder_master.set_index('BuilderRegionKey')['ClusterId'].to_dict()
    
    edges = edges.copy()
    edges['origin_cluster'] = edges['Origin_builder'].map(cluster_map)
    edges['dest_cluster'] = edges['Dest_builder'].map(cluster_map)
    
    matrix = edges.groupby(['origin_cluster', 'dest_cluster'])['Referrals'].sum().unstack(fill_value=0)
    
    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=[f"C{c}" for c in matrix.columns],
        y=[f"C{c}" for c in matrix.index],
        colorscale=[[0, '#0a0a0f'], [0.5, '#3b82f6'], [1, '#8b5cf6']],
        hoverongaps=False,
        hovertemplate="From %{y} → %{x}<br>Refs: %{z}<extra></extra>"
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=40, r=10, t=10, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='To Cluster', color='#8b8b9e'),
        yaxis=dict(title='From Cluster', color='#8b8b9e'),
        font=dict(color='#8b8b9e')
    )
    
    return fig

# ============================================================================
# PAGE COMPONENTS
# ============================================================================
def render_command_bar(data):
    """Render top command bar with key stats."""
    bm = data['builder_master']
    G = data['graph']
    sf = data['shortfalls']
    
    total_nodes = len(G.nodes)
    total_edges = len(G.edges)
    total_refs = data['edges']['Referrals'].sum() if not data['edges'].empty else 0
    at_risk = len(sf[sf['Risk_Score'] > 25]) if not sf.empty else 0
    total_profit = bm['Profit'].sum() if 'Profit' in bm.columns else 0
    
    st.markdown(f"""
    <div class="command-bar">
        <div>
            <h1 class="command-title">⚡ Network Command Center</h1>
            <p class="command-subtitle">Real-time referral ecosystem intelligence</p>
        </div>
        <div style="display: flex; gap: 2rem; align-items: center;">
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700; color: white;">{total_nodes}</div>
                <div style="font-size: 0.7rem; color: #8b8b9e; text-transform: uppercase;">Nodes</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700; color: white;">{total_refs:,}</div>
                <div style="font-size: 0.7rem; color: #8b8b9e; text-transform: uppercase;">Referrals</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700; color: #ef4444;">{at_risk}</div>
                <div style="font-size: 0.7rem; color: #8b8b9e; text-transform: uppercase;">At Risk</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700; color: #10b981;">${total_profit/1000:.0f}K</div>
                <div style="font-size: 0.7rem; color: #8b8b9e; text-transform: uppercase;">Profit</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_builder_spotlight(builder, data):
    """Render detailed builder spotlight panel."""
    stats = get_builder_stats(builder, data)
    if not stats:
        st.warning(f"No data for {builder}")
        return
    
    # Risk badge
    if stats['risk_score'] > 50:
        badge = '<span class="risk-badge critical">⚠️ Critical</span>'
    elif stats['risk_score'] > 25:
        badge = '<span class="risk-badge warning">⚡ At Risk</span>'
    else:
        badge = '<span class="risk-badge healthy">✓ Healthy</span>'
    
    st.markdown(f"""
    <div class="glass-card">
        <div class="glass-card-header">
            <div class="glass-card-title">🎯 {builder}</div>
            {badge}
        </div>
        <div class="stat-grid">
            {render_stat_card("Cluster", f"#{stats['cluster']}", icon="🔮")}
            {render_stat_card("Profit", f"${stats['profit']:,.0f}", icon="💰")}
            {render_stat_card("ROAS", f"{stats['roas']:.2f}x", icon="📈")}
            {render_stat_card("Inbound", f"{stats['refs_in']:,}", icon="📥")}
            {render_stat_card("Outbound", f"{stats['refs_out']:,}", icon="📤")}
            {render_stat_card("Shortfall", f"{stats['shortfall']:.0f}", icon="⚡")}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Trend
    trend_fig = render_trend_sparkline(data['monthly'], builder)
    if trend_fig:
        st.plotly_chart(trend_fig, use_container_width=True, config={'displayModeBar': False})
    
    # Connections
    edges = data['edges']
    inbound = edges[edges['Dest_builder'] == builder].nlargest(5, 'Referrals')
    outbound = edges[edges['Origin_builder'] == builder].nlargest(5, 'Referrals')
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📥 Top Sources**")
        if not inbound.empty:
            for _, row in inbound.iterrows():
                st.markdown(f"• {row['Origin_builder']}: **{int(row['Referrals'])}** refs")
        else:
            st.caption("No inbound referrals")
    
    with c2:
        st.markdown("**📤 Top Destinations**")
        if not outbound.empty:
            for _, row in outbound.iterrows():
                st.markdown(f"• {row['Dest_builder']}: **{int(row['Referrals'])}** refs")
        else:
            st.caption("No outbound referrals")

def render_campaign_panel(data):
    """Render campaign planning panel."""
    targets = st.session_state.targets
    sf = data['shortfalls']
    leverage = data['leverage']
    
    if not targets:
        st.markdown("""
        <div class="insight-box">
            <span class="icon">💡</span>
            <span class="text">Select builders from the network to add them to your campaign targets.</span>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Campaign summary
    total_shortfall = sum(
        sf[sf['BuilderRegionKey'] == t]['Projected_Shortfall'].iloc[0]
        for t in targets if not sf[sf['BuilderRegionKey'] == t].empty
    )
    
    st.markdown(f"""
    <div class="glass-card">
        <div class="glass-card-header">
            <div class="glass-card-title">🚀 Campaign Targets ({len(targets)})</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #ef4444;">{int(total_shortfall):,} leads needed</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Target chips
    chip_html = ""
    for t in targets:
        t_sf = sf[sf['BuilderRegionKey'] == t]
        risk = t_sf['Risk_Score'].iloc[0] if not t_sf.empty else 0
        dot_class = "critical" if risk > 50 else "warning" if risk > 25 else "healthy"
        chip_html += f'<span class="builder-chip"><span class="dot {dot_class}"></span>{t[:20]}</span>'
    
    st.markdown(f'<div style="margin-bottom: 1rem;">{chip_html}</div></div>', unsafe_allow_html=True)
    
    # Budget input
    c1, c2 = st.columns([2, 1])
    with c1:
        budget = st.number_input("Campaign Budget ($)", min_value=1000, value=50000, step=5000)
    with c2:
        if st.button("⚡ Optimize", type="primary", use_container_width=True):
            # Run simulation
            analysis = analyze_campaign_network(targets, leverage, sf)
            sources = analysis.get('sources', [])
            sim = simulate_campaign_spend(targets, budget, sources, sf)
            st.session_state.simulation_results = sim
    
    # Results
    if st.session_state.simulation_results:
        sim = st.session_state.simulation_results
        summ = sim['summary']
        
        st.markdown("---")
        st.markdown("### 📊 Optimization Results")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Projected Leads", f"{int(summ['leads_to_targets']):,}")
        m2.metric("Coverage", f"{summ['coverage_pct']:.1%}")
        m3.metric("Effective CPR", f"${summ['effective_cpr']:.0f}")
        m4.metric("Leakage", f"{summ['leakage_pct']:.1%}")
        
        # Allocation table
        allocs = pd.DataFrame(sim['allocations'])
        if not allocs.empty:
            st.markdown("### 💳 Budget Allocation")
            st.dataframe(
                allocs[['source', 'budget', 'leads_to_targets', 'effective_cpr']].rename(columns={
                    'source': 'Source', 'budget': 'Budget', 
                    'leads_to_targets': 'Est. Leads', 'effective_cpr': 'CPR'
                }).style.format({'Budget': '${:,.0f}', 'Est. Leads': '{:.0f}', 'CPR': '${:.0f}'}),
                hide_index=True, use_container_width=True
            )

def render_pathfinder(data):
    """Render path finding tool."""
    G = data['graph']
    leverage = data['leverage']
    builders = sorted(G.nodes())
    
    st.markdown("### 🔍 Path Finder")
    st.caption("Discover optimal referral paths between any two builders")
    
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        source = st.selectbox("From Builder", [""] + builders, key="path_source")
    with c2:
        target = st.selectbox("To Builder", [""] + builders, key="path_target")
    with c3:
        st.write("")
        find = st.button("Find Paths", type="primary", use_container_width=True)
    
    if find and source and target:
        paths = find_optimal_paths(source, target, G, leverage)
        
        if paths:
            for i, p in enumerate(paths):
                path_str = " → ".join(p['path'])
                st.markdown(f"""
                <div class="glass-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-weight: 600; color: white;">Path {i+1}</div>
                            <div style="color: #8b8b9e; font-size: 0.85rem; margin-top: 0.25rem;">{path_str}</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.25rem; font-weight: 700; color: #3b82f6;">{p['volume']:.0f}</div>
                            <div style="font-size: 0.7rem; color: #8b8b9e;">Volume ({p['hops']} hops)</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No paths found between these builders.")

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    events = load_data()
    if events is None:
        st.markdown("""
        <div style="text-align: center; padding: 4rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">⚡</div>
            <h2 style="color: white; margin-bottom: 0.5rem;">Network Command Center</h2>
            <p style="color: #8b8b9e;">Upload your Events data on the Home page to begin.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Process data
    with st.spinner("Initializing network analysis..."):
        data = process_network(events)
    
    # Command bar
    render_command_bar(data)
    
    # Builder selection
    all_builders = sorted(data['graph'].nodes())
    
    col1, col2 = st.columns([3, 1])
    with col1:
        focus = st.selectbox("🔍 Focus Builder", [""] + all_builders, key="focus_select")
    with col2:
        st.write("")
        if focus and st.button("➕ Add to Campaign", use_container_width=True):
            if focus not in st.session_state.targets:
                st.session_state.targets.append(focus)
                st.rerun()
    
    if focus:
        st.session_state.focus_builder = focus
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🌐 Network", "📊 Analytics", "🚀 Campaign", "🔍 Explore"])
    
    with tab1:
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.markdown("### Network Topology")
            G = data['graph']
            pos = nx.spring_layout(G, seed=42, k=0.8)
            fig = render_network_viz(G, pos, data['builder_master'], 
                                      st.session_state.focus_builder, 
                                      st.session_state.targets)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        with c2:
            if st.session_state.focus_builder:
                render_builder_spotlight(st.session_state.focus_builder, data)
            else:
                st.markdown("""
                <div class="glass-card">
                    <div class="glass-card-title">💡 Quick Start</div>
                    <p style="color: #8b8b9e; font-size: 0.9rem; margin-top: 0.5rem;">
                        Select a builder above to see detailed analytics, or click nodes in the network.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show at-risk builders
                sf = data['shortfalls']
                at_risk = sf[sf['Risk_Score'] > 25].nlargest(5, 'Risk_Score')
                if not at_risk.empty:
                    st.markdown("### ⚠️ At-Risk Builders")
                    for _, row in at_risk.iterrows():
                        risk = row['Risk_Score']
                        badge_class = "critical" if risk > 50 else "warning"
                        st.markdown(f"""
                        <div class="builder-chip">
                            <span class="dot {badge_class}"></span>
                            {row['BuilderRegionKey'][:20]}
                            <span style="color: #8b8b9e; font-size: 0.75rem;">({int(risk)})</span>
                        </div>
                        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Cluster Flow Analysis")
        
        c1, c2 = st.columns(2)
        with c1:
            heatmap = render_cluster_heatmap(data['builder_master'], data['edges'])
            if heatmap:
                st.plotly_chart(heatmap, use_container_width=True)
        
        with c2:
            # Cluster summary
            bm = data['builder_master']
            if not bm.empty and 'ClusterId' in bm.columns:
                cluster_stats = bm.groupby('ClusterId').agg({
                    'BuilderRegionKey': 'count',
                    'Profit': 'sum',
                    'Referrals_in': 'sum',
                }).rename(columns={'BuilderRegionKey': 'Members'})
                
                st.markdown("### Cluster Performance")
                st.dataframe(
                    cluster_stats.style.format({'Profit': '${:,.0f}', 'Referrals_in': '{:,.0f}'}),
                    use_container_width=True
                )
        
        # Top performers
        st.markdown("### 🏆 Top Performers")
        top = data['pnl'].nlargest(10, 'Profit')[['BuilderRegionKey', 'Revenue', 'MediaCost', 'Profit', 'ROAS']]
        st.dataframe(
            top.style.format({
                'Revenue': '${:,.0f}', 'MediaCost': '${:,.0f}', 
                'Profit': '${:,.0f}', 'ROAS': '{:.2f}x'
            }),
            hide_index=True, use_container_width=True
        )
    
    with tab3:
        render_campaign_panel(data)
        
        # Clear targets button
        if st.session_state.targets:
            if st.button("🗑️ Clear All Targets"):
                st.session_state.targets = []
                st.session_state.simulation_results = None
                st.rerun()
    
    with tab4:
        render_pathfinder(data)
        
        st.markdown("---")
        
        # Data export
        st.markdown("### 📥 Export Data")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            csv = data['builder_master'].to_csv(index=False)
            st.download_button("Builder Master (CSV)", csv, "builder_master.csv", "text/csv")
        
        with c2:
            csv = data['edges'].to_csv(index=False)
            st.download_button("Edge List (CSV)", csv, "edges.csv", "text/csv")
        
        with c3:
            csv = data['shortfalls'].to_csv(index=False)
            st.download_button("Shortfalls (CSV)", csv, "shortfalls.csv", "text/csv")

if __name__ == "__main__":
    main()