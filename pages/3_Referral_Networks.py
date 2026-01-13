"""
Referral Network Analysis
A structured analytical view of the referral ecosystem.
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

st.set_page_config(page_title="Referral Network Analysis", page_icon="🔗", layout="wide")

# ============================================================================
# CLEAN, READABLE STYLES
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
#MainMenu, footer, .stDeployButton { display: none; }

/* Page header */
.page-header {
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 1rem;
    margin-bottom: 2rem;
}
.page-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #111827;
    margin: 0;
}
.page-subtitle {
    color: #6b7280;
    font-size: 0.95rem;
    margin-top: 0.25rem;
}

/* Section styling */
.section {
    margin-bottom: 2.5rem;
}
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e5e7eb;
}
.section-number {
    background: #111827;
    color: white;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 600;
    flex-shrink: 0;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #111827;
    margin: 0;
}
.section-desc {
    color: #6b7280;
    font-size: 0.85rem;
    margin-left: auto;
}

/* KPI row */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.kpi-box {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1rem;
}
.kpi-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #6b7280;
    margin-bottom: 0.25rem;
}
.kpi-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #111827;
}
.kpi-sub {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-top: 0.15rem;
}

/* Insight callout */
.insight {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
}
.insight-title {
    font-weight: 600;
    color: #92400e;
    font-size: 0.85rem;
    margin-bottom: 0.35rem;
}
.insight-text {
    color: #78350f;
    font-size: 0.9rem;
    line-height: 1.5;
}

/* Action callout */
.action-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 4px solid #3b82f6;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
}
.action-title {
    font-weight: 600;
    color: #1e40af;
    font-size: 0.85rem;
    margin-bottom: 0.35rem;
}
.action-text {
    color: #1e3a8a;
    font-size: 0.9rem;
    line-height: 1.5;
}

/* Builder card */
.builder-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}
.builder-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1rem;
}
.builder-name {
    font-size: 1rem;
    font-weight: 600;
    color: #111827;
}
.builder-cluster {
    font-size: 0.75rem;
    color: #6b7280;
    margin-top: 0.15rem;
}

/* Status pills */
.pill {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.6rem;
    border-radius: 12px;
    font-size: 0.7rem;
    font-weight: 600;
}
.pill-critical { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
.pill-warning { background: #fffbeb; color: #d97706; border: 1px solid #fde68a; }
.pill-healthy { background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }
.pill-neutral { background: #f3f4f6; color: #4b5563; border: 1px solid #e5e7eb; }

/* Mini metrics */
.mini-metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid #f3f4f6;
}
.mini-metric {
    text-align: center;
}
.mini-metric-value {
    font-size: 1rem;
    font-weight: 600;
    color: #111827;
}
.mini-metric-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    color: #9ca3af;
    margin-top: 0.1rem;
}

/* Flow table */
.flow-row {
    display: flex;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid #f3f4f6;
}
.flow-row:last-child { border-bottom: none; }
.flow-source {
    flex: 1;
    font-size: 0.85rem;
    color: #374151;
}
.flow-value {
    font-weight: 600;
    color: #111827;
    font-size: 0.9rem;
}
.flow-bar {
    width: 60px;
    height: 6px;
    background: #e5e7eb;
    border-radius: 3px;
    margin-left: 0.75rem;
    overflow: hidden;
}
.flow-bar-fill {
    height: 100%;
    background: #3b82f6;
    border-radius: 3px;
}

/* Campaign target list */
.target-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.75rem 0;
}
.target-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 0.4rem 0.7rem;
    font-size: 0.8rem;
    color: #374151;
}
.target-chip .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
}
.target-chip .dot.red { background: #ef4444; }
.target-chip .dot.amber { background: #f59e0b; }
.target-chip .dot.green { background: #10b981; }

/* Allocation table */
.alloc-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}
.alloc-table th {
    text-align: left;
    padding: 0.75rem 0.5rem;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #6b7280;
    border-bottom: 2px solid #e5e7eb;
    font-weight: 600;
}
.alloc-table td {
    padding: 0.75rem 0.5rem;
    border-bottom: 1px solid #f3f4f6;
    color: #374151;
}
.alloc-table tr:hover td {
    background: #f9fafb;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================
if 'targets' not in st.session_state:
    st.session_state.targets = []
if 'focus_builder' not in st.session_state:
    st.session_state.focus_builder = None
if 'sim_results' not in st.session_state:
    st.session_state.sim_results = None

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data(show_spinner=False)
def load_data():
    if 'events_file' not in st.session_state:
        return None
    events = load_events(st.session_state['events_file'])
    return normalize_events(events) if events is not None else None

@st.cache_data(show_spinner=False)
def process_network(_events, start_date=None, end_date=None):
    df = _events.copy()
    
    if start_date and end_date:
        mask = (df['lead_date'] >= pd.Timestamp(start_date)) & (df['lead_date'] <= pd.Timestamp(end_date))
        df = df[mask]
    
    period_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days if start_date and end_date else 90
    
    pnl = build_builder_pnl(df, lens='recipient', freq='ALL')
    shortfalls = calculate_shortfalls(df, period_days=period_days)
    leverage = analyze_network_leverage(df)
    clusters = run_referral_clustering(df, target_max_clusters=12)
    
    builder_master = clusters.get('builder_master', pd.DataFrame())
    if not builder_master.empty and 'BuilderRegionKey' in pnl.columns:
        builder_master = builder_master.merge(
            pnl[['BuilderRegionKey', 'Profit', 'ROAS', 'MediaCost', 'Revenue', 'N_referrals']],
            on='BuilderRegionKey', how='left'
        ).fillna(0)
    
    # Monthly trends
    df['month'] = pd.to_datetime(df['lead_date']).dt.to_period('M').dt.start_time
    monthly = df[df['is_referral'] == True].groupby('month').size().reset_index(name='referrals')
    
    return {
        'events': df,
        'pnl': pnl,
        'shortfalls': shortfalls,
        'leverage': leverage,
        'builder_master': builder_master,
        'edges': clusters.get('edges_clean', pd.DataFrame()),
        'graph': clusters.get('graph', nx.Graph()),
        'cluster_summary': clusters.get('cluster_summary', pd.DataFrame()),
        'monthly': monthly,
        'period_days': period_days,
    }

def get_builder_detail(builder, data):
    bm = data['builder_master']
    sf = data['shortfalls']
    edges = data['edges']
    
    row = bm[bm['BuilderRegionKey'] == builder]
    if row.empty:
        return None
    
    r = row.iloc[0]
    sf_row = sf[sf['BuilderRegionKey'] == builder]
    
    inbound = edges[edges['Dest_builder'] == builder].copy() if not edges.empty else pd.DataFrame()
    outbound = edges[edges['Origin_builder'] == builder].copy() if not edges.empty else pd.DataFrame()
    
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
        'inbound_df': inbound.nlargest(10, 'Referrals') if not inbound.empty else pd.DataFrame(),
        'outbound_df': outbound.nlargest(10, 'Referrals') if not outbound.empty else pd.DataFrame(),
    }

# ============================================================================
# VISUALIZATION
# ============================================================================
def render_network_graph(G, pos, builder_master, focus=None, targets=None):
    fig = go.Figure()
    
    targets = targets or []
    cluster_map = builder_master.set_index('BuilderRegionKey')['ClusterId'].to_dict() if not builder_master.empty else {}
    profit_map = builder_master.set_index('BuilderRegionKey')['Profit'].to_dict() if 'Profit' in builder_master.columns else {}
    
    colors = px.colors.qualitative.Set2
    
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
        line=dict(width=0.4, color='#d1d5db'),
        hoverinfo='skip', showlegend=False
    ))
    
    # Nodes
    node_x, node_y, node_color, node_size, node_text, node_line_color, node_line_width = [], [], [], [], [], [], []
    degrees = dict(G.degree(weight='weight'))
    max_deg = max(degrees.values()) if degrees else 1
    
    for node in G.nodes():
        if node not in pos:
            continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        deg = degrees.get(node, 0)
        size = 8 + (deg / max_deg) * 25
        
        cid = cluster_map.get(node, 0)
        color = colors[cid % len(colors)]
        
        line_color = '#e5e7eb'
        line_width = 1
        
        if node == focus:
            line_color = '#3b82f6'
            line_width = 3
            size += 8
        elif node in targets:
            line_color = '#10b981'
            line_width = 2
            size += 4
        
        profit = profit_map.get(node, 0)
        node_color.append(color)
        node_size.append(size)
        node_line_color.append(line_color)
        node_line_width.append(line_width)
        node_text.append(f"<b>{node}</b><br>Cluster {cid}<br>Profit: ${profit:,.0f}<br>Volume: {deg:,.0f}")
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers',
        marker=dict(
            size=node_size, color=node_color,
            line=dict(color=node_line_color, width=node_line_width)
        ),
        text=node_text, hoverinfo='text', showlegend=False
    ))
    
    fig.update_layout(
        height=450,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode='closest'
    )
    
    return fig

def render_trend_chart(monthly):
    if monthly.empty:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly['month'], y=monthly['referrals'],
        mode='lines+markers',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, title=None),
        yaxis=dict(showgrid=True, gridcolor='#f3f4f6', title=None),
        showlegend=False
    )
    
    return fig

def render_cluster_flows(edges, builder_master):
    if edges.empty or builder_master.empty:
        return None
    
    cluster_map = builder_master.set_index('BuilderRegionKey')['ClusterId'].to_dict()
    
    df = edges.copy()
    df['from_cluster'] = df['Origin_builder'].map(cluster_map)
    df['to_cluster'] = df['Dest_builder'].map(cluster_map)
    
    matrix = df.groupby(['from_cluster', 'to_cluster'])['Referrals'].sum().unstack(fill_value=0)
    
    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=[f"Cluster {c}" for c in matrix.columns],
        y=[f"Cluster {c}" for c in matrix.index],
        colorscale='Blues',
        hoverongaps=False,
        hovertemplate="From %{y} → %{x}<br>Referrals: %{z:,}<extra></extra>"
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='white',
        xaxis=dict(title=None, side='top'),
        yaxis=dict(title=None, autorange='reversed'),
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    events = load_data()
    
    if events is None:
        st.warning("⚠️ Please upload Events data on the Home page to begin analysis.")
        st.page_link("app.py", label="← Go to Home", icon="🏠")
        return
    
    # Sidebar filters
    with st.sidebar:
        st.markdown("### Filters")
        
        dates = pd.to_datetime(events['lead_date'], errors='coerce').dropna()
        min_d, max_d = dates.min().date(), dates.max().date()
        
        date_range = st.date_input("Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        
        st.markdown("---")
        st.markdown("### Campaign Targets")
        
        if st.session_state.targets:
            for t in st.session_state.targets:
                col1, col2 = st.columns([4, 1])
                col1.caption(t[:25])
                if col2.button("×", key=f"rm_{t}"):
                    st.session_state.targets.remove(t)
                    st.rerun()
            
            if st.button("Clear All", use_container_width=True):
                st.session_state.targets = []
                st.session_state.sim_results = None
                st.rerun()
        else:
            st.caption("No targets selected")
    
    # Process data
    start_d, end_d = (date_range[0], date_range[1]) if len(date_range) == 2 else (min_d, max_d)
    
    with st.spinner("Analyzing network..."):
        data = process_network(events, start_d, end_d)
    
    G = data['graph']
    bm = data['builder_master']
    sf = data['shortfalls']
    edges = data['edges']
    
    # ========================================================================
    # PAGE HEADER
    # ========================================================================
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">🔗 Referral Network Analysis</h1>
        <p class="page-subtitle">Understanding referral flows, identifying opportunities, and optimizing media allocation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # SECTION 1: EXECUTIVE SUMMARY
    # ========================================================================
    st.markdown("""
    <div class="section">
        <div class="section-header">
            <span class="section-number">1</span>
            <span class="section-title">Executive Summary</span>
            <span class="section-desc">Key network metrics at a glance</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    total_nodes = len(G.nodes)
    total_edges = len(G.edges)
    total_refs = edges['Referrals'].sum() if not edges.empty else 0
    total_profit = bm['Profit'].sum() if 'Profit' in bm.columns else 0
    at_risk_count = len(sf[sf['Risk_Score'] > 25]) if not sf.empty else 0
    avg_roas = bm['ROAS'].mean() if 'ROAS' in bm.columns else 0
    
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-box">
            <div class="kpi-label">Network Size</div>
            <div class="kpi-value">{total_nodes}</div>
            <div class="kpi-sub">active builders</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">Total Referrals</div>
            <div class="kpi-value">{total_refs:,}</div>
            <div class="kpi-sub">in period</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">Network Profit</div>
            <div class="kpi-value">${total_profit/1000:,.0f}K</div>
            <div class="kpi-sub">total attributed</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">Avg ROAS</div>
            <div class="kpi-value">{avg_roas:.2f}x</div>
            <div class="kpi-sub">return on spend</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">At-Risk Builders</div>
            <div class="kpi-value">{at_risk_count}</div>
            <div class="kpi-sub">need attention</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Trend chart
    trend_fig = render_trend_chart(data['monthly'])
    if trend_fig:
        st.plotly_chart(trend_fig, use_container_width=True, config={'displayModeBar': False})
    
    # ========================================================================
    # SECTION 2: NETWORK TOPOLOGY
    # ========================================================================
    st.markdown("""
    <div class="section">
        <div class="section-header">
            <span class="section-number">2</span>
            <span class="section-title">Network Topology</span>
            <span class="section-desc">Visual map of referral relationships</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Builder selector
        all_builders = sorted(G.nodes())
        
        c1, c2 = st.columns([3, 1])
        with c1:
            selected = st.selectbox("Select a builder to focus", [""] + all_builders, 
                                     index=0 if not st.session_state.focus_builder else 
                                     all_builders.index(st.session_state.focus_builder) + 1 if st.session_state.focus_builder in all_builders else 0,
                                     label_visibility="collapsed", placeholder="Search builders...")
        with c2:
            if selected and st.button("➕ Add to Targets", use_container_width=True):
                if selected not in st.session_state.targets:
                    st.session_state.targets.append(selected)
                    st.rerun()
        
        if selected:
            st.session_state.focus_builder = selected
        
        # Network visualization
        pos = nx.spring_layout(G, seed=42, k=0.7)
        net_fig = render_network_graph(G, pos, bm, st.session_state.focus_builder, st.session_state.targets)
        st.plotly_chart(net_fig, use_container_width=True, config={'displayModeBar': False})
        
        st.caption("Node size = referral volume. Colors = clusters. Blue border = selected. Green border = campaign target.")
    
    with col2:
        # Builder detail panel
        if st.session_state.focus_builder:
            detail = get_builder_detail(st.session_state.focus_builder, data)
            
            if detail:
                # Risk pill
                risk = detail['risk_score']
                if risk > 50:
                    pill = '<span class="pill pill-critical">Critical Risk</span>'
                elif risk > 25:
                    pill = '<span class="pill pill-warning">At Risk</span>'
                else:
                    pill = '<span class="pill pill-healthy">Healthy</span>'
                
                st.markdown(f"""
                <div class="builder-card">
                    <div class="builder-header">
                        <div>
                            <div class="builder-name">{detail['builder']}</div>
                            <div class="builder-cluster">Cluster {detail['cluster']} • {detail['role'].replace('_', ' ').title()}</div>
                        </div>
                        {pill}
                    </div>
                    <div class="mini-metrics">
                        <div class="mini-metric">
                            <div class="mini-metric-value">${detail['profit']:,.0f}</div>
                            <div class="mini-metric-label">Profit</div>
                        </div>
                        <div class="mini-metric">
                            <div class="mini-metric-value">{detail['roas']:.2f}x</div>
                            <div class="mini-metric-label">ROAS</div>
                        </div>
                        <div class="mini-metric">
                            <div class="mini-metric-value">{detail['shortfall']:.0f}</div>
                            <div class="mini-metric-label">Shortfall</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Inbound sources
                st.markdown("**Top Inbound Sources**")
                if not detail['inbound_df'].empty:
                    max_refs = detail['inbound_df']['Referrals'].max()
                    for _, row in detail['inbound_df'].head(5).iterrows():
                        pct = (row['Referrals'] / max_refs * 100) if max_refs > 0 else 0
                        st.markdown(f"""
                        <div class="flow-row">
                            <span class="flow-source">{row['Origin_builder'][:20]}</span>
                            <span class="flow-value">{int(row['Referrals'])}</span>
                            <div class="flow-bar"><div class="flow-bar-fill" style="width: {pct}%"></div></div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No inbound referrals")
                
                # Outbound destinations
                st.markdown("**Top Outbound Destinations**")
                if not detail['outbound_df'].empty:
                    max_refs = detail['outbound_df']['Referrals'].max()
                    for _, row in detail['outbound_df'].head(5).iterrows():
                        pct = (row['Referrals'] / max_refs * 100) if max_refs > 0 else 0
                        st.markdown(f"""
                        <div class="flow-row">
                            <span class="flow-source">{row['Dest_builder'][:20]}</span>
                            <span class="flow-value">{int(row['Referrals'])}</span>
                            <div class="flow-bar"><div class="flow-bar-fill" style="width: {pct}%"></div></div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No outbound referrals")
        else:
            st.markdown("""
            <div class="builder-card">
                <p style="color: #6b7280; font-size: 0.9rem; margin: 0;">
                    Select a builder from the dropdown to see detailed analytics.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show at-risk builders
            at_risk = sf[sf['Risk_Score'] > 25].nlargest(5, 'Risk_Score') if not sf.empty else pd.DataFrame()
            if not at_risk.empty:
                st.markdown("**⚠️ Builders Needing Attention**")
                for _, row in at_risk.iterrows():
                    risk = row['Risk_Score']
                    pill_class = "pill-critical" if risk > 50 else "pill-warning"
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid #f3f4f6;">
                        <span style="font-size: 0.85rem; color: #374151;">{row['BuilderRegionKey'][:25]}</span>
                        <span class="pill {pill_class}">{int(risk)}</span>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ========================================================================
    # SECTION 3: CLUSTER ANALYSIS
    # ========================================================================
    st.markdown("""
    <div class="section">
        <div class="section-header">
            <span class="section-number">3</span>
            <span class="section-title">Cluster Analysis</span>
            <span class="section-desc">How referrals flow between builder communities</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Inter-Cluster Flow Matrix**")
        heatmap = render_cluster_flows(edges, bm)
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True, config={'displayModeBar': False})
        st.caption("Darker colors = higher referral volume between clusters")
    
    with col2:
        st.markdown("**Cluster Performance**")
        if not bm.empty and 'ClusterId' in bm.columns:
            cluster_perf = bm.groupby('ClusterId').agg({
                'BuilderRegionKey': 'count',
                'Profit': 'sum',
                'Referrals_in': 'sum',
                'ROAS': 'mean'
            }).rename(columns={
                'BuilderRegionKey': 'Builders',
                'Referrals_in': 'Referrals'
            }).reset_index()
            
            st.dataframe(
                cluster_perf.style.format({
                    'Profit': '${:,.0f}',
                    'Referrals': '{:,.0f}',
                    'ROAS': '{:.2f}x'
                }),
                hide_index=True,
                use_container_width=True,
                height=300
            )
    
    # ========================================================================
    # SECTION 4: CAMPAIGN OPTIMIZATION
    # ========================================================================
    st.markdown("""
    <div class="section">
        <div class="section-header">
            <span class="section-number">4</span>
            <span class="section-title">Campaign Optimization</span>
            <span class="section-desc">Allocate budget to fill gaps efficiently</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    targets = st.session_state.targets
    
    if not targets:
        st.markdown("""
        <div class="insight">
            <div class="insight-title">💡 How to use this section</div>
            <div class="insight-text">
                Select builders from the network above and click "Add to Targets" to build a campaign.
                The optimizer will find the most efficient sources to fill their lead gaps.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Target summary
        total_shortfall = sum(
            sf[sf['BuilderRegionKey'] == t]['Projected_Shortfall'].iloc[0]
            for t in targets if not sf[sf['BuilderRegionKey'] == t].empty
        )
        
        # Target chips
        chips_html = ""
        for t in targets:
            t_sf = sf[sf['BuilderRegionKey'] == t]
            risk = t_sf['Risk_Score'].iloc[0] if not t_sf.empty else 0
            dot_class = "red" if risk > 50 else "amber" if risk > 25 else "green"
            chips_html += f'<span class="target-chip"><span class="dot {dot_class}"></span>{t[:20]}</span>'
        
        st.markdown(f"""
        <div class="builder-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <div>
                    <div style="font-weight: 600; color: #111827;">Campaign Targets</div>
                    <div style="font-size: 0.8rem; color: #6b7280;">{len(targets)} builders selected</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #dc2626;">{int(total_shortfall):,}</div>
                    <div style="font-size: 0.75rem; color: #6b7280;">leads needed</div>
                </div>
            </div>
            <div class="target-list">{chips_html}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Budget input and optimization
        col1, col2 = st.columns([2, 1])
        with col1:
            budget = st.number_input("Campaign Budget ($)", min_value=1000, value=50000, step=5000)
        with col2:
            st.write("")
            optimize = st.button("🎯 Optimize Allocation", type="primary", use_container_width=True)
        
        if optimize:
            analysis = analyze_campaign_network(targets, data['leverage'], sf)
            sources = analysis.get('sources', [])
            sim = simulate_campaign_spend(targets, budget, sources, sf)
            st.session_state.sim_results = sim
        
        # Results
        if st.session_state.sim_results:
            sim = st.session_state.sim_results
            summ = sim['summary']
            
            st.markdown("---")
            
            # Result KPIs
            st.markdown(f"""
            <div class="kpi-row">
                <div class="kpi-box">
                    <div class="kpi-label">Projected Leads</div>
                    <div class="kpi-value">{int(summ['leads_to_targets']):,}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Gap Coverage</div>
                    <div class="kpi-value">{summ['coverage_pct']:.0%}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Effective CPR</div>
                    <div class="kpi-value">${summ['effective_cpr']:.0f}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Leakage</div>
                    <div class="kpi-value">{summ['leakage_pct']:.0%}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Insight
            if summ['coverage_pct'] >= 0.8:
                st.markdown("""
                <div class="action-box">
                    <div class="action-title">✅ Strong Coverage</div>
                    <div class="action-text">This budget can cover most of the shortfall. Review the allocation below and proceed with confidence.</div>
                </div>
                """, unsafe_allow_html=True)
            elif summ['coverage_pct'] >= 0.5:
                st.markdown("""
                <div class="insight">
                    <div class="insight-title">⚡ Partial Coverage</div>
                    <div class="insight-text">Consider increasing budget or prioritizing the highest-risk targets to improve coverage.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="insight">
                    <div class="insight-title">⚠️ Limited Coverage</div>
                    <div class="insight-text">Budget is insufficient to meaningfully address shortfalls. Consider focusing on fewer targets or increasing budget significantly.</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Allocation table
            allocs = pd.DataFrame(sim['allocations'])
            if not allocs.empty and len(allocs[allocs['budget'] > 0]) > 0:
                st.markdown("**Recommended Allocation**")
                
                display_df = allocs[allocs['budget'] > 0][['source', 'budget', 'leads_to_targets', 'effective_cpr', 'target_rate']].copy()
                display_df.columns = ['Source', 'Budget', 'Est. Leads', 'CPR', 'Precision']
                
                st.dataframe(
                    display_df.style.format({
                        'Budget': '${:,.0f}',
                        'Est. Leads': '{:.0f}',
                        'CPR': '${:.0f}',
                        'Precision': '{:.0%}'
                    }),
                    hide_index=True,
                    use_container_width=True
                )
                
                # Download
                csv = display_df.to_csv(index=False)
                st.download_button("📥 Download Allocation", csv, "campaign_allocation.csv", "text/csv")
    
    # ========================================================================
    # SECTION 5: DATA EXPORT
    # ========================================================================
    st.markdown("""
    <div class="section">
        <div class="section-header">
            <span class="section-number">5</span>
            <span class="section-title">Data Export</span>
            <span class="section-desc">Download underlying data for further analysis</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Builder Master**")
        st.caption(f"{len(bm)} builders")
        csv = bm.to_csv(index=False)
        st.download_button("Download CSV", csv, "builder_master.csv", "text/csv", use_container_width=True)
    
    with col2:
        st.markdown("**Edge List**")
        st.caption(f"{len(edges)} connections")
        csv = edges.to_csv(index=False)
        st.download_button("Download CSV", csv, "edges.csv", "text/csv", use_container_width=True)
    
    with col3:
        st.markdown("**Shortfall Analysis**")
        st.caption(f"{len(sf)} builders")
        csv = sf.to_csv(index=False)
        st.download_button("Download CSV", csv, "shortfalls.csv", "text/csv", use_container_width=True)


if __name__ == "__main__":
    main()