"""
Referral Network Analysis
Algorithmic path optimization for efficient media allocation.
"""
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import sys
from pathlib import Path

root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data_loader import load_events
from src.normalization import normalize_events
from src.referral_clusters import run_referral_clustering
from src.builder_pnl import build_builder_pnl

st.set_page_config(page_title="Referral Network Analysis", page_icon="🔗", layout="wide")

# ============================================================================
# STYLES
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, .stDeployButton { display: none; }

.page-header { border-bottom: 2px solid #e5e7eb; padding-bottom: 1rem; margin-bottom: 1.5rem; }
.page-title { font-size: 1.6rem; font-weight: 700; color: #111827; margin: 0; }
.page-subtitle { color: #6b7280; font-size: 0.9rem; margin-top: 0.25rem; }

.section { margin-bottom: 2rem; }
.section-header { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.75rem; }
.section-num { background: #111827; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 600; }
.section-title { font-size: 1rem; font-weight: 600; color: #111827; }

.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.75rem; margin-bottom: 1rem; }
.kpi-box { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; }
.kpi-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em; color: #6b7280; margin-bottom: 0.25rem; }
.kpi-value { font-size: 1.3rem; font-weight: 700; color: #111827; }

.card { background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 1rem; }

.pill { display: inline-block; padding: 0.2rem 0.5rem; border-radius: 10px; font-size: 0.7rem; font-weight: 600; }
.pill-red { background: #fef2f2; color: #dc2626; }
.pill-amber { background: #fffbeb; color: #d97706; }
.pill-green { background: #f0fdf4; color: #16a34a; }
.pill-blue { background: #eff6ff; color: #2563eb; }

.insight-box { background: #fffbeb; border-left: 3px solid #f59e0b; padding: 0.75rem 1rem; margin: 0.75rem 0; border-radius: 0 6px 6px 0; }
.success-box { background: #f0fdf4; border-left: 3px solid #10b981; padding: 0.75rem 1rem; margin: 0.75rem 0; border-radius: 0 6px 6px 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class MediaPath:
    target: str
    source: str
    path_type: str  # 'direct', 'network'
    hops: List[str]
    historical_refs: float
    source_spend: float
    source_total_refs: float
    effective_cpr: float
    transfer_rate: float

@dataclass
class TargetAnalysis:
    builder: str
    shortfall: float
    risk_score: float
    current_refs: float
    paths: List[MediaPath]
    best_path: Optional[MediaPath]
    recommendation: str

# ============================================================================
# OPTIMIZER
# ============================================================================
class NetworkOptimizer:
    def __init__(self, events: pd.DataFrame, builder_master: pd.DataFrame, edges: pd.DataFrame):
        self.events = events
        self.builder_master = builder_master
        self.edges = edges
        self._build_indexes()
    
    def _build_indexes(self):
        """Build lookup indexes for fast access."""
        bm = self.builder_master
        
        # Builder metrics
        self.builder_spend = {}
        self.builder_roas = {}
        self.builder_refs_in = {}
        self.builder_refs_out = {}
        
        if not bm.empty:
            for _, row in bm.iterrows():
                b = row['BuilderRegionKey']
                self.builder_spend[b] = float(row.get('MediaCost', 0) or 0)
                self.builder_roas[b] = float(row.get('ROAS', 0) or 0)
                self.builder_refs_in[b] = float(row.get('Referrals_in', 0) or 0)
                self.builder_refs_out[b] = float(row.get('Referrals_out', 0) or 0)
        
        # Edge flows: source -> {dest: refs}
        self.flows = {}
        if not self.edges.empty:
            for _, row in self.edges.iterrows():
                src = row.get('Origin_builder')
                dst = row.get('Dest_builder')
                refs = float(row.get('Referrals', 0) or 0)
                if src and dst and refs > 0:
                    if src not in self.flows:
                        self.flows[src] = {}
                    self.flows[src][dst] = refs
    
    def find_paths(self, target: str) -> List[MediaPath]:
        """Find all viable paths to reach target."""
        paths = []
        
        # 1. DIRECT: Spend on target itself
        target_spend = self.builder_spend.get(target, 0)
        target_refs_in = self.builder_refs_in.get(target, 0)
        
        if target_spend > 0 and target_refs_in > 0:
            cpr = target_spend / target_refs_in
            paths.append(MediaPath(
                target=target,
                source=target,
                path_type='direct',
                hops=[target],
                historical_refs=target_refs_in,
                source_spend=target_spend,
                source_total_refs=target_refs_in,
                effective_cpr=cpr,
                transfer_rate=1.0
            ))
        
        # 2. NETWORK: Find sources that send refs to target
        for source, dests in self.flows.items():
            if target in dests:
                refs_to_target = dests[target]
                source_spend = self.builder_spend.get(source, 0)
                source_total = self.builder_refs_out.get(source, 0)
                
                if source_spend > 0 and source_total > 0 and refs_to_target > 0:
                    transfer_rate = refs_to_target / source_total
                    # Effective CPR = cost to generate one referral to target
                    # = (total spend / total refs out) / transfer rate
                    base_cpr = source_spend / source_total
                    eff_cpr = base_cpr / transfer_rate
                    
                    paths.append(MediaPath(
                        target=target,
                        source=source,
                        path_type='network',
                        hops=[source, target],
                        historical_refs=refs_to_target,
                        source_spend=source_spend,
                        source_total_refs=source_total,
                        effective_cpr=eff_cpr,
                        transfer_rate=transfer_rate
                    ))
        
        # Sort by effective CPR (lowest first)
        paths.sort(key=lambda p: p.effective_cpr if p.effective_cpr > 0 else float('inf'))
        return paths
    
    def analyze_target(self, target: str) -> TargetAnalysis:
        """Analyze a single target builder."""
        # Get shortfall from events
        refs = self.events[
            (self.events['is_referral'] == True) & 
            (self.events['Dest_BuilderRegionKey'] == target)
        ]
        current_refs = len(refs)
        
        # Estimate shortfall (simple heuristic: 20% growth target)
        shortfall = max(0, current_refs * 0.2) if current_refs > 0 else 10
        
        # Risk score based on recent velocity
        risk_score = 50 if current_refs < 10 else 25 if current_refs < 50 else 10
        
        paths = self.find_paths(target)
        best = paths[0] if paths else None
        
        # Recommendation
        if not paths:
            rec = 'no_data'
        elif best.path_type == 'direct':
            network_paths = [p for p in paths if p.path_type == 'network']
            if network_paths and network_paths[0].effective_cpr < best.effective_cpr * 0.85:
                rec = 'network'
                best = network_paths[0]
            else:
                rec = 'direct'
        else:
            rec = 'network'
        
        return TargetAnalysis(
            builder=target,
            shortfall=shortfall,
            risk_score=risk_score,
            current_refs=current_refs,
            paths=paths,
            best_path=best,
            recommendation=rec
        )
    
    def optimize_budget(self, targets: List[str], budget: float) -> Tuple[List[Dict], Dict]:
        """Optimize budget allocation across targets."""
        if not targets or budget <= 0:
            return [], {}
        
        # Analyze all targets
        analyses = {t: self.analyze_target(t) for t in targets}
        
        # Collect all sources and their efficiency for each target
        source_data = {}  # source -> {targets: [], best_cpr: X, total_capacity: Y}
        
        for target, analysis in analyses.items():
            for path in analysis.paths[:5]:  # Top 5 paths per target
                src = path.source
                if src not in source_data:
                    source_data[src] = {
                        'targets': {},
                        'spend': path.source_spend,
                        'total_refs': path.source_total_refs,
                    }
                source_data[src]['targets'][target] = {
                    'cpr': path.effective_cpr,
                    'transfer_rate': path.transfer_rate,
                    'path_type': path.path_type,
                }
        
        # Score sources: lower CPR + serving more targets = better
        scored_sources = []
        for src, data in source_data.items():
            cprs = [t['cpr'] for t in data['targets'].values()]
            avg_cpr = sum(cprs) / len(cprs)
            num_targets = len(data['targets'])
            # Synergy bonus: 10% discount per additional target
            synergy = 1 - (num_targets - 1) * 0.1
            adj_cpr = avg_cpr * max(synergy, 0.5)
            
            scored_sources.append({
                'source': src,
                'targets': data['targets'],
                'avg_cpr': avg_cpr,
                'adj_cpr': adj_cpr,
                'num_targets': num_targets,
                'synergy': 1 / max(synergy, 0.5),
            })
        
        scored_sources.sort(key=lambda x: x['adj_cpr'])
        
        # Allocate budget greedily
        allocations = []
        remaining = budget
        target_leads = {t: 0.0 for t in targets}
        target_needs = {t: analyses[t].shortfall for t in targets}
        
        for src_data in scored_sources:
            if remaining <= 100:
                break
            
            src = src_data['source']
            
            # Which targets can this source help?
            can_help = [t for t in src_data['targets'] if target_leads[t] < target_needs[t]]
            if not can_help:
                continue
            
            # Calculate allocation
            # Allocate proportionally to need, capped at 35% of total budget per source
            max_for_source = budget * 0.35
            total_need = sum(max(0, target_needs[t] - target_leads[t]) for t in can_help)
            
            if total_need <= 0:
                continue
            
            # Cost to fill needs
            avg_cpr = src_data['avg_cpr']
            cost_to_fill = total_need * avg_cpr
            
            alloc_amount = min(remaining, max_for_source, cost_to_fill)
            
            if alloc_amount < 100:
                continue
            
            # Distribute leads to targets
            leads_from_alloc = alloc_amount / avg_cpr
            leads_per_target = {}
            
            for t in can_help:
                t_info = src_data['targets'][t]
                need = max(0, target_needs[t] - target_leads[t])
                share = need / total_need if total_need > 0 else 1 / len(can_help)
                leads = leads_from_alloc * share
                leads_per_target[t] = leads
                target_leads[t] += leads
            
            allocations.append({
                'source': src,
                'budget': alloc_amount,
                'leads': sum(leads_per_target.values()),
                'cpr': avg_cpr,
                'targets': can_help,
                'leads_per_target': leads_per_target,
                'is_direct': src in targets,
                'synergy': src_data['synergy'],
            })
            
            remaining -= alloc_amount
        
        # Summary
        total_leads = sum(a['leads'] for a in allocations)
        total_shortfall = sum(target_needs.values())
        total_allocated = sum(a['budget'] for a in allocations)
        
        summary = {
            'total_budget': budget,
            'allocated': total_allocated,
            'remaining': remaining,
            'total_leads': total_leads,
            'total_shortfall': total_shortfall,
            'coverage': total_leads / total_shortfall if total_shortfall > 0 else 1.0,
            'avg_cpr': total_allocated / total_leads if total_leads > 0 else 0,
            'target_coverage': {t: target_leads[t] / target_needs[t] if target_needs[t] > 0 else 1.0 for t in targets},
            'target_leads': target_leads,
            'target_needs': target_needs,
        }
        
        return allocations, summary

# ============================================================================
# SESSION STATE
# ============================================================================
if 'targets' not in st.session_state:
    st.session_state.targets = []
if 'focus' not in st.session_state:
    st.session_state.focus = None
if 'results' not in st.session_state:
    st.session_state.results = None

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
def process_data(_events, start_date, end_date):
    df = _events.copy()
    df['lead_date'] = pd.to_datetime(df['lead_date'], errors='coerce')
    
    mask = (df['lead_date'] >= pd.Timestamp(start_date)) & (df['lead_date'] <= pd.Timestamp(end_date))
    df = df[mask]
    
    # Build PnL
    pnl = build_builder_pnl(df, lens='recipient', freq='ALL')
    
    # Run clustering
    clusters = run_referral_clustering(df, target_max_clusters=12)
    builder_master = clusters.get('builder_master', pd.DataFrame())
    edges = clusters.get('edges_clean', pd.DataFrame())
    graph = clusters.get('graph', nx.Graph())
    
    # Merge PnL into builder_master
    if not builder_master.empty and not pnl.empty:
        builder_master = builder_master.merge(
            pnl[['BuilderRegionKey', 'Profit', 'ROAS', 'MediaCost', 'Revenue']],
            on='BuilderRegionKey', how='left'
        ).fillna(0)
    
    # Monthly flow trend
    df['month'] = df['lead_date'].dt.to_period('M').dt.start_time
    refs = df[df['is_referral'] == True]
    monthly = refs.groupby('month').agg(
        referrals=('LeadId', 'count'),
        unique_sources=('MediaPayer_BuilderRegionKey', 'nunique'),
        unique_dests=('Dest_BuilderRegionKey', 'nunique'),
    ).reset_index()
    
    # Top flows
    top_flows = refs.groupby(['MediaPayer_BuilderRegionKey', 'Dest_BuilderRegionKey']).size().reset_index(name='refs')
    top_flows = top_flows.nlargest(20, 'refs')
    
    return {
        'events': df,
        'pnl': pnl,
        'builder_master': builder_master,
        'edges': edges,
        'graph': graph,
        'monthly': monthly,
        'top_flows': top_flows,
    }

# ============================================================================
# VISUALIZATIONS
# ============================================================================
def render_flow_trend(monthly):
    """Render referral flow trend chart."""
    if monthly.empty:
        return None
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=monthly['month'], y=monthly['referrals'], name='Referrals', marker_color='#3b82f6', opacity=0.8),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=monthly['month'], y=monthly['unique_sources'], name='Unique Sources', 
                   line=dict(color='#10b981', width=2), mode='lines+markers'),
        secondary_y=True
    )
    
    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        hovermode='x unified'
    )
    fig.update_yaxes(title_text='Referrals', secondary_y=False, gridcolor='#f3f4f6')
    fig.update_yaxes(title_text='Unique Builders', secondary_y=True, showgrid=False)
    
    return fig

def render_network(G, builder_master, focus=None, targets=None):
    """Render network graph."""
    pos = nx.spring_layout(G, seed=42, k=0.8)
    
    targets = targets or []
    cluster_map = builder_master.set_index('BuilderRegionKey')['ClusterId'].to_dict() if not builder_master.empty else {}
    colors = px.colors.qualitative.Set2
    
    fig = go.Figure()
    
    # Edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        if u in pos and v in pos:
            edge_x.extend([pos[u][0], pos[v][0], None])
            edge_y.extend([pos[u][1], pos[v][1], None])
    
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#d1d5db'), hoverinfo='skip'))
    
    # Nodes
    degrees = dict(G.degree(weight='weight'))
    max_deg = max(degrees.values()) if degrees else 1
    
    for node in G.nodes():
        if node not in pos:
            continue
        
        x, y = pos[node]
        deg = degrees.get(node, 0)
        size = 10 + (deg / max_deg) * 25
        cid = cluster_map.get(node, 0)
        color = colors[cid % len(colors)]
        
        line_color, line_width = 'white', 1
        if node == focus:
            line_color, line_width, size = '#3b82f6', 3, size + 10
        elif node in targets:
            line_color, line_width = '#10b981', 2
        
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers',
            marker=dict(size=size, color=color, line=dict(color=line_color, width=line_width)),
            text=f"<b>{node}</b><br>Cluster {cid}<br>Volume: {deg:,.0f}",
            hoverinfo='text', showlegend=False
        ))
    
    fig.update_layout(
        height=400, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='white', plot_bgcolor='white',
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        hovermode='closest'
    )
    return fig

def render_top_flows(top_flows):
    """Render top referral flows as Sankey."""
    if top_flows.empty:
        return None
    
    sources = top_flows['MediaPayer_BuilderRegionKey'].tolist()
    dests = top_flows['Dest_BuilderRegionKey'].tolist()
    values = top_flows['refs'].tolist()
    
    all_nodes = list(set(sources + dests))
    node_idx = {n: i for i, n in enumerate(all_nodes)}
    
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20,
            label=[n[:20] for n in all_nodes],
            color='#3b82f6'
        ),
        link=dict(
            source=[node_idx[s] for s in sources],
            target=[node_idx[d] for d in dests],
            value=values,
            color='rgba(59, 130, 246, 0.3)'
        )
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='white'
    )
    return fig

# ============================================================================
# MAIN
# ============================================================================
def main():
    events = load_data()
    
    if events is None:
        st.warning("⚠️ Upload Events data on the Home page to begin.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Filters")
        dates = pd.to_datetime(events['lead_date'], errors='coerce').dropna()
        min_d, max_d = dates.min().date(), dates.max().date()
        date_range = st.date_input("Date Range", value=(min_d, max_d))
        
        st.markdown("---")
        st.markdown("### Campaign Targets")
        if st.session_state.targets:
            for t in st.session_state.targets:
                c1, c2 = st.columns([4, 1])
                c1.write(t[:25])
                if c2.button("×", key=f"rm_{t}"):
                    st.session_state.targets.remove(t)
                    st.session_state.results = None
                    st.rerun()
            if st.button("Clear All"):
                st.session_state.targets = []
                st.session_state.results = None
                st.rerun()
        else:
            st.caption("No targets yet")
    
    # Load data
    start_d, end_d = date_range if len(date_range) == 2 else (min_d, max_d)
    with st.spinner("Loading..."):
        data = process_data(events, start_d, end_d)
    
    G = data['graph']
    bm = data['builder_master']
    edges = data['edges']
    
    # Initialize optimizer
    optimizer = NetworkOptimizer(data['events'], bm, edges)
    
    # ========================================================================
    # HEADER
    # ========================================================================
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">🔗 Referral Network Analysis</h1>
        <p class="page-subtitle">Analyze referral flows and optimize media allocation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # SECTION 1: NETWORK OVERVIEW
    # ========================================================================
    st.markdown("""
    <div class="section-header">
        <span class="section-num">1</span>
        <span class="section-title">Network Overview</span>
    </div>
    """, unsafe_allow_html=True)
    
    # KPIs
    total_refs = len(data['events'][data['events']['is_referral'] == True])
    total_builders = len(G.nodes)
    total_profit = bm['Profit'].sum() if 'Profit' in bm.columns else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Builders", f"{total_builders:,}")
    col2.metric("Referrals", f"{total_refs:,}")
    col3.metric("Total Profit", f"${total_profit:,.0f}")
    col4.metric("Avg ROAS", f"{bm['ROAS'].mean():.2f}x" if 'ROAS' in bm.columns else "N/A")
    
    # Flow trend
    st.markdown("**Referral Flow Over Time**")
    trend_fig = render_flow_trend(data['monthly'])
    if trend_fig:
        st.plotly_chart(trend_fig, use_container_width=True, config={'displayModeBar': False})
    
    # Network + Top Flows
    c1, c2 = st.columns([1.2, 1])
    
    with c1:
        st.markdown("**Network Topology**")
        all_builders = sorted(G.nodes())
        
        bc1, bc2 = st.columns([3, 1])
        selected = bc1.selectbox("Select builder", [""] + all_builders, label_visibility="collapsed", placeholder="Search builder...")
        if bc2.button("➕ Add", disabled=not selected):
            if selected and selected not in st.session_state.targets:
                st.session_state.targets.append(selected)
                st.session_state.results = None
                st.rerun()
        
        if selected:
            st.session_state.focus = selected
        
        net_fig = render_network(G, bm, st.session_state.focus, st.session_state.targets)
        st.plotly_chart(net_fig, use_container_width=True, config={'displayModeBar': False})
    
    with c2:
        st.markdown("**Top Referral Flows**")
        flow_fig = render_top_flows(data['top_flows'])
        if flow_fig:
            st.plotly_chart(flow_fig, use_container_width=True, config={'displayModeBar': False})
    
    # Builder detail
    if st.session_state.focus:
        st.markdown("---")
        analysis = optimizer.analyze_target(st.session_state.focus)
        
        st.markdown(f"**Builder Analysis: {analysis.builder}**")
        
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Current Referrals", f"{analysis.current_refs:,}")
        mc2.metric("Est. Shortfall", f"{analysis.shortfall:.0f}")
        mc3.metric("Risk Score", f"{analysis.risk_score:.0f}")
        mc4.metric("Available Paths", f"{len(analysis.paths)}")
        
        if analysis.paths:
            st.markdown("**Path Options** (sorted by efficiency)")
            
            path_data = []
            for p in analysis.paths[:8]:
                path_data.append({
                    'Type': '🎯 Direct' if p.path_type == 'direct' else '🔗 Network',
                    'Source': p.source[:25],
                    'Historical Refs': int(p.historical_refs),
                    'Transfer Rate': f"{p.transfer_rate:.1%}",
                    'Effective CPR': f"${p.effective_cpr:,.0f}",
                })
            
            st.dataframe(pd.DataFrame(path_data), hide_index=True, use_container_width=True)
            
            if analysis.best_path:
                rec_type = "Direct spend" if analysis.recommendation == 'direct' else "Network leverage"
                st.success(f"**Recommendation:** {rec_type} via **{analysis.best_path.source}** at ${analysis.best_path.effective_cpr:,.0f}/lead")
    
    # ========================================================================
    # SECTION 2: CAMPAIGN OPTIMIZATION
    # ========================================================================
    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <span class="section-num">2</span>
        <span class="section-title">Campaign Optimization</span>
    </div>
    """, unsafe_allow_html=True)
    
    targets = st.session_state.targets
    
    if not targets:
        st.info("👆 Add builders to targets using the selector above to enable optimization.")
        return
    
    # Show targets
    st.markdown(f"**Targets ({len(targets)})**")
    target_cols = st.columns(min(len(targets), 4))
    for i, t in enumerate(targets):
        analysis = optimizer.analyze_target(t)
        with target_cols[i % 4]:
            st.markdown(f"""
            <div class="card">
                <div style="font-weight: 600; margin-bottom: 0.25rem;">{t[:22]}</div>
                <div style="font-size: 0.8rem; color: #6b7280;">
                    Shortfall: {analysis.shortfall:.0f} • {len(analysis.paths)} paths
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Budget input
    st.markdown("---")
    bc1, bc2 = st.columns([2, 1])
    budget = bc1.number_input("Campaign Budget ($)", min_value=1000, value=25000, step=5000)
    
    if bc2.button("🎯 Optimize Allocation", type="primary", use_container_width=True):
        allocations, summary = optimizer.optimize_budget(targets, budget)
        st.session_state.results = {'allocations': allocations, 'summary': summary}
    
    # ========================================================================
    # SECTION 3: RESULTS
    # ========================================================================
    if st.session_state.results:
        allocations = st.session_state.results['allocations']
        summary = st.session_state.results['summary']
        
        st.markdown("---")
        st.markdown("""
        <div class="section-header">
            <span class="section-num">3</span>
            <span class="section-title">Optimization Results</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Summary metrics
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Projected Leads", f"{summary['total_leads']:,.0f}")
        rc2.metric("Coverage", f"{summary['coverage']:.0%}")
        rc3.metric("Blended CPR", f"${summary['avg_cpr']:,.0f}")
        rc4.metric("Sources Used", f"{len(allocations)}")
        
        # Coverage bar
        coverage_pct = min(summary['coverage'] * 100, 100)
        bar_color = '#10b981' if coverage_pct >= 80 else '#f59e0b' if coverage_pct >= 50 else '#ef4444'
        st.markdown(f"""
        <div style="margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #6b7280; margin-bottom: 0.25rem;">
                <span>Gap Coverage</span>
                <span>{summary['total_leads']:.0f} / {summary['total_shortfall']:.0f} leads</span>
            </div>
            <div style="background: #e5e7eb; height: 8px; border-radius: 4px; overflow: hidden;">
                <div style="background: {bar_color}; width: {coverage_pct}%; height: 100%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if summary['coverage'] >= 0.8:
            st.markdown('<div class="success-box">✓ <b>Strong coverage.</b> This allocation efficiently addresses most shortfall.</div>', unsafe_allow_html=True)
        elif summary['coverage'] >= 0.5:
            st.markdown('<div class="insight-box">⚡ <b>Partial coverage.</b> Consider increasing budget for better results.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight-box">⚠️ <b>Limited coverage.</b> Budget may be too low or no efficient paths available.</div>', unsafe_allow_html=True)
        
        # Allocation table
        if allocations:
            st.markdown("**Recommended Allocation**")
            
            alloc_data = []
            for a in allocations:
                synergy_text = f"⚡ {len(a['targets'])} targets" if len(a['targets']) > 1 else ""
                alloc_data.append({
                    'Source': a['source'][:30],
                    'Type': '🎯 Direct' if a['is_direct'] else '🔗 Network',
                    'Budget': f"${a['budget']:,.0f}",
                    'Est. Leads': f"{a['leads']:,.0f}",
                    'CPR': f"${a['cpr']:,.0f}",
                    'Synergy': synergy_text,
                })
            
            st.dataframe(pd.DataFrame(alloc_data), hide_index=True, use_container_width=True)
            
            # Per-target breakdown
            st.markdown("**Coverage by Target**")
            
            for t in targets:
                cov = summary['target_coverage'].get(t, 0)
                leads = summary['target_leads'].get(t, 0)
                need = summary['target_needs'].get(t, 0)
                
                bar_w = min(cov * 100, 100)
                bar_c = '#10b981' if cov >= 0.8 else '#f59e0b' if cov >= 0.5 else '#3b82f6'
                
                st.markdown(f"""
                <div style="display: flex; align-items: center; padding: 0.5rem 0; gap: 1rem; border-bottom: 1px solid #f3f4f6;">
                    <div style="flex: 1; font-size: 0.85rem;">{t[:35]}</div>
                    <div style="width: 80px; text-align: right; font-size: 0.85rem;">
                        <b>{leads:.0f}</b> / {need:.0f}
                    </div>
                    <div style="width: 100px;">
                        <div style="font-size: 0.75rem; color: #6b7280; text-align: right;">{cov:.0%}</div>
                        <div style="background: #e5e7eb; height: 6px; border-radius: 3px; overflow: hidden;">
                            <div style="background: {bar_c}; width: {bar_w}%; height: 100%;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Export
            st.markdown("---")
            export_df = pd.DataFrame([{
                'Source': a['source'],
                'Budget': a['budget'],
                'Projected Leads': a['leads'],
                'CPR': a['cpr'],
                'Type': 'Direct' if a['is_direct'] else 'Network',
                'Targets': ', '.join(a['targets']),
            } for a in allocations])
            
            st.download_button(
                "📥 Download Allocation Plan",
                export_df.to_csv(index=False),
                "allocation_plan.csv",
                "text/csv"
            )


if __name__ == "__main__":
    main()