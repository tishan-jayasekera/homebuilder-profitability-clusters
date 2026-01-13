"""
Referral Network Analysis
With granular flow tracing: see exactly where every dollar goes.
"""
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import List, Dict, Optional
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

.section-header { display: flex; align-items: center; gap: 0.6rem; margin: 1.5rem 0 1rem 0; }
.section-num { background: #111827; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 600; }
.section-title { font-size: 1rem; font-weight: 600; color: #111827; }

.card { background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 1.25rem; margin-bottom: 1rem; }
.card-title { font-weight: 600; color: #111827; font-size: 0.95rem; margin-bottom: 0.75rem; }

.metric-row { display: flex; gap: 1.5rem; margin-bottom: 1rem; }
.metric-item { }
.metric-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280; }
.metric-value { font-size: 1.1rem; font-weight: 600; color: #111827; }

.flow-card { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem; }
.flow-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; }
.flow-source { font-weight: 600; color: #111827; }
.flow-spend { font-size: 1.1rem; font-weight: 700; color: #111827; }

.flow-breakdown { margin-top: 0.5rem; }
.flow-row { display: flex; align-items: center; padding: 0.4rem 0; border-bottom: 1px solid #e5e7eb; }
.flow-row:last-child { border-bottom: none; }
.flow-dest { flex: 1; font-size: 0.85rem; color: #374151; }
.flow-amount { width: 80px; text-align: right; font-size: 0.85rem; font-weight: 500; }
.flow-pct { width: 50px; text-align: right; font-size: 0.75rem; color: #6b7280; }
.flow-bar-container { width: 100px; margin-left: 0.75rem; }
.flow-bar { height: 6px; border-radius: 3px; }
.flow-bar.target { background: #10b981; }
.flow-bar.leakage { background: #f59e0b; }

.target-tag { display: inline-block; background: #dcfce7; color: #166534; font-size: 0.7rem; font-weight: 600; padding: 0.15rem 0.4rem; border-radius: 4px; margin-left: 0.5rem; }
.leakage-tag { display: inline-block; background: #fef3c7; color: #92400e; font-size: 0.7rem; font-weight: 600; padding: 0.15rem 0.4rem; border-radius: 4px; margin-left: 0.5rem; }

.summary-box { background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 1px solid #bae6fd; border-radius: 10px; padding: 1.25rem; margin: 1rem 0; }
.summary-title { font-weight: 600; color: #0c4a6e; margin-bottom: 0.5rem; }
.summary-text { color: #0369a1; font-size: 0.9rem; line-height: 1.5; }

.warning-box { background: #fffbeb; border: 1px solid #fde68a; border-left: 4px solid #f59e0b; border-radius: 0 8px 8px 0; padding: 1rem; margin: 1rem 0; }
.warning-text { color: #92400e; font-size: 0.9rem; }

.success-box { background: #f0fdf4; border: 1px solid #bbf7d0; border-left: 4px solid #10b981; border-radius: 0 8px 8px 0; padding: 1rem; margin: 1rem 0; }
.success-text { color: #166534; font-size: 0.9rem; }

.efficiency-meter { margin: 1rem 0; }
.efficiency-label { display: flex; justify-content: space-between; font-size: 0.8rem; color: #6b7280; margin-bottom: 0.25rem; }
.efficiency-bar { height: 10px; background: #e5e7eb; border-radius: 5px; overflow: hidden; display: flex; }
.efficiency-segment { height: 100%; }
.efficiency-segment.target { background: #10b981; }
.efficiency-segment.leakage { background: #f59e0b; }
.efficiency-segment.unspent { background: #e5e7eb; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA STRUCTURES
# ============================================================================
@dataclass
class FlowDestination:
    """Where referrals from a source go."""
    builder: str
    refs: int
    pct: float
    is_target: bool

@dataclass
class SourceAnalysis:
    """Complete analysis of a media source."""
    source: str
    total_spend: float
    total_refs_out: int
    cpr: float  # Cost per referral generated
    destinations: List[FlowDestination]
    refs_to_targets: int
    refs_to_others: int
    target_rate: float  # % that reach targets
    leakage_rate: float  # % that go elsewhere

@dataclass
class AllocationPlan:
    """A specific budget allocation to a source."""
    source: str
    budget: float
    projected_total_refs: float
    projected_to_targets: Dict[str, float]  # target -> refs
    projected_leakage: Dict[str, float]  # other -> refs
    efficiency: float  # % reaching targets
    effective_cpr: float  # Cost per ref to targets

# ============================================================================
# OPTIMIZER
# ============================================================================
class FlowOptimizer:
    """
    Optimizer that traces exactly where every dollar flows.
    """
    
    def __init__(self, events: pd.DataFrame, builder_master: pd.DataFrame, edges: pd.DataFrame):
        self.events = events
        self.builder_master = builder_master
        self.edges = edges
        self._index_data()
    
    def _index_data(self):
        """Build lookup indexes."""
        # Builder spend
        self.spend = {}
        if not self.builder_master.empty:
            for _, row in self.builder_master.iterrows():
                b = row['BuilderRegionKey']
                self.spend[b] = float(row.get('MediaCost', 0) or 0)
        
        # Outbound flows: source -> [(dest, refs), ...]
        self.outflows = {}
        if not self.edges.empty:
            for _, row in self.edges.iterrows():
                src = row.get('Origin_builder')
                dst = row.get('Dest_builder')
                refs = int(row.get('Referrals', 0) or 0)
                if src and dst and refs > 0:
                    if src not in self.outflows:
                        self.outflows[src] = []
                    self.outflows[src].append((dst, refs))
    
    def analyze_source(self, source: str, targets: List[str]) -> Optional[SourceAnalysis]:
        """
        Analyze a source: where do its referrals go?
        """
        if source not in self.outflows:
            return None
        
        flows = self.outflows[source]
        total_refs = sum(refs for _, refs in flows)
        
        if total_refs == 0:
            return None
        
        spend = self.spend.get(source, 0)
        cpr = spend / total_refs if total_refs > 0 else 0
        
        destinations = []
        refs_to_targets = 0
        refs_to_others = 0
        
        for dest, refs in sorted(flows, key=lambda x: -x[1]):
            is_target = dest in targets
            pct = refs / total_refs
            
            destinations.append(FlowDestination(
                builder=dest,
                refs=refs,
                pct=pct,
                is_target=is_target
            ))
            
            if is_target:
                refs_to_targets += refs
            else:
                refs_to_others += refs
        
        return SourceAnalysis(
            source=source,
            total_spend=spend,
            total_refs_out=total_refs,
            cpr=cpr,
            destinations=destinations,
            refs_to_targets=refs_to_targets,
            refs_to_others=refs_to_others,
            target_rate=refs_to_targets / total_refs if total_refs > 0 else 0,
            leakage_rate=refs_to_others / total_refs if total_refs > 0 else 0,
        )
    
    def find_sources_for_targets(self, targets: List[str]) -> List[SourceAnalysis]:
        """
        Find all sources that can reach any of the targets.
        Returns sources sorted by efficiency (target_rate / cpr).
        """
        sources = []
        
        for source in self.outflows.keys():
            analysis = self.analyze_source(source, targets)
            if analysis and analysis.refs_to_targets > 0:
                sources.append(analysis)
        
        # Sort by: highest target rate, then lowest CPR
        sources.sort(key=lambda s: (-s.target_rate, s.cpr))
        
        return sources
    
    def create_allocation(self, source_analysis: SourceAnalysis, budget: float, targets: List[str]) -> AllocationPlan:
        """
        Project what happens if we allocate $budget to this source.
        """
        # How many total refs will this budget generate?
        if source_analysis.cpr > 0:
            total_refs = budget / source_analysis.cpr
        else:
            total_refs = 0
        
        # Distribute to destinations based on historical rates
        to_targets = {}
        to_leakage = {}
        
        for dest in source_analysis.destinations:
            projected_refs = total_refs * dest.pct
            
            if dest.builder in targets:
                to_targets[dest.builder] = projected_refs
            else:
                to_leakage[dest.builder] = projected_refs
        
        refs_to_targets = sum(to_targets.values())
        efficiency = refs_to_targets / total_refs if total_refs > 0 else 0
        eff_cpr = budget / refs_to_targets if refs_to_targets > 0 else float('inf')
        
        return AllocationPlan(
            source=source_analysis.source,
            budget=budget,
            projected_total_refs=total_refs,
            projected_to_targets=to_targets,
            projected_leakage=to_leakage,
            efficiency=efficiency,
            effective_cpr=eff_cpr,
        )
    
    def optimize(self, targets: List[str], budget: float, max_sources: int = 5) -> List[AllocationPlan]:
        """
        Optimize budget allocation across sources.
        
        Strategy:
        1. Find all sources that reach targets
        2. Rank by efficiency (target rate)
        3. Allocate to most efficient sources first
        4. Cap per-source at 40% of budget
        """
        if not targets or budget <= 0:
            return []
        
        # Find and rank sources
        sources = self.find_sources_for_targets(targets)
        
        if not sources:
            return []
        
        allocations = []
        remaining = budget
        
        for source in sources[:max_sources * 2]:  # Consider more than we'll use
            if remaining < 500:  # Minimum allocation
                break
            
            if len(allocations) >= max_sources:
                break
            
            # Cap allocation at 40% of total budget
            max_alloc = budget * 0.4
            alloc_amount = min(remaining, max_alloc)
            
            # Create allocation plan
            plan = self.create_allocation(source, alloc_amount, targets)
            
            # Only include if efficiency > 10%
            if plan.efficiency >= 0.1:
                allocations.append(plan)
                remaining -= alloc_amount
        
        return allocations

# ============================================================================
# SESSION STATE
# ============================================================================
if 'targets' not in st.session_state:
    st.session_state.targets = []
if 'focus' not in st.session_state:
    st.session_state.focus = None
if 'allocations' not in st.session_state:
    st.session_state.allocations = None

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
    
    pnl = build_builder_pnl(df, lens='recipient', freq='ALL')
    clusters = run_referral_clustering(df, target_max_clusters=12)
    
    builder_master = clusters.get('builder_master', pd.DataFrame())
    edges = clusters.get('edges_clean', pd.DataFrame())
    graph = clusters.get('graph', nx.Graph())
    
    if not builder_master.empty and not pnl.empty:
        builder_master = builder_master.merge(
            pnl[['BuilderRegionKey', 'Profit', 'ROAS', 'MediaCost', 'Revenue']],
            on='BuilderRegionKey', how='left'
        ).fillna(0)
    
    # Monthly trend
    df['month'] = df['lead_date'].dt.to_period('M').dt.start_time
    refs = df[df['is_referral'] == True]
    monthly = refs.groupby('month').size().reset_index(name='referrals')
    
    return {
        'events': df,
        'pnl': pnl,
        'builder_master': builder_master,
        'edges': edges,
        'graph': graph,
        'monthly': monthly,
    }

# ============================================================================
# VISUALIZATIONS
# ============================================================================
def render_trend(monthly):
    if monthly.empty:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly['month'], y=monthly['referrals'],
        mode='lines+markers', fill='tozeroy',
        line=dict(color='#3b82f6', width=2),
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    fig.update_layout(
        height=180, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='white', plot_bgcolor='white',
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f3f4f6'),
        showlegend=False
    )
    return fig

def render_network(G, builder_master, focus=None, targets=None):
    pos = nx.spring_layout(G, seed=42, k=0.8)
    targets = targets or []
    
    cluster_map = builder_master.set_index('BuilderRegionKey')['ClusterId'].to_dict() if not builder_master.empty else {}
    colors = px.colors.qualitative.Set2
    
    fig = go.Figure()
    
    # Edges
    for u, v in G.edges():
        if u in pos and v in pos:
            fig.add_trace(go.Scatter(
                x=[pos[u][0], pos[v][0]], y=[pos[u][1], pos[v][1]],
                mode='lines', line=dict(width=0.5, color='#e5e7eb'),
                hoverinfo='skip', showlegend=False
            ))
    
    # Nodes
    degrees = dict(G.degree(weight='weight'))
    max_deg = max(degrees.values()) if degrees else 1
    
    for node in G.nodes():
        if node not in pos:
            continue
        
        x, y = pos[node]
        deg = degrees.get(node, 0)
        size = 8 + (deg / max_deg) * 20
        cid = cluster_map.get(node, 0)
        color = colors[cid % len(colors)]
        
        line_color, line_width = 'white', 1
        if node == focus:
            line_color, line_width, size = '#3b82f6', 3, size + 8
        elif node in targets:
            line_color, line_width = '#10b981', 2.5
            size += 4
        
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers',
            marker=dict(size=size, color=color, line=dict(color=line_color, width=line_width)),
            text=f"{node}", hoverinfo='text', showlegend=False
        ))
    
    fig.update_layout(
        height=350, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='white', plot_bgcolor='white',
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig

def render_flow_detail(source_analysis: SourceAnalysis, max_rows: int = 10):
    """Render detailed flow breakdown for a source."""
    
    target_html = ""
    leakage_html = ""
    
    targets_shown = 0
    leakage_shown = 0
    
    for dest in source_analysis.destinations:
        pct_width = dest.pct * 100
        bar_class = "target" if dest.is_target else "leakage"
        tag = '<span class="target-tag">TARGET</span>' if dest.is_target else ''
        
        row_html = f'''
        <div class="flow-row">
            <div class="flow-dest">{dest.builder[:30]}{tag}</div>
            <div class="flow-amount">{dest.refs:,}</div>
            <div class="flow-pct">{dest.pct:.0%}</div>
            <div class="flow-bar-container">
                <div class="flow-bar {bar_class}" style="width: {pct_width}%"></div>
            </div>
        </div>
        '''
        
        if dest.is_target and targets_shown < max_rows:
            target_html += row_html
            targets_shown += 1
        elif not dest.is_target and leakage_shown < max_rows:
            leakage_html += row_html
            leakage_shown += 1
    
    return target_html, leakage_html

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
                c1.write(t[:22])
                if c2.button("×", key=f"rm_{t}"):
                    st.session_state.targets.remove(t)
                    st.session_state.allocations = None
                    st.rerun()
            
            if st.button("Clear All", use_container_width=True):
                st.session_state.targets = []
                st.session_state.allocations = None
                st.rerun()
        else:
            st.caption("No targets yet")
    
    # Process data
    start_d, end_d = date_range if len(date_range) == 2 else (min_d, max_d)
    with st.spinner("Analyzing network..."):
        data = process_data(events, start_d, end_d)
    
    G = data['graph']
    bm = data['builder_master']
    edges = data['edges']
    
    optimizer = FlowOptimizer(data['events'], bm, edges)
    
    # ========================================================================
    # HEADER
    # ========================================================================
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">🔗 Referral Network Analysis</h1>
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
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Builders", f"{len(G.nodes):,}")
    c2.metric("Referrals", f"{total_refs:,}")
    c3.metric("Total Profit", f"${bm['Profit'].sum():,.0f}" if 'Profit' in bm.columns else "N/A")
    c4.metric("Period", f"{(end_d - start_d).days} days")
    
    # Trend
    trend_fig = render_trend(data['monthly'])
    if trend_fig:
        st.plotly_chart(trend_fig, use_container_width=True, config={'displayModeBar': False})
    
    # Network + Selection
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        all_builders = sorted(G.nodes())
        bc1, bc2 = st.columns([3, 1])
        selected = bc1.selectbox("Select builder to analyze", [""] + all_builders, 
                                  label_visibility="collapsed", placeholder="Search builder...")
        if bc2.button("➕ Add Target", disabled=not selected, use_container_width=True):
            if selected and selected not in st.session_state.targets:
                st.session_state.targets.append(selected)
                st.session_state.allocations = None
                st.rerun()
        
        if selected:
            st.session_state.focus = selected
        
        net_fig = render_network(G, bm, st.session_state.focus, st.session_state.targets)
        st.plotly_chart(net_fig, use_container_width=True, config={'displayModeBar': False})
        st.caption("🔵 Selected builder | 🟢 Campaign targets | Node size = referral volume")
    
    with col2:
        # Source flow analysis
        if st.session_state.focus:
            st.markdown(f"**Flow Analysis: {st.session_state.focus[:25]}**")
            
            # Analyze this builder as a SOURCE
            targets_list = st.session_state.targets if st.session_state.targets else []
            source_analysis = optimizer.analyze_source(st.session_state.focus, targets_list)
            
            if source_analysis and source_analysis.total_refs_out > 0:
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-item">
                        <div class="metric-label">Total Spend</div>
                        <div class="metric-value">${source_analysis.total_spend:,.0f}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Refs Out</div>
                        <div class="metric-value">{source_analysis.total_refs_out:,}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">CPR</div>
                        <div class="metric-value">${source_analysis.cpr:,.0f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if targets_list:
                    st.markdown(f"""
                    <div class="efficiency-meter">
                        <div class="efficiency-label">
                            <span>To Targets: {source_analysis.target_rate:.0%}</span>
                            <span>Leakage: {source_analysis.leakage_rate:.0%}</span>
                        </div>
                        <div class="efficiency-bar">
                            <div class="efficiency-segment target" style="width: {source_analysis.target_rate*100}%"></div>
                            <div class="efficiency-segment leakage" style="width: {source_analysis.leakage_rate*100}%"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("**Where referrals go:**")
                
                for dest in source_analysis.destinations[:8]:
                    tag = "🎯" if dest.is_target else ""
                    bar_color = "#10b981" if dest.is_target else "#f59e0b"
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; padding: 0.3rem 0; font-size: 0.85rem;">
                        <span style="flex: 1;">{tag} {dest.builder[:22]}</span>
                        <span style="width: 50px; text-align: right; font-weight: 500;">{dest.refs}</span>
                        <span style="width: 40px; text-align: right; color: #6b7280;">{dest.pct:.0%}</span>
                        <div style="width: 60px; margin-left: 0.5rem; background: #e5e7eb; height: 4px; border-radius: 2px;">
                            <div style="width: {dest.pct*100}%; background: {bar_color}; height: 100%; border-radius: 2px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("This builder doesn't send referrals to others (or has no media spend data).")
        else:
            st.info("Select a builder to see where their referrals flow.")
    
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
        st.info("👆 Add builders to your campaign targets to enable optimization. "
                "The optimizer will find the most efficient sources and show you exactly where every dollar flows.")
        return
    
    # Show targets
    st.markdown(f"**Campaign Targets** ({len(targets)})")
    target_cols = st.columns(min(len(targets), 4))
    for i, t in enumerate(targets):
        with target_cols[i % 4]:
            st.markdown(f'<div class="card" style="padding: 0.75rem;"><b>{t[:22]}</b></div>', unsafe_allow_html=True)
    
    # Budget input
    st.markdown("---")
    bc1, bc2 = st.columns([2, 1])
    budget = bc1.number_input("Campaign Budget ($)", min_value=1000, value=25000, step=5000)
    
    if bc2.button("🎯 Find Best Sources", type="primary", use_container_width=True):
        allocations = optimizer.optimize(targets, budget)
        st.session_state.allocations = allocations
    
    # ========================================================================
    # SECTION 3: DETAILED FLOW ANALYSIS
    # ========================================================================
    if st.session_state.allocations:
        allocations = st.session_state.allocations
        
        st.markdown("---")
        st.markdown("""
        <div class="section-header">
            <span class="section-num">3</span>
            <span class="section-title">Allocation Plan with Flow Detail</span>
        </div>
        """, unsafe_allow_html=True)
        
        if not allocations:
            st.warning("No efficient sources found for these targets. Try adding different targets or increasing budget.")
            return
        
        # Summary
        total_budget = sum(a.budget for a in allocations)
        total_to_targets = sum(sum(a.projected_to_targets.values()) for a in allocations)
        total_leakage = sum(sum(a.projected_leakage.values()) for a in allocations)
        total_refs = total_to_targets + total_leakage
        overall_efficiency = total_to_targets / total_refs if total_refs > 0 else 0
        blended_cpr = total_budget / total_to_targets if total_to_targets > 0 else 0
        
        # Summary metrics
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Total Allocated", f"${total_budget:,.0f}")
        sc2.metric("Projected to Targets", f"{total_to_targets:,.0f} refs")
        sc3.metric("Efficiency", f"{overall_efficiency:.0%}")
        sc4.metric("Effective CPR", f"${blended_cpr:,.0f}")
        
        # Efficiency visualization
        st.markdown(f"""
        <div class="efficiency-meter">
            <div class="efficiency-label">
                <span>💰 Budget: ${total_budget:,.0f}</span>
                <span>🎯 To Targets: {total_to_targets:.0f} ({overall_efficiency:.0%})</span>
                <span>⚠️ Leakage: {total_leakage:.0f} ({1-overall_efficiency:.0%})</span>
            </div>
            <div class="efficiency-bar">
                <div class="efficiency-segment target" style="width: {overall_efficiency*100}%"></div>
                <div class="efficiency-segment leakage" style="width: {(1-overall_efficiency)*100}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if overall_efficiency >= 0.6:
            st.markdown('<div class="success-box"><b>✓ Good efficiency.</b> Majority of spend reaches your targets.</div>', unsafe_allow_html=True)
        elif overall_efficiency >= 0.3:
            st.markdown('<div class="warning-box"><b>⚡ Moderate leakage.</b> Consider if the leakage destinations are valuable.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box"><b>⚠️ High leakage.</b> Most spend goes to non-target builders. Review if this is acceptable.</div>', unsafe_allow_html=True)
        
        # Detailed breakdown per source
        st.markdown("### Source-by-Source Breakdown")
        st.markdown("*See exactly where every dollar flows for each recommended source.*")
        
        for alloc in allocations:
            source_analysis = optimizer.analyze_source(alloc.source, targets)
            
            with st.expander(f"**{alloc.source}** — ${alloc.budget:,.0f} allocation", expanded=True):
                # Source header metrics
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Budget", f"${alloc.budget:,.0f}")
                mc2.metric("Total Refs", f"{alloc.projected_total_refs:.0f}")
                mc3.metric("To Targets", f"{sum(alloc.projected_to_targets.values()):.0f}")
                mc4.metric("Efficiency", f"{alloc.efficiency:.0%}")
                
                # Flow breakdown
                col_t, col_l = st.columns(2)
                
                with col_t:
                    st.markdown("**🎯 Flow to Targets**")
                    if alloc.projected_to_targets:
                        for dest, refs in sorted(alloc.projected_to_targets.items(), key=lambda x: -x[1]):
                            cost = refs * alloc.effective_cpr if alloc.efficiency > 0 else 0
                            st.markdown(f"""
                            <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid #e5e7eb; font-size: 0.85rem;">
                                <span style="color: #166534;">→ {dest[:25]}</span>
                                <span><b>{refs:.0f}</b> refs (${cost:.0f})</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.caption("No flow to targets")
                
                with col_l:
                    st.markdown("**⚠️ Leakage (to non-targets)**")
                    if alloc.projected_leakage:
                        # Show top 5 leakage destinations
                        sorted_leakage = sorted(alloc.projected_leakage.items(), key=lambda x: -x[1])[:5]
                        for dest, refs in sorted_leakage:
                            cost = refs * (alloc.budget / alloc.projected_total_refs) if alloc.projected_total_refs > 0 else 0
                            st.markdown(f"""
                            <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid #e5e7eb; font-size: 0.85rem;">
                                <span style="color: #92400e;">→ {dest[:25]}</span>
                                <span><b>{refs:.0f}</b> refs (${cost:.0f})</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if len(alloc.projected_leakage) > 5:
                            others = sum(v for k, v in list(alloc.projected_leakage.items())[5:])
                            st.caption(f"+ {len(alloc.projected_leakage) - 5} other destinations ({others:.0f} refs)")
                    else:
                        st.caption("No leakage — 100% efficient!")
        
        # Per-target summary
        st.markdown("---")
        st.markdown("### Coverage by Target")
        
        target_totals = {t: 0.0 for t in targets}
        for alloc in allocations:
            for t, refs in alloc.projected_to_targets.items():
                if t in target_totals:
                    target_totals[t] += refs
        
        for t in targets:
            refs = target_totals[t]
            # Simple estimate of what they need (could be improved with actual shortfall data)
            has_refs = refs > 0
            
            st.markdown(f"""
            <div style="display: flex; align-items: center; padding: 0.6rem 0; border-bottom: 1px solid #e5e7eb;">
                <div style="flex: 1; font-size: 0.9rem;">{t}</div>
                <div style="width: 120px; text-align: right;">
                    <span style="font-weight: 600; color: {'#166534' if has_refs else '#dc2626'};">{refs:.0f}</span>
                    <span style="color: #6b7280;"> refs</span>
                </div>
                <div style="width: 80px; margin-left: 1rem;">
                    <div style="background: #e5e7eb; height: 6px; border-radius: 3px;">
                        <div style="background: {'#10b981' if has_refs else '#ef4444'}; width: {min(refs/10*100, 100)}%; height: 100%; border-radius: 3px;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Export
        st.markdown("---")
        export_data = []
        for alloc in allocations:
            for dest, refs in alloc.projected_to_targets.items():
                export_data.append({
                    'Source': alloc.source,
                    'Destination': dest,
                    'Type': 'Target',
                    'Budget Share': alloc.budget * (refs / alloc.projected_total_refs) if alloc.projected_total_refs > 0 else 0,
                    'Projected Refs': refs,
                })
            for dest, refs in list(alloc.projected_leakage.items())[:10]:
                export_data.append({
                    'Source': alloc.source,
                    'Destination': dest,
                    'Type': 'Leakage',
                    'Budget Share': alloc.budget * (refs / alloc.projected_total_refs) if alloc.projected_total_refs > 0 else 0,
                    'Projected Refs': refs,
                })
        
        if export_data:
            export_df = pd.DataFrame(export_data)
            st.download_button(
                "📥 Download Full Flow Analysis",
                export_df.to_csv(index=False),
                "flow_analysis.csv",
                "text/csv"
            )


if __name__ == "__main__":
    main()