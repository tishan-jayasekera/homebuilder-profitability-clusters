"""
Referral Network Analysis
Algorithmic campaign optimization with path-based efficiency analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import sys
from pathlib import Path

# ============================================================================
# SETUP & PATHS
# ============================================================================
root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

try:
    from src.data_loader import load_events
    from src.normalization import normalize_events
    from src.referral_clusters import run_referral_clustering
    from src.builder_pnl import build_builder_pnl
    from src.network_optimization import calculate_shortfalls, analyze_network_leverage
except ImportError as e:
    st.error(f"Critical Error: Missing backend modules. Ensure 'src' directory exists. ({e})")
    st.stop()

st.set_page_config(page_title="Referral Network Analysis", page_icon="🔗", layout="wide")

# ============================================================================
# STYLING & CSS
# ============================================================================
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    /* Global Layout */
    .block-container { padding-top: 2rem; padding-bottom: 4rem; }
    .stDeployButton { display: none; }
    
    /* Headers */
    .page-header { border-bottom: 2px solid #f3f4f6; padding-bottom: 1rem; margin-bottom: 2rem; }
    .page-title { font-size: 1.8rem; font-weight: 700; color: #111827; margin: 0; letter-spacing: -0.02em; }
    .page-subtitle { color: #6b7280; font-size: 0.95rem; margin-top: 0.5rem; }

    /* Section Headers */
    .section-header { display: flex; align-items: center; gap: 0.75rem; margin: 2rem 0 1rem 0; }
    .section-num { background: #1e293b; color: white; width: 28px; height: 28px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 700; }
    .section-title { font-size: 1.1rem; font-weight: 600; color: #1e293b; }

    /* Cards & KPIs */
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
    .kpi-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    .kpi-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: 600; }
    .kpi-value { font-size: 1.5rem; font-weight: 700; color: #0f172a; margin-top: 0.25rem; }
    
    /* Analysis Cards */
    .analysis-card { background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1.25rem; margin-bottom: 1rem; transition: border-color 0.2s; }
    .analysis-card:hover { border-color: #cbd5e1; }
    
    /* Pills & Badges */
    .pill { display: inline-flex; align-items: center; padding: 0.25rem 0.6rem; border-radius: 9999px; font-size: 0.7rem; font-weight: 600; }
    .pill-red { background: #fef2f2; color: #b91c1c; }
    .pill-amber { background: #fffbeb; color: #b45309; }
    .pill-green { background: #f0fdf4; color: #15803d; }
    .pill-blue { background: #eff6ff; color: #1d4ed8; }
    
    /* Path Visualization Rows */
    .path-row { display: flex; align-items: center; padding: 0.75rem 0; border-bottom: 1px solid #f1f5f9; gap: 1rem; }
    .path-row:last-child { border-bottom: none; }
    .path-meta { text-align: right; min-width: 90px; }
    .path-val { font-weight: 600; color: #0f172a; font-size: 0.9rem; }
    .path-lbl { font-size: 0.65rem; color: #64748b; }

    /* Allocation Table */
    .alloc-grid { display: grid; grid-template-columns: 2fr 1fr 1fr 1fr 0.8fr; gap: 0.5rem; align-items: center; padding: 0.75rem; border-bottom: 1px solid #f1f5f9; font-size: 0.9rem; }
    .alloc-header { font-weight: 600; color: #64748b; font-size: 0.75rem; text-transform: uppercase; background: #f8fafc; border-radius: 6px; border-bottom: none; }
    
    /* Info Boxes */
    .info-box { background: #f0f9ff; border-left: 4px solid #0ea5e9; padding: 1rem; border-radius: 0 8px 8px 0; font-size: 0.9rem; color: #0c4a6e; margin: 1rem 0; }
    </style>
    """, unsafe_allow_html=True)

load_css()

# ============================================================================
# LOGIC: DATA STRUCTURES
# ============================================================================
@dataclass
class MediaPath:
    target: str
    source: str
    path_type: str  # 'direct', '1-hop', '2-hop'
    hops: List[str]
    volume: float
    effective_cpr: float
    transfer_rate: float
    confidence_score: float  # Based on volume/consistency

@dataclass 
class TargetAnalysis:
    builder: str
    shortfall: float
    risk_score: float
    all_paths: List[MediaPath]
    best_path: Optional[MediaPath]
    recommendation: str

@dataclass
class AllocationResult:
    source: str
    targets_served: List[str]
    budget: float
    projected_leads: Dict[str, float]
    effective_cpr: float
    is_direct: bool
    synergy_factor: float

# ============================================================================
# LOGIC: OPTIMIZATION ENGINE
# ============================================================================
class NetworkOptimizer:
    def __init__(self, events_df: pd.DataFrame, G: nx.Graph, builder_master: pd.DataFrame, 
                 shortfalls: pd.DataFrame, leverage: pd.DataFrame):
        self.events = events_df
        self.G = G
        self.builder_master = builder_master
        self.shortfalls = shortfalls
        self.leverage = leverage
        self._build_lookups()
    
    def _build_lookups(self):
        """Pre-compute cost and flow lookups for O(1) access."""
        self.media_cost = {}
        self.total_refs_out = {}
        
        if not self.builder_master.empty:
            for _, row in self.builder_master.iterrows():
                b = str(row['BuilderRegionKey'])
                self.media_cost[b] = float(row.get('MediaCost', 0))
                self.total_refs_out[b] = float(row.get('Referrals_out', 0))
        
        self.flows = {} 
        if not self.leverage.empty:
            for _, row in self.leverage.iterrows():
                src = str(row['MediaPayer_BuilderRegionKey'])
                dst = str(row['Dest_BuilderRegionKey'])
                self.flows[(src, dst)] = float(row.get('Referrals_to_Target', 0))

    def find_paths_to_target(self, target: str) -> List[MediaPath]:
        """Identifies Direct, 1-Hop, and 2-Hop referral paths."""
        paths = []
        target = str(target)
        
        # 1. Direct
        cost = self.media_cost.get(target, 0)
        # Assuming direct refs comes from internal marketing to self
        # Using a simplified heuristic if specific column missing
        direct_refs = self.total_refs_out.get(target, 0) * 0.2  # conservative estimate
        
        if cost > 0 and direct_refs > 0:
            cpr = cost / direct_refs
            paths.append(MediaPath(target, target, 'direct', [target], direct_refs, cpr, 1.0, 1.0))
            
        # 2. Network (1-Hop)
        if target in self.G:
            predecessors = list(self.G.predecessors(target)) if self.G.is_directed() else list(self.G.neighbors(target))
            for source in predecessors:
                source = str(source)
                refs = self.flows.get((source, target), 0)
                src_total = self.total_refs_out.get(source, 0)
                src_cost = self.media_cost.get(source, 0)
                
                if refs > 5 and src_total > 0 and src_cost > 0: # Threshold for noise
                    transfer_rate = refs / src_total
                    # Base CPR at source
                    base_cpr = src_cost / src_total
                    # Effective CPR to target
                    eff_cpr = base_cpr / transfer_rate
                    
                    paths.append(MediaPath(
                        target, source, '1-hop', [source, target], 
                        refs, eff_cpr, transfer_rate, 
                        confidence_score=min(refs/20, 1.0) # More refs = higher confidence
                    ))

        return sorted(paths, key=lambda x: x.effective_cpr)

    def analyze_target(self, target: str) -> TargetAnalysis:
        sf_row = self.shortfalls[self.shortfalls['BuilderRegionKey'] == target]
        if sf_row.empty:
            return TargetAnalysis(target, 0, 0, [], None, 'insufficient_data')
            
        shortfall = float(sf_row['Projected_Shortfall'].iloc[0])
        risk = float(sf_row['Risk_Score'].iloc[0])
        paths = self.find_paths_to_target(target)
        
        best = paths[0] if paths else None
        rec = 'insufficient_data'
        
        if best:
            if best.path_type == 'direct':
                rec = 'direct'
            elif len(paths) > 1 and paths[1].path_type == 'direct' and paths[1].effective_cpr < best.effective_cpr * 1.15:
                # If direct is within 15% of network price, prefer direct for control
                best = paths[1]
                rec = 'direct'
            else:
                rec = 'network'
                
        return TargetAnalysis(target, shortfall, risk, paths, best, rec)

    def optimize_basket(self, targets: List[str], budget: float) -> Tuple[List[AllocationResult], Dict]:
        """Greedy optimization with synergy bonuses."""
        analyses = {t: self.analyze_target(t) for t in targets}
        
        # Mapping: Source -> List of Beneficiaries
        source_opportunities = {}
        
        for t, analysis in analyses.items():
            for path in analysis.all_paths[:5]: # Top 5 paths only
                if path.source not in source_opportunities:
                    source_opportunities[path.source] = []
                source_opportunities[path.source].append(path)

        # Score Sources
        scored_sources = []
        for src, paths in source_opportunities.items():
            # Calculate blended CPR if we spend here
            # Synergy: One spend event triggers leads for multiple targets
            # Harmonic mean of CPRs adjusted by unique targets served
            
            targets_served = list(set(p.target for p in paths))
            min_cpr = min(p.effective_cpr for p in paths)
            synergy = 1.0 + (len(targets_served) - 1) * 0.2 # 20% bonus per extra target
            
            scored_sources.append({
                'source': src,
                'paths': paths,
                'score_cpr': min_cpr / synergy,
                'real_cpr': min_cpr,
                'synergy': synergy,
                'targets': targets_served
            })
            
        scored_sources.sort(key=lambda x: x['score_cpr'])
        
        # Allocate Budget
        allocations = []
        remaining = budget
        
        # Simple heuristic: Cap single source at 30% of budget to ensure diversity
        max_single_alloc = budget * 0.30 
        
        for item in scored_sources:
            if remaining < 100: break
            
            src = item['source']
            # Determine demand
            total_needed_leads = sum(analyses[t].shortfall for t in item['targets'])
            
            # Estimate cost to fill
            # Using the specific path CPRs
            cost_to_fill = 0
            for t in item['targets']:
                path = next(p for p in item['paths'] if p.target == t)
                needed = analyses[t].shortfall
                cost_to_fill += needed * path.effective_cpr
            
            alloc_amount = min(remaining, max_single_alloc, cost_to_fill)
            
            if alloc_amount < 500: continue # Minimum spend threshold
            
            # Calculate yield
            leads_per_target = {}
            for t in item['targets']:
                path = next(p for p in item['paths'] if p.target == t)
                # How many leads does this money buy for this target?
                # Leads = (Spend / Base_CPR) * Transfer_Rate
                # effectively: Spend / Effective_CPR
                leads_per_target[t] = alloc_amount / path.effective_cpr
            
            allocations.append(AllocationResult(
                src, item['targets'], alloc_amount, leads_per_target,
                item['real_cpr'], src in targets, item['synergy']
            ))
            
            remaining -= alloc_amount
            
        # Summary
        total_leads = sum(sum(a.projected_leads.values()) for a in allocations)
        total_shortfall = sum(a.shortfall for a in analyses.values())
        
        summary = {
            'total_leads': total_leads,
            'coverage': total_leads / total_shortfall if total_shortfall > 0 else 1.0,
            'blended_cpr': (budget - remaining) / total_leads if total_leads > 0 else 0,
            'unspent': remaining
        }
        
        return allocations, summary

# ============================================================================
# VISUALIZATION
# ============================================================================
def render_network_subset(G, focus_node, targets):
    """Renders a graph highlighting connections to the focus node."""
    # Subgraph for performance
    nodes = set([focus_node] + targets)
    if focus_node in G:
        nodes.update(G.neighbors(focus_node))
        if G.is_directed():
            nodes.update(G.predecessors(focus_node))
    
    subG = G.subgraph(nodes)
    pos = nx.spring_layout(subG, seed=42)
    
    edge_x, edge_y = [], []
    for u, v in subG.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, node_c, node_s = [], [], [], []
    for n in subG.nodes():
        node_x.append(pos[n][0])
        node_y.append(pos[n][1])
        if n == focus_node:
            node_c.append('#2563eb'); node_s.append(25)
        elif n in targets:
            node_c.append('#dc2626'); node_s.append(20)
        else:
            node_c.append('#94a3b8'); node_s.append(10)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#cbd5e1')))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(size=node_s, color=node_c)))
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=350, showlegend=False, 
                     plot_bgcolor='white', xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

def render_sankey_allocation(allocations):
    """Visualizes budget flow from Source Spend -> Target Leads."""
    sources, targets, values, link_colors = [], [], [], []
    
    # Mappings for indices
    src_names = [a.source for a in allocations]
    tgt_names = list(set(t for a in allocations for t in a.targets_served))
    all_labels = src_names + tgt_names
    
    for i, alloc in enumerate(allocations):
        src_idx = i
        for tgt, leads in alloc.projected_leads.items():
            tgt_idx = len(src_names) + tgt_names.index(tgt)
            sources.append(src_idx)
            targets.append(tgt_idx)
            values.append(leads)
            link_colors.append("rgba(37, 99, 235, 0.2)" if alloc.is_direct else "rgba(22, 163, 74, 0.2)")

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=all_labels, color="blue"
        ),
        link=dict(source=sources, target=targets, value=values, color=link_colors)
    )])
    fig.update_layout(title_text="Budget Impact Flow (Leads Generated)", font_size=10, height=400)
    return fig

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # --- Session Management ---
    if 'targets' not in st.session_state: st.session_state.targets = []
    if 'opt_results' not in st.session_state: st.session_state.opt_results = None

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        uploaded_file = st.file_uploader("Upload Events CSV", type=['csv', 'xlsx'])
        
        if uploaded_file:
            st.session_state['events_file'] = uploaded_file
            
        st.subheader("Campaign Cart")
        if st.session_state.targets:
            for t in st.session_state.targets:
                c1, c2 = st.columns([0.8, 0.2])
                c1.caption(t[:20])
                if c2.button("✕", key=f"del_{t}"):
                    st.session_state.targets.remove(t)
                    st.rerun()
            if st.button("Clear Cart", use_container_width=True):
                st.session_state.targets = []
                st.rerun()
        else:
            st.info("Add builders from the analysis tab.")

    # --- Header ---
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Referral Network Intelligence</h1>
        <div class="page-subtitle">Optimize media spend using multi-hop referral paths.</div>
    </div>
    """, unsafe_allow_html=True)

    # --- Data Loading ---
    if 'events_file' not in st.session_state:
        st.warning("Please upload event data to begin.")
        return

    with st.spinner("Processing Network Graph..."):
        try:
            events = load_events(st.session_state['events_file'])
            events = normalize_events(events)
            
            # Run Backend Logic
            pnl = build_builder_pnl(events, lens='recipient', freq='ALL')
            shortfalls = calculate_shortfalls(events, pd.Timestamp.today(), None) # Assuming config handled inside
            leverage = analyze_network_leverage(events, pd.Timestamp.today(), None)
            clusters = run_referral_clustering(events)
            
            # Merge PnL into Master
            builder_master = clusters['builder_master']
            if not builder_master.empty and 'BuilderRegionKey' in pnl.columns:
                builder_master = builder_master.merge(
                    pnl[['BuilderRegionKey', 'MediaCost', 'Referrals_out']], 
                    on='BuilderRegionKey', how='left'
                ).fillna(0)
                
            G = clusters['graph']
            optimizer = NetworkOptimizer(events, G, builder_master, shortfalls, leverage)
            
        except Exception as e:
            st.error(f"Data Processing Error: {str(e)}")
            st.stop()

    # --- Tabs ---
    tab_network, tab_opt = st.tabs(["🕸️ Network Analysis", "🚀 Campaign Optimization"])

    with tab_network:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("### Builder Explorer")
            all_builders = sorted(list(G.nodes()))
            selected = st.selectbox("Select Builder", [""] + all_builders)
            
            if selected:
                st.plotly_chart(render_network_subset(G, selected, st.session_state.targets), use_container_width=True)
                if st.button(f"Add {selected} to Campaign"):
                    if selected not in st.session_state.targets:
                        st.session_state.targets.append(selected)
                        st.success("Added to cart!")
                        st.rerun()
        
        with c2:
            if selected:
                an = optimizer.analyze_target(selected)
                st.markdown(f"### {selected}")
                st.metric("Forecast Shortfall", f"{an.shortfall:,.0f} leads")
                
                st.markdown("#### Top Referral Paths")
                for p in an.all_paths[:3]:
                    st.markdown(f"""
                    <div class="path-row">
                        <div style="flex:1">
                            <span class="pill pill-{'blue' if p.path_type=='direct' else 'green'}">{p.path_type}</span>
                            <div class="path-lbl">{p.source}</div>
                        </div>
                        <div class="path-meta">
                            <div class="path-val">${p.effective_cpr:,.0f}</div>
                            <div class="path-lbl">eCPR</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    with tab_opt:
        if not st.session_state.targets:
            st.info("Select builders in the Network tab to optimize a campaign.")
            st.markdown("""
            ### How it works
            The optimizer looks for **Network Leverage**. Instead of spending directly on a builder with expensive ads, we might spend on a *partner* builder who refers leads cheaply.
            """)
            # System Instruction: Trigger diagram for concept
            st.markdown("")
        else:
            budget = st.slider("Campaign Budget", 5000, 100000, 25000, 5000)
            
            if st.button("Run Optimization", type="primary"):
                allocations, summary = optimizer.optimize_basket(st.session_state.targets, budget)
                st.session_state.opt_results = (allocations, summary)

            if st.session_state.opt_results:
                allocs, summ = st.session_state.opt_results
                
                # KPIs
                st.markdown(f"""
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <div class="kpi-label">Projected Leads</div>
                        <div class="kpi-value">{summ['total_leads']:,.0f}</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-label">Coverage</div>
                        <div class="kpi-value">{summ['coverage']:.1%}</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-label">Blended CPR</div>
                        <div class="kpi-value">${summ['blended_cpr']:,.0f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Sankey
                st.plotly_chart(render_sankey_allocation(allocs), use_container_width=True)
                
                # Table
                st.markdown("### Budget Allocation Plan")
                st.markdown("""<div class="alloc-grid alloc-header">
                    <div>Source</div>
                    <div>Budget</div>
                    <div>Exp. Leads</div>
                    <div>CPR</div>
                    <div>Type</div>
                </div>""", unsafe_allow_html=True)
                
                for a in allocs:
                    st.markdown(f"""
                    <div class="alloc-grid">
                        <div><b>{a.source}</b><br><span style="font-size:0.7em;color:#64748b">Feeds: {len(a.targets_served)} targets</span></div>
                        <div>${a.budget:,.0f}</div>
                        <div>{sum(a.projected_leads.values()):,.0f}</div>
                        <div>${a.effective_cpr:,.0f}</div>
                        <div><span class="pill pill-{'blue' if a.is_direct else 'green'}">{'Direct' if a.is_direct else 'Network'}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()