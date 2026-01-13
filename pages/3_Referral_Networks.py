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

# Add parent directory to path
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
    analyze_network_health, simulate_campaign_spend,
    analyze_campaign_network
)

st.set_page_config(page_title="Network Intelligence", page_icon="🔗", layout="wide")

# ==========================================
# STYLING & CSS
# ==========================================
STYLES = """
<style>
    /* Modern Card Style */
    .card {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.5rem;
        transition: transform 0.2s;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Metrics */
    .metric-container {
        display: flex;
        flex-direction: column;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7280;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
        line-height: 1;
    }
    .metric-sub {
        font-size: 0.875rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }

    /* Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        line-height: 1;
    }
    .badge-red { background-color: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
    .badge-yellow { background-color: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
    .badge-green { background-color: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
    .badge-blue { background-color: #eff6ff; color: #1e40af; border: 1px solid #dbeafe; }
    
    /* Eco Styles */
    .eco-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        border-bottom: 1px solid #f3f4f6;
        padding-bottom: 0.75rem;
    }
    .eco-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #111827;
        margin: 0;
    }
    
    .eco-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .eco-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    .eco-list li {
        padding: 0.75rem 0;
        border-bottom: 1px solid #f3f4f6;
        font-size: 0.9rem;
        color: #374151;
    }
    .eco-list li:last-child {
        border-bottom: none;
    }
    .eco-list b {
        color: #111827;
        font-weight: 600;
    }
    .eco-lever-heading {
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: #111827;
        font-size: 0.95rem;
        display: block;
    }
    .eco-downstream {
        color: #6b7280;
        font-size: 0.85rem;
        display: block;
        margin-top: 0.25rem;
        font-style: italic;
    }

    /* Section Headers */
    .step-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #111827;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .step-number {
        background: #1f2937;
        color: white;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    /* Custom Scrollbar for tables if needed */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1; 
    }
    ::-webkit-scrollbar-thumb {
        background: #888; 
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555; 
    }
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

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
    
    result = {
        'two_way': [], 'inbound': [], 'outbound': []
    }
    
    for _, row in merged.iterrows():
        partner = row['partner']
        if partner in two_way:
            result['two_way'].append({'partner': partner, 'in': int(row['refs_in']), 'out': int(row['refs_out'])})
        elif partner in inbound_only:
            result['inbound'].append({'partner': partner, 'count': int(row['refs_in']), 'value': row['value_in']})
        elif partner in outbound_only:
            result['outbound'].append({'partner': partner, 'count': int(row['refs_out'])})
    
    result['two_way'] = sorted(result['two_way'], key=lambda x: x['in'] + x['out'], reverse=True)
    result['inbound'] = sorted(result['inbound'], key=lambda x: x['count'], reverse=True)
    result['outbound'] = sorted(result['outbound'], key=lambda x: x['count'], reverse=True)
    
    return result

# ==========================================
# LOGIC: GUIDANCE ENGINE
# ==========================================
def compute_focus_guidance(builders, edges_sub, focus_builder):
    """
    Compute targeted media guidance:
    - Direct paths (Leverage)
    - Indirect paths (1-hop)
    - Self media efficiency
    """
    if not focus_builder or focus_builder not in builders["BuilderRegionKey"].values:
        return None, None, set(), set(), None, None, None, np.nan, np.nan

    # Inbound referrals to focus builder
    inbound = edges_sub.loc[
        edges_sub["Dest_builder"] == focus_builder, "Referrals"
    ].sum()

    # Even if inbound is 0, we might want self stats
    fb = builders.set_index("BuilderRegionKey").loc[focus_builder]

    # Self path
    if fb["Referrals_in"] > 0 and fb["MediaCost"] > 0:
        raw_self = fb["MediaCost"] / fb["Referrals_in"]
        eff_self = raw_self / max(fb["ROAS"], 1e-9)
    else:
        raw_self = np.nan
        eff_self = np.nan

    # Direct paths
    direct = edges_sub[edges_sub["Dest_builder"] == focus_builder].copy()
    if not direct.empty:
        # Merge metrics from origin builders
        direct = direct.merge(
            builders[["BuilderRegionKey", "ROAS", "MediaCost", "Profit"]],
            left_on="Origin_builder",
            right_on="BuilderRegionKey",
            how="left"
        )

        direct["Inbound_share"] = direct["Referrals"] / inbound if inbound > 0 else 0
        direct["Raw_MPR"] = direct["MediaCost"] / direct["Referrals"].replace(0, np.nan)
        direct["Eff_MPR"] = direct["Raw_MPR"] / direct["ROAS"].replace(0, np.nan)

        m = direct["Eff_MPR"].replace([np.inf, -np.inf], np.nan)
        if m.notna().any():
            mx, mn = m.max(), m.min()
            denom = mx - mn if mx != mn else 1
            direct["Leverage"] = 100 * (1 - (m - mn) / denom)
        else:
            direct["Leverage"] = 0

        direct = direct.sort_values("Eff_MPR")
        direct_sources = set(direct["Origin_builder"].astype(str))
    else:
        direct = pd.DataFrame()
        direct_sources = set()

    # Indirect paths (1 hop)
    mids = list(direct_sources)
    second = pd.DataFrame()
    second_sources = set()
    
    if mids:
        second = edges_sub[edges_sub["Dest_builder"].isin(mids)].copy()

    if not second.empty:
        mid_ref_map = direct.set_index("Origin_builder")["Referrals"].to_dict()

        second["Ref_mid_to_focus"] = second["Dest_builder"].map(mid_ref_map).fillna(0)
        tot_mid = (
            second.groupby("Dest_builder")["Referrals"]
            .transform("sum")
            .replace(0, np.nan)
        )

        second["Ref_to_focus_est"] = np.where(
            tot_mid.isna(),
            0,
            (second["Referrals"] / tot_mid) * second["Ref_mid_to_focus"]
        )

        second = second.merge(
            builders[["BuilderRegionKey", "ROAS", "MediaCost"]],
            left_on="Origin_builder",
            right_on="BuilderRegionKey",
            how="left"
        )

        second["Raw_MPR_est"] = np.where(
            second["Ref_to_focus_est"] <= 0,
            np.nan,
            second["MediaCost"] / second["Ref_to_focus_est"]
        )

        second["Eff_MPR"] = second["Raw_MPR_est"] / second["ROAS"].replace(0, np.nan)

        second["PathLev_raw"] = (
            second["Referrals"].fillna(0) *
            second["Ref_mid_to_focus"].fillna(0) *
            second["ROAS"].clip(lower=0).fillna(0)
        )

        mx = second["PathLev_raw"].max()
        second["PathLev"] = (
            (second["PathLev_raw"] / mx) * 100 if (mx and mx > 0) else 0
        )

        second = second.sort_values("Eff_MPR")
        second_sources = set(second["Origin_builder"].astype(str))
    else:
        second = None
        second_sources = set()

    # Best path selection
    cands = [("Self", eff_self, focus_builder)]

    if len(direct) > 0:
        r = direct.iloc[0]
        cands.append(("Direct", r["Eff_MPR"], r["Origin_builder"]))

    if second is not None and len(second) > 0:
        r = second.iloc[0]
        cands.append(("Indirect", r["Eff_MPR"], r["Origin_builder"]))

    cands_valid = [c for c in cands if not pd.isna(c[1])]
    best_choice = min(cands_valid, key=lambda x: x[1]) if cands_valid else None

    # Downstream from focus
    focus_out = edges_sub[edges_sub["Origin_builder"] == focus_builder].copy()
    if not focus_out.empty:
        tot = focus_out["Referrals"].sum()
        focus_out["Out_share"] = np.where(
            tot > 0,
            focus_out["Referrals"] / tot,
            0
        )
        focus_out = focus_out.sort_values("Referrals", ascending=False)
    else:
        focus_out = None

    return (
        direct,
        second,
        direct_sources,
        second_sources,
        focus_out,
        cands,
        best_choice,
        raw_self,
        eff_self,
    )

# ==========================================
# VISUALIZATION COMPONENTS
# ==========================================
def render_metric_card(label, value, sub="", color="#111827"):
    """HTML Helper for metric cards."""
    st.markdown(f"""
    <div class="metric-container">
        <span class="metric-label">{label}</span>
        <span class="metric-value" style="color: {color}">{value}</span>
        <span class="metric-sub">{sub}</span>
    </div>
    """, unsafe_allow_html=True)

def render_network_map(G, pos, builder_master_df, selected_builder=None, connections=None):
    """
    High-end "Palantir-style" graph visualization.
    """
    fig = go.Figure()
    
    # --- PALETTE & STYLING ---
    CLUSTER_COLORS = [
        '#6366f1', '#ec4899', '#10b981', '#f59e0b', '#3b82f6', 
        '#8b5cf6', '#ef4444', '#14b8a6', '#f97316', '#06b6d4',
        '#84cc16', '#a855f7'
    ]
    
    # Specific Role Colors
    ROLE_COLORS = {
        'source': '#10b981',   # Emerald (Inbound)
        'target': '#f59e0b',   # Amber (Outbound)
        'mutual': '#3b82f6',   # Blue (Two-way)
        'selected': '#ef4444', # Red (Focus)
        'dim': '#e5e7eb'       # Grey (Background)
    }

    # Map nodes to Cluster ID
    cluster_map = {}
    if not builder_master_df.empty and 'ClusterId' in builder_master_df.columns:
        cluster_map = builder_master_df.set_index('BuilderRegionKey')['ClusterId'].to_dict()

    # Determine Highlight Sets
    highlight_nodes = set()
    role_map = {}
    
    if selected_builder and connections:
        highlight_nodes.add(selected_builder)
        role_map[selected_builder] = 'selected'
        
        for c in connections['inbound']:
            highlight_nodes.add(c['partner'])
            role_map[c['partner']] = 'source'
        for c in connections['outbound']:
            highlight_nodes.add(c['partner'])
            role_map[c['partner']] = 'target'
        for c in connections['two_way']:
            highlight_nodes.add(c['partner'])
            role_map[c['partner']] = 'mutual'

    # --- 1. EDGES ---
    edge_x_dim, edge_y_dim = [], []
    edge_x_hi, edge_y_hi = [], []
    edge_cols_hi = []
    edge_widths_hi = []
    
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos: continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data.get('weight', 1)
        
        is_highlight = False
        color = ROLE_COLORS['dim']
        width = 0.5
        
        if selected_builder:
            if u == selected_builder or v == selected_builder:
                is_highlight = True
                neighbor = v if u == selected_builder else u
                role = role_map.get(neighbor, 'dim')
                color = ROLE_COLORS.get(role, ROLE_COLORS['dim'])
                width = 2.0 + (weight/10) # scale by weight
        
        if is_highlight:
            fig.add_trace(go.Scattergl(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines',
                line=dict(color=color, width=width),
                opacity=0.9, hoverinfo='skip', showlegend=False
            ))
        else:
            edge_x_dim.extend([x0, x1, None])
            edge_y_dim.extend([y0, y1, None])

    # Background edges (one trace for performance)
    fig.add_trace(go.Scattergl(
        x=edge_x_dim, y=edge_y_dim,
        mode='lines',
        line=dict(color='#e5e7eb', width=0.5),
        opacity=0.3 if selected_builder else 0.5,
        hoverinfo='skip', showlegend=False
    ))

    # --- 2. NODES ---
    node_x, node_y = [], []
    node_colors = []
    node_sizes = []
    node_texts = []
    node_lines = []
    
    degrees = dict(G.degree(weight='weight'))
    max_deg = max(degrees.values()) if degrees else 1
    
    for node in G.nodes():
        if node not in pos: continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        deg = degrees.get(node, 0)
        # Sizing based on degree
        base_size = 12 + (deg / max_deg) * 25
        border_col = 'white'
        border_width = 1
        
        cid = cluster_map.get(node, 0)
        
        if selected_builder:
            if node in highlight_nodes:
                role = role_map.get(node, 'dim')
                c = ROLE_COLORS.get(role, ROLE_COLORS['dim'])
                s = base_size + 5 if node == selected_builder else base_size + 2
                op = 1.0
                border_width = 2
                
                # Halo for selected
                if node == selected_builder:
                    border_col = '#111827'
                    border_width = 3
                    s += 5
                
                role_txt = {
                    'selected': 'SELECTED',
                    'source': 'Source (Inbound)',
                    'target': 'Dest (Outbound)',
                    'mutual': 'Partner (Two-way)'
                }.get(role, '')
                txt = f"<b>{node}</b><br>{role_txt}<br>Volume: {int(deg)}"
            else:
                c = CLUSTER_COLORS[(cid - 1) % len(CLUSTER_COLORS)] if cid > 0 else '#9ca3af'
                s = base_size * 0.7
                op = 0.2
                txt = f"{node}"
        else:
            c = CLUSTER_COLORS[(cid - 1) % len(CLUSTER_COLORS)] if cid > 0 else '#9ca3af'
            s = base_size
            op = 0.9
            border_col = 'white'
            txt = f"<b>{node}</b><br>Cluster {cid}<br>Volume: {int(deg)}"

        node_colors.append(c)
        node_sizes.append(s)
        node_texts.append(txt)
        node_lines.append(dict(color=border_col, width=border_width))

    # Node trace
    fig.add_trace(go.Scattergl(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(color=[l['color'] for l in node_lines], width=[l['width'] for l in node_lines]),
            opacity=1.0 if not selected_builder else [1.0 if n in highlight_nodes else 0.35 for n in G.nodes()]
        ),
        text=node_texts,
        hoverinfo='text',
        showlegend=False
    ))
    
    # Add Halo for selected builder
    if selected_builder and selected_builder in pos:
        sx, sy = pos[selected_builder]
        fig.add_trace(go.Scattergl(
            x=[sx], y=[sy],
            mode='markers',
            marker=dict(size=node_sizes[list(G.nodes()).index(selected_builder)] + 15, color=ROLE_COLORS['selected'], opacity=0.2),
            hoverinfo='skip', showlegend=False
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode='closest',
        dragmode='pan'
    )
    
    return fig

def render_builder_panel(builder, connections, shortfall_df, leverage_df, builders_df, edges_df):
    """
    Enhanced builder detail panel with Guidance Engine.
    Uses 'compute_focus_guidance' logic.
    """
    # 1. Run Guidance Engine
    (
        direct, second, direct_srcs, second_srcs, focus_out,
        cands, best_choice, raw_self, eff_self
    ) = compute_focus_guidance(builders_df, edges_df, builder)

    # 2. Risk Status
    row = shortfall_df[shortfall_df['BuilderRegionKey'] == builder]
    risk_score = 0
    shortfall = 0
    
    if not row.empty:
        risk_score = row['Risk_Score'].iloc[0]
        shortfall = row['Projected_Shortfall'].iloc[0]
    
    if risk_score > 50:
        badge = f'<span class="badge badge-red">Critical Risk: {int(risk_score)}</span>'
        status_text = f"Shortfall: <b>{int(shortfall):,} leads</b>"
    elif risk_score > 20:
        badge = f'<span class="badge badge-yellow">At Risk: {int(risk_score)}</span>'
        status_text = f"Shortfall: <b>{int(shortfall):,} leads</b>"
    else:
        badge = f'<span class="badge badge-green">Safe</span>'
        status_text = "On Track"

    # 3. Build HTML Summary Card
    bullets = ""
    
    # Recommended path bullet
    if best_choice:
        lbl, eff, src = best_choice
        eff_txt = "n/a" if pd.isna(eff) else f"~${eff:,.0f}/eff-ref"
        path_text = f"Self direct" if lbl == "Self" else f"{lbl} via <b>{src}</b>"
        bullets += f"<li><b>Strategy:</b> {path_text} <span style='color:#6B7280; font-size:0.85em'>({eff_txt})</span></li>"

    # Self media bullet
    if not pd.isna(raw_self):
        eff_self_txt = "n/a" if pd.isna(eff_self) else f"${eff_self:,.0f}"
        
        # --- Downstream Impact Simulation ---
        self_downstream_txt = ""
        if focus_out is not None and not focus_out.empty and eff_self and eff_self > 0:
            # Simulate +$10k spend: leads generated = 10,000 / eff_self
            incremental_leads = 10000 / eff_self
            impacts = []
            
            # Show top 3 downstream beneficiaries
            for _, r in focus_out.head(3).iterrows():
                dest = r["Dest_builder"]
                share = r["Out_share"]
                leads_passed = incremental_leads * share
                if leads_passed >= 0.5: # Show if significant
                    impacts.append(f"{leads_passed:.0f} → {dest}")
            
            if impacts:
                self_downstream_txt = (
                    f"<span class='eco-downstream'>Downstream from +$10k media at self: "
                    f"{' | '.join(impacts)}</span>"
                )
        # ------------------------------------

        bullets += f"<li class='eco-lever-heading'>Self Media Efficiency</li>"
        bullets += f"<ul><li><b>{builder}</b><br>– CPR: ${raw_self:,.0f}<br>– Eff. Cost: {eff_self_txt}<br>{self_downstream_txt}</li></ul>"

    # Primary Levers (Top 3 Direct)
    if direct is not None and not direct.empty:
        bullets += "<li class='eco-lever-heading'>Top Leverage Partners (Direct)</li><ul>"
        for _, r in direct.head(3).iterrows():
            src = r["Origin_builder"]
            eff = r["Eff_MPR"]
            sh = r["Inbound_share"]
            eff_txt = "n/a" if pd.isna(eff) else f"${eff:,.0f}"
            bullets += f"<li><b>{src}</b><br>– Share: {sh:.0%}<br>– Eff. Cost: {eff_txt}</li>"
        bullets += "</ul>"

    # Determine Cluster & Stats
    b_data = builders_df[builders_df['BuilderRegionKey'] == builder]
    cid = b_data['ClusterId'].iloc[0] if not b_data.empty else "?"
    profit = b_data['Profit'].iloc[0] if not b_data.empty else 0
    roas = b_data['ROAS'].iloc[0] if not b_data.empty else 0

    # Cluster Analytics
    cluster_analytics_html = ""
    if cid != "?":
        cluster_members = builders_df[builders_df['ClusterId'] == cid]
        cluster_size = len(cluster_members)
        cluster_profit = cluster_members['Profit'].sum()
        cluster_referrals_in = cluster_members['Referrals_in'].sum() if 'Referrals_in' in cluster_members.columns else 0
        
        # Calculate Flows
        # Internal vs External Inbound
        # This requires edges data which we have in edges_df
        # Filter edges where Dest is in cluster
        cluster_nodes = set(cluster_members['BuilderRegionKey'])
        cluster_inbound_edges = edges_df[edges_df['Dest_builder'].isin(cluster_nodes)]
        
        internal_flow = cluster_inbound_edges[cluster_inbound_edges['Origin_builder'].isin(cluster_nodes)]['Referrals'].sum()
        external_flow = cluster_inbound_edges[~cluster_inbound_edges['Origin_builder'].isin(cluster_nodes)]['Referrals'].sum()
        total_flow = internal_flow + external_flow
        
        internal_pct = (internal_flow / total_flow * 100) if total_flow > 0 else 0
        
        cluster_analytics_html = f"""
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #f3f4f6;">
            <div class="eco-lever-heading" style="margin-top:0;">Cluster {cid} Analytics</div>
            <div class="eco-metric-row" style="border-bottom: none; margin-bottom: 0;">
                <div class="eco-metric" style="font-size: 0.8rem;">Size: <span>{cluster_size}</span></div>
                <div class="eco-metric" style="font-size: 0.8rem;">Profit: <span>${cluster_profit:,.0f}</span></div>
            </div>
            <div class="eco-metric-row" style="border-bottom: none;">
                <div class="eco-metric" style="font-size: 0.8rem;">Flow: <span>{internal_pct:.0f}% Internal</span></div>
                <div class="eco-metric" style="font-size: 0.8rem;">Vol: <span>{total_flow:,.0f} Refs</span></div>
            </div>
        </div>
        """

    card_html = f"""
    <div class="card">
        <div class="eco-header">
            <h3 class="eco-title">{builder}</h3>
            {badge}
        </div>
        <div class="eco-grid">
            <div class="eco-pill-row">
                <span class="eco-pill eco-pill--primary"><span class="eco-pill-dot"></span>Cluster {cid}</span>
            </div>
            <div>{status_text}</div>
        </div>
        <div class="eco-metric-row">
            <div class="eco-metric">Profit: <span>${profit:,.0f}</span></div>
            <div class="eco-metric">ROAS: <span>{roas:.2f}x</span></div>
        </div>
        <div class="eco-bullets">
            <ul class="eco-list">{bullets}</ul>
        </div>
        {cluster_analytics_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
    
    # 4. Detailed Flow Tabs
    tab_in, tab_out, tab_partners = st.tabs(["📥 Supply (In)", "📤 Demand (Out)", "🤝 Partners"])
    
    with tab_in:
        # Use full connections, not just direct guidance
        if connections['inbound']:
            df = pd.DataFrame(connections['inbound'])
            # Ensure proper renaming and column existence
            df = df.rename(columns={'partner': 'Source', 'count': 'Volume', 'value': 'Spend'})
            
            # Merge efficiency from direct guidance (if available)
            if direct is not None and not direct.empty:
                df = df.merge(direct[['Origin_builder', 'Eff_MPR']], left_on='Source', right_on='Origin_builder', how='left')
                df = df.drop(columns=['Origin_builder'])
                df.rename(columns={'Eff_MPR': 'Eff. Cost'}, inplace=True)
            
            # Format columns
            format_dict = {'Spend': '${:,.0f}'}
            if 'Eff. Cost' in df.columns:
                format_dict['Eff. Cost'] = '${:,.0f}'
                
            st.dataframe(
                df.style.format(format_dict)
                .background_gradient(subset=['Volume'], cmap='Greens'),
                hide_index=True, use_container_width=True, height=200
            )
        else:
            st.caption("No inbound lead sources found.")

    with tab_out:
        if connections['outbound']:
            df = pd.DataFrame(connections['outbound'])
            df = df.rename(columns={'partner': 'Destination', 'count': 'Volume'})
            st.dataframe(
                df.style.background_gradient(subset=['Volume'], cmap='Oranges'),
                hide_index=True, use_container_width=True, height=200
            )
        else:
            st.info("No outbound referrals found.")

    with tab_partners:
        if connections['two_way']:
            df = pd.DataFrame(connections['two_way'])
            df.columns = ['Partner', 'In', 'Out']
            st.dataframe(df, hide_index=True, use_container_width=True, height=200)
        else:
            st.info("No mutual partnerships found.")

# ==========================================
# CAMPAIGN CART (SIDEBAR)
# ==========================================
def render_sidebar_cart(shortfall_df):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛒 Campaign Cart")
    
    targets = st.session_state.campaign_targets
    
    if not targets:
        st.sidebar.info("Add builders to plan.")
        return
    
    # Calculate Total Gap
    total_gap = 0
    cart_data = []
    
    for t in targets:
        row = shortfall_df[shortfall_df['BuilderRegionKey'] == t]
        gap = 0
        risk = 0
        if not row.empty:
            gap = row['Projected_Shortfall'].iloc[0] if row['Projected_Shortfall'].iloc[0] > 0 else 0
            risk = row['Risk_Score'].iloc[0]
        
        total_gap += gap
        cart_data.append({'name': t, 'gap': gap, 'risk': risk})
    
    # Summary Card
    st.sidebar.markdown(f"""
    <div style="background:#f9fafb; padding:1rem; border:1px solid #e5e7eb; border-radius:8px; margin-bottom:1rem; text-align:center;">
        <div style="font-size:0.75rem; color:#6b7280; text-transform:uppercase; font-weight:600;">Total Shortfall</div>
        <div style="font-size:1.5rem; font-weight:700; color:#dc2626;">{int(total_gap):,}</div>
        <div style="font-size:0.75rem; color:#6b7280;">Leads needed</div>
    </div>
    """, unsafe_allow_html=True)
    
    # List Items
    cart_data.sort(key=lambda x: x['risk'], reverse=True)
    
    for item in cart_data:
        c1, c2 = st.sidebar.columns([0.85, 0.15])
        
        # Risk Dot
        dot_color = "🔴" if item['risk'] > 50 else ("🟠" if item['risk'] > 20 else "🟢")
        
        with c1:
            st.markdown(f"**{item['name'][:18]}..**")
            st.caption(f"{dot_color} Gap: {int(item['gap'])}")
        
        with c2:
            if st.button("✕", key=f"rm_{item['name']}", help="Remove"):
                remove_from_cart(item['name'])
                st.rerun()
    
    if st.sidebar.button("Clear Cart", use_container_width=True):
        st.session_state.campaign_targets = []
        st.rerun()

# ==========================================
# CAMPAIGN PLANNER (EXECUTION PLAN)
# ==========================================
def render_campaign_planner(targets, shortfall_df, leverage_df, builder_master):
    if not targets:
        st.info("👆 Add builders to the campaign cart (Sidebar) to unlock the planner.")
        return

    st.markdown("## 🚀 Execution Plan")
    st.caption("Turn insight into action: Allocate budget to the most efficient supply sources.")
    
    # Run analysis
    analysis = analyze_campaign_network(targets, leverage_df, shortfall_df)
    sources = analysis['sources']
    
    # Inject "Self Spend" options into sources list
    # For each target, check their own self-media efficiency
    # If better than avg network CPR, add as a "source"
    for t in targets:
        b_data = builder_master[builder_master['BuilderRegionKey'] == t]
        if not b_data.empty:
            b_row = b_data.iloc[0]
            # Calculate Self CPR (Effective)
            if b_row['Referrals_in'] > 0 and b_row['MediaCost'] > 0:
                raw_self = b_row['MediaCost'] / b_row['Referrals_in']
                eff_self = raw_self / max(b_row['ROAS'], 1e-9)
                
                # Add to sources list as a "Self Direct" option
                sources.append({
                    'source': f"SELF ({t})",
                    'total_refs_sent': 0, # Not relevant for self
                    'refs_to_targets': 999, # Infinite capacity assumption for now
                    'refs_to_others': 0,
                    'target_rate': 1.0, # 100% hits target
                    'leakage_rate': 0.0,
                    'base_cpr': raw_self,
                    'effective_cpr': eff_self
                })
    
    # Re-sort sources including new Self options
    sources = sorted(sources, key=lambda x: x['effective_cpr'])
    
    if not sources:
        st.warning("No historical sources found for these targets. Cannot optimize media.")
        return

    # --- STEP 1: PARAMETERS ---
    st.markdown('<div class="step-header"><div class="step-number">1</div><span>Define Scope & Budget</span></div>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    
    # Metric Scope
    total_shortfall = sum(
        shortfall_df[shortfall_df['BuilderRegionKey'] == t]['Projected_Shortfall'].iloc[0]
        for t in targets if not shortfall_df[shortfall_df['BuilderRegionKey'] == t].empty
    )
    
    # Estimate budget required to fully cover shortfall
    # Simple logic: assume we use the best available CPRs
    needed = total_shortfall
    est_budget = 0
    for src in sources:
        if needed <= 0: break
        
        # Assume source can provide up to 50 leads or 20% of its volume, whichever is safer
        # For 'Self', assume capacity is flexible
        if "SELF" in src['source']:
            capacity = needed # unlimited for self
        else:
            capacity = max(src['total_refs_sent'] * 0.5, 20) # heuristic cap
            
        take = min(needed, capacity)
        cost = take * src['effective_cpr']
        est_budget += cost
        needed -= take
    
    if est_budget == 0 and total_shortfall > 0:
        est_budget = 50000 # Fallback default
        
    with c1:
        st.markdown(f"""
        <div class="card" style="height:100%; padding: 1.25rem;">
            <div class="metric-label">Campaign Goal</div>
            <div class="metric-value text-red-600">{int(total_shortfall):,}</div>
            <div class="metric-sub">Leads needed</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="card" style="height:100%; padding: 1.25rem;">
            <div class="metric-label">Network Power</div>
            <div class="metric-value">{len(sources)}</div>
            <div class="metric-sub">Available sources</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown('<div class="card" style="height:100%; padding: 1.25rem;">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Budget Allocation</div>', unsafe_allow_html=True)
        # Use number input with default value = est_budget
        budget = st.number_input("Total Spend ($)", min_value=1000, value=int(est_budget), step=5000, label_visibility="collapsed")
        
        if st.button("Generate Plan ⚡", type="primary", use_container_width=True):
            sim = simulate_campaign_spend(targets, budget, sources, shortfall_df)
            st.session_state.campaign_simulation = sim
        st.markdown('</div>', unsafe_allow_html=True)

    # --- STEP 2: RESULTS ---
    if 'campaign_simulation' in st.session_state:
        st.markdown('<div class="step-header"><div class="step-number">2</div><span>Review Outcomes</span></div>', unsafe_allow_html=True)
        
        sim = st.session_state.campaign_simulation
        summ = sim['summary']
        
        # Outcomes Grid
        rc1, rc2, rc3 = st.columns(3)
        with rc1: render_metric_card("Projected Leads", f"{int(summ['leads_to_targets']):,}", "To Targets")
        with rc2: render_metric_card("Gap Coverage", f"{summ['coverage_pct']:.1%}", f"{int(summ['shortfall_covered']):,} covered")
        with rc3: render_metric_card("Effective CPR", f"${int(summ['effective_cpr'])}", "Cost per Lead")
        
        # Visual Progress
        st.markdown(f"""
        <div style="background:white; padding:15px; border-radius:8px; border:1px solid #e5e7eb; margin-top:10px; margin-bottom:20px;">
            <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#4b5563; margin-bottom:8px; font-weight:600;">
                <span>Progress to Goal</span>
                <span>{int(summ['shortfall_covered']):,} / {int(summ['target_shortfall']):,} leads</span>
            </div>
            <div style="width:100%; background:#f3f4f6; height:12px; border-radius:6px; overflow:hidden;">
                <div style="width:{min(summ['coverage_pct']*100, 100)}%; background:#10b981; height:100%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # SANKEY DIAGRAM
        st.markdown("#### Media Flow")
        # Budget -> Sources -> [Targets, Leakage]
        # Nodes: 0=Budget, 1..N=Sources, N+1=Targets, N+2=Leakage
        
        allocs = sim['allocations']
        # Filter tiny allocations for cleaner chart
        allocs = [a for a in allocs if a['budget'] > 100]
        
        if allocs:
            labels = ["Budget"] + [a['source'] for a in allocs] + ["Targets", "Leakage"]
            colors = ["#6366f1"] + ["#3b82f6"] * len(allocs) + ["#16a34a", "#ef4444"]
            
            budget_idx = 0
            targets_idx = len(allocs) + 1
            leakage_idx = len(allocs) + 2
            
            sources = []
            targets_sankey = []
            values = []
            
            # Budget -> Source links
            for i, a in enumerate(allocs):
                source_idx = i + 1
                sources.append(budget_idx)
                targets_sankey.append(source_idx)
                values.append(a['budget']) # Flow is money
                
                # Source -> Targets (convert leads back to equivalent money flow or just use leads? Usually clearer to stick to one unit. Let's use LEADS for the second step, but Sankey requires consistent units for flow continuity.
                # Actually, budget flows into source, source produces LEADS. This is a unit change. 
                # Better to show BUDGET flow: Budget -> Source -> [Effective Spend, Wasted Spend]
                # Effective Spend = Cost of leads that hit target. Wasted = Cost of leakage.
                
                eff_spend = a['leads_to_targets'] * a['effective_cpr']
                leak_spend = a['budget'] - eff_spend
                
                if eff_spend > 0:
                    sources.append(source_idx)
                    targets_sankey.append(targets_idx)
                    values.append(eff_spend)
                
                if leak_spend > 0:
                    sources.append(source_idx)
                    targets_sankey.append(leakage_idx)
                    values.append(leak_spend)

            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15, thickness=20, line=dict(color="black", width=0.5),
                    label=labels, color=colors
                ),
                link=dict(
                    source=sources, target=targets_sankey, value=values,
                    color='rgba(200, 200, 200, 0.3)'
                )
            )])
            fig_sankey.update_layout(title_text="Budget Efficiency Flow", font_size=10, height=400)
            st.plotly_chart(fig_sankey, use_container_width=True)

        # --- STEP 3: ACTION LIST ---
        st.markdown('<div class="step-header"><div class="step-number">3</div><span>Buy List (Allocation)</span></div>', unsafe_allow_html=True)
        
        alloc_df = pd.DataFrame(sim['allocations'])
        
        if not alloc_df.empty:
            # Justification Logic
            def generate_justification(row):
                cpr = row['effective_cpr']
                rate = row['target_rate']
                src_name = row['source']
                
                # Check if it's a self-spend row
                if "SELF" in src_name:
                    builder_name = src_name.replace('SELF (', '').replace(')', '')
                    return f"Direct Buy: Most efficient path. Buying directly for {builder_name} beats referral costs."
                
                # Efficiency Text
                if cpr < 300: eff_desc = "Highly efficient"
                elif cpr < 600: eff_desc = "Cost-effective"
                else: eff_desc = "Strategic"
                
                # Target identification
                # Simple logic: say "Serves targets" or be specific if we tracked which targets specifically
                # For now, generic text is safer than guessing which specific target in the set it hits most without granular data
                target_text = "campaign targets"
                
                return f"{eff_desc} source (${int(cpr)}/lead). Delivers volume to {target_text} with {int(rate*100)}% precision."

            alloc_df['Justification'] = alloc_df.apply(generate_justification, axis=1)

            # Prepare clean table
            df_disp = alloc_df.rename(columns={
                'source': 'Source Builder',
                'budget': 'Budget',
                'leads_to_targets': 'Leads (Target)',
                'effective_cpr': 'CPR'
            })
            
            # Recommendation Text
            top_source = df_disp.iloc[0]
            st.info(f"💡 Strategy: Allocate ${int(top_source['Budget']):,} to {top_source['Source Builder']} as your primary driver. They offer the best efficiency at ${int(top_source['CPR'])}/lead.")

            # Data Editor with Progress Bars
            st.dataframe(
                df_disp[['Source Builder', 'Budget', 'Leads (Target)', 'CPR', 'Justification']],
                column_config={
                    "Budget": st.column_config.ProgressColumn(
                        "Budget Allocation",
                        format="$%d",
                        min_value=0,
                        max_value=int(df_disp['Budget'].max()),
                    ),
                    "Leads (Target)": st.column_config.NumberColumn("Est. Leads", format="%d"),
                    "CPR": st.column_config.NumberColumn("Eff. CPR", format="$%d"),
                    "Justification": st.column_config.TextColumn("Investment Rationale", width="large"),
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Downloads
            csv = df_disp.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Buy List (CSV)",
                csv,
                "buy_list.csv",
                "text/csv",
                key='download-csv'
            )
    else:
        st.info("👈 Set a budget and click 'Generate Plan' to see the execution strategy.")

# ==========================================
# MAIN APP FLOW
# ==========================================
def main():
    st.title("🔗 Network Intelligence")
    
    # Load Data
    events = load_and_process()
    if events is None:
        st.warning("⚠️ Please upload 'Events Master' on the Home page.")
        return

    # Global Settings
    with st.sidebar:
        st.markdown("## ⚙️ Filters")
        dates = pd.to_datetime(events['lead_date'], errors='coerce').dropna()
        if not dates.empty:
            min_d, max_d = dates.min().date(), dates.max().date()
            dr = st.date_input("Period", value=(min_d, max_d))
        else:
            dr = None

    # Processing
    events_filtered = events.copy() 
    if dr and len(dr) == 2:
        start_date, end_date = pd.Timestamp(dr[0]), pd.Timestamp(dr[1])
        mask = (events_filtered['lead_date'] >= start_date) & (events_filtered['lead_date'] <= end_date)
        events_filtered = events_filtered.loc[mask]
        period_days = (end_date - start_date).days
    else:
        period_days = 90
    
    with st.spinner("Mapping Ecosystem..."):
        # 1. P&L for stats
        pnl = build_builder_pnl(events_filtered, lens='recipient', freq='ALL')
        
        # 2. Shortfalls
        shortfall_df = calculate_shortfalls(events_filtered, period_days=period_days)
        
        # 3. Network & Clusters
        leverage_df = analyze_network_leverage(events_filtered)
        cluster_res = run_referral_clustering(events_filtered, target_max_clusters=12)
        G = cluster_res.get('graph', nx.Graph())
        
        # 4. Enriched Master Data
        builder_master = cluster_res.get('builder_master', pd.DataFrame())
        if not builder_master.empty:
            # Merge P&L metrics into builder_master for graph tooltips and logic
            if 'BuilderRegionKey' in pnl.columns:
                pnl_subset = pnl[['BuilderRegionKey', 'Profit', 'ROAS', 'MediaCost']].copy()
                builder_master = builder_master.merge(pnl_subset, on='BuilderRegionKey', how='left')
                builder_master[['Profit', 'ROAS', 'MediaCost']] = builder_master[['Profit', 'ROAS', 'MediaCost']].fillna(0)
        
        # 5. Edges for Guidance
        edges = cluster_res.get('edges_clean', pd.DataFrame())

    # Sidebar Cart
    render_sidebar_cart(shortfall_df)

    # 1. Search & Add
    all_builders = get_all_builders(events_filtered)
    col_search, col_add = st.columns([4, 1])
    with col_search:
        selected = st.selectbox("Find Builder", [""] + all_builders, index=0)
    with col_add:
        # Spacer for alignment
        st.write("")
        st.write("") 
        if st.button("➕ Add to Cart", type="secondary", use_container_width=True, disabled=not selected):
            add_to_cart(selected)
            st.rerun()

    if selected:
        st.session_state.selected_builder = selected

    # 2. Network Visualizer
    st.markdown("### Ecosystem Map")
    c_graph, c_detail = st.columns([2, 1])
    
    with c_graph:
        pos = nx.spring_layout(G, seed=42, k=0.6)
        conns = get_builder_connections(events_filtered, st.session_state.selected_builder) if st.session_state.selected_builder else None
        
        fig = render_network_map(G, pos, builder_master, st.session_state.selected_builder, conns)
        st.plotly_chart(fig, use_container_width=True)
        
    with c_detail:
        if st.session_state.selected_builder:
            render_builder_panel(
                st.session_state.selected_builder, 
                conns, 
                shortfall_df, 
                leverage_df,
                builder_master,
                edges
            )
        else:
            st.info("Select a builder to see specific flows and supply/demand data.")
            st.markdown(f"""
            <div class="card">
                <div class="metric-label">Total Network Nodes</div>
                <div class="metric-value">{len(G.nodes)}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    # 3. Campaign Planner (Full Width)
    render_campaign_planner(
        st.session_state.campaign_targets, 
        shortfall_df, 
        leverage_df,
        builder_master # Pass builder_master for self-spend check
    )

if __name__ == "__main__":
    main()