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
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
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
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
        margin-top: 0.25rem;
    }
    .metric-sub {
        font-size: 0.875rem;
        color: #9ca3af;
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
    
    /* Eco Styles (Ported) */
    .eco-pill-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
    .eco-pill { display: inline-flex; align-items: center; gap: 6px; font-size: 11px; padding: 3px 10px; border-radius: 999px; background: #F3F4F6; color: #4B5563; }
    .eco-pill-dot { width: 6px; height: 6px; border-radius: 999px; background: currentColor; }
    .eco-pill--primary { background: #DBEAFE; color: #1D4ED8; }
    .eco-pill--positive { background: #DCFCE7; color: #15803D; }
    .eco-pill--warning { background: #FEF3C7; color: #92400E; }
    .eco-pill--negative { background: #FEE2E2; color: #B91C1C; }
    
    .eco-metric-row { display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px; margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #f3f4f6; }
    .eco-metric { font-weight: 500; color: #6B7280; }
    .eco-metric span { font-weight: 600; color: #111827; margin-left: 4px; }
    
    .eco-bullets { font-size: 13px; line-height: 1.5; color: #374151; }
    .eco-bullets ul { margin: 4px 0; padding-left: 18px; }
    .eco-bullets li { margin-bottom: 4px; }
    .eco-lever-heading { font-weight: 600; margin-top: 8px; color: #111827; }
    .eco-downstream { color: #6B7280; font-size: 12px; display: block; margin-top: 2px; }

    /* Section Headers */
    .step-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .step-number {
        background: #e5e7eb;
        color: #374151;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
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
            direct["Leverage"] = 100 * (1 - (m - mn) / (mx - mn if mx != mn else 1))
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

    # Best path
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
    
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos: continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        is_highlight = False
        color = ROLE_COLORS['dim']
        
        if selected_builder:
            if u == selected_builder or v == selected_builder:
                is_highlight = True
                neighbor = v if u == selected_builder else u
                role = role_map.get(neighbor, 'dim')
                color = ROLE_COLORS.get(role, ROLE_COLORS['dim'])
        
        if is_highlight:
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines',
                line=dict(color=color, width=1.5),
                opacity=0.8, hoverinfo='skip', showlegend=False
            ))
        else:
            edge_x_dim.extend([x0, x1, None])
            edge_y_dim.extend([y0, y1, None])

    # Background edges
    fig.add_trace(go.Scatter(
        x=edge_x_dim, y=edge_y_dim,
        mode='lines',
        line=dict(color='#e5e7eb', width=0.5),
        opacity=0.3 if selected_builder else 0.4,
        hoverinfo='skip', showlegend=False
    ))

    # --- 2. NODES ---
    node_x, node_y = [], []
    node_colors = []
    node_sizes = []
    node_texts = []
    
    degrees = dict(G.degree(weight='weight'))
    max_deg = max(degrees.values()) if degrees else 1
    
    for node in G.nodes():
        if node not in pos: continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        deg = degrees.get(node, 0)
        base_size = 8 + (deg / max_deg) * 15
        
        if selected_builder:
            if node in highlight_nodes:
                role = role_map.get(node, 'dim')
                c = ROLE_COLORS.get(role, ROLE_COLORS['dim'])
                s = base_size + 5 if node == selected_builder else base_size + 2
                
                role_txt = {
                    'selected': 'SELECTED',
                    'source': 'Source (Inbound)',
                    'target': 'Dest (Outbound)',
                    'mutual': 'Partner (Two-way)'
                }.get(role, '')
                txt = f"<b>{node}</b><br>{role_txt}<br>Volume: {int(deg)}"
            else:
                c = ROLE_COLORS['dim']
                s = base_size * 0.8
                txt = f"{node}"
        else:
            cid = cluster_map.get(node, 0)
            c = CLUSTER_COLORS[(cid - 1) % len(CLUSTER_COLORS)] if cid > 0 else '#9ca3af'
            s = base_size
            txt = f"<b>{node}</b><br>Cluster {cid}<br>Volume: {int(deg)}"

        node_colors.append(c)
        node_sizes.append(s)
        node_texts.append(txt)

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color='white'),
            opacity=1.0 if not selected_builder else [1.0 if n in highlight_nodes else 0.2 for n in G.nodes()]
        ),
        text=node_texts,
        hoverinfo='text',
        showlegend=False
    ))
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode='closest'
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
        badge = f'<span class="badge badge-red">Risk: {int(risk_score)}</span>'
        status_text = f"CRITICAL GAP: {int(shortfall):,} leads"
    elif risk_score > 20:
        badge = f'<span class="badge badge-yellow">Risk: {int(risk_score)}</span>'
        status_text = f"At Risk: {int(shortfall):,} gap"
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
        bullets += f"<li><b>Recommended:</b> {path_text} <span style='color:#6B7280'>({eff_txt})</span></li>"

    # Self media bullet
    if not pd.isna(raw_self):
        eff_self_txt = "n/a" if pd.isna(eff_self) else f"${eff_self:,.0f}"
        bullets += f"<li class='eco-lever-heading'>Self Media Efficiency</li>"
        bullets += f"<ul><li><b>{builder}</b><br>– CPR: ${raw_self:,.0f}<br>– Eff. Cost: {eff_self_txt}</li></ul>"

    # Primary Levers (Top 3 Direct)
    if direct is not None and not direct.empty:
        bullets += "<li class='eco-lever-heading'>Top Media Levers (Direct)</li><ul>"
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

    card_html = f"""
    <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <h3 style="margin:0; font-size:1.2rem; color:#111827;">{builder}</h3>
            {badge}
        </div>
        <div class="eco-pill-row">
            <span class="eco-pill eco-pill--primary"><span class="eco-pill-dot"></span>Cluster {cid}</span>
            <span class="eco-pill"><span class="eco-pill-dot"></span>{status_text}</span>
        </div>
        <div class="eco-metric-row">
            <div class="eco-metric">Profit: <span>${profit:,.0f}</span></div>
            <div class="eco-metric">ROAS: <span>{roas:.2f}x</span></div>
        </div>
        <div class="eco-bullets">
            <ul>{bullets}</ul>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
    
    # 4. Detailed Flow Tables
    tab_in, tab_out, tab_partners = st.tabs(["📥 Supply (In)", "📤 Demand (Out)", "🤝 Partners"])
    
    with tab_in:
        if connections['inbound']:
            df = pd.DataFrame(connections['inbound'])
            df = df.rename(columns={'partner': 'Source', 'count': 'Vol', 'value': 'Spend'})
            # Merge efficiency if available in direct guidance
            if direct is not None and not direct.empty:
                df = df.merge(direct[['Origin_builder', 'Eff_MPR']], left_on='Source', right_on='Origin_builder', how='left')
                df = df.drop(columns=['Origin_builder'])
                df.rename(columns={'Eff_MPR': 'Eff. Cost'}, inplace=True)
            
            st.dataframe(
                df.style.format({'Spend': '${:,.0f}', 'Eff. Cost': '${:,.0f}'})
                .background_gradient(subset=['Vol'], cmap='Greens'),
                hide_index=True, use_container_width=True, height=200
            )
        else:
            st.caption("No inbound lead sources found.")

    with tab_out:
        if connections['outbound']:
            df = pd.DataFrame(connections['outbound'])
            df = df.rename(columns={'partner': 'Destination', 'count': 'Vol'})
            st.dataframe(
                df.style.background_gradient(subset=['Vol'], cmap='Oranges'),
                hide_index=True, use_container_width=True, height=200
            )
        else:
            st.caption("No outbound referrals found.")

    with tab_partners:
        if connections['two_way']:
            df = pd.DataFrame(connections['two_way'])
            df.columns = ['Partner', 'In', 'Out']
            st.dataframe(df, hide_index=True, use_container_width=True, height=200)
        else:
            st.caption("No mutual partnerships found.")

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
def render_campaign_planner(targets, shortfall_df, leverage_df):
    if not targets:
        st.info("👆 Add builders to the campaign cart (Sidebar) to unlock the planner.")
        return

    st.markdown("## 🚀 Execution Plan")
    st.caption("Turn insight into action: Allocate budget to the most efficient supply sources.")
    
    # Run analysis
    analysis = analyze_campaign_network(targets, leverage_df, shortfall_df)
    sources = analysis['sources']
    
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
    
    with c1:
        st.markdown(f"""
        <div class="card" style="height:100%">
            <div class="metric-label">Campaign Goal</div>
            <div class="metric-value text-red-600">{int(total_shortfall):,}</div>
            <div class="metric-sub">Leads needed</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="card" style="height:100%">
            <div class="metric-label">Network Power</div>
            <div class="metric-value">{len(sources)}</div>
            <div class="metric-sub">Available sources</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown('<div class="card" style="height:100%">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Budget Allocation</div>', unsafe_allow_html=True)
        budget = st.slider("Total Spend ($)", 5000, 500000, 50000, step=5000, label_visibility="collapsed")
        
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

        # --- STEP 3: ACTION LIST ---
        st.markdown('<div class="step-header"><div class="step-number">3</div><span>Buy List (Allocation)</span></div>', unsafe_allow_html=True)
        
        allocs = pd.DataFrame(sim['allocations'])
        
        if not allocs.empty:
            # Justification Logic
            def generate_justification(row):
                cpr = row['effective_cpr']
                rate = row['target_rate']
                if cpr < 300: eff_desc = "Highly efficient"
                elif cpr < 600: eff_desc = "Cost-effective"
                else: eff_desc = "Strategic"
                
                return f"{eff_desc} source (${int(cpr)}/lead). {int(rate*100)}% of volume reaches targets."

            allocs['Justification'] = allocs.apply(generate_justification, axis=1)

            # Prepare clean table
            df_disp = allocs.rename(columns={
                'source': 'Source Builder',
                'budget': 'Budget',
                'leads_to_targets': 'Leads (Target)',
                'effective_cpr': 'CPR'
            })
            
            # Recommendation Text
            top_source = df_disp.iloc[0]
            st.info(f"💡 **Strategy:** Allocate **${int(top_source['Budget']):,}** to **{top_source['Source Builder']}** as your primary driver. They offer the best efficiency at ${int(top_source['CPR'])}/lead.")

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
                    "Justification": st.column_config.TextColumn("Investment Rationale", width="medium"),
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
        leverage_df
    )

if __name__ == "__main__":
    main()