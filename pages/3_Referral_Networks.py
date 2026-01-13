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
    - If global view: Shows clusters with distinct colors.
    - If builder selected: Highlights ego-network (neighbors) and dims background.
    """
    fig = go.Figure()
    
    # --- PALETTE & STYLING ---
    # Distinct colors for clusters (up to 12)
    CLUSTER_COLORS = [
        '#6366f1', '#ec4899', '#10b981', '#f59e0b', '#3b82f6', 
        '#8b5cf6', '#ef4444', '#14b8a6', '#f97316', '#06b6d4',
        '#84cc16', '#a855f7'
    ]
    
    # Specific Role Colors (when builder selected)
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
    # We draw edges in two passes: Background (dimmed) and Foreground (highlighted)
    
    edge_x_dim, edge_y_dim = [], []
    edge_x_high, edge_y_high = [], []
    edge_colors_high = []
    
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos: continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        is_highlight = False
        color = ROLE_COLORS['dim']
        
        if selected_builder:
            # Check if edge connects to selected builder
            if u == selected_builder or v == selected_builder:
                is_highlight = True
                # Determine flow direction color based on the neighbor's role
                neighbor = v if u == selected_builder else u
                role = role_map.get(neighbor, 'dim')
                color = ROLE_COLORS.get(role, ROLE_COLORS['dim'])
        
        if is_highlight:
            # Add None to break lines in single scatter trace
            edge_x_high.extend([x0, x1, None])
            edge_y_high.extend([y0, y1, None])
            # For colored edges in a single trace, we need separate traces or a trick.
            # To keep it simple but performant, we'll draw highlighted edges individually later
            # or just use a uniform highlight color if strictly necessary. 
            # Better approach for detailed coloring: draw individual lines for highlights.
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines',
                line=dict(color=color, width=1.5),
                opacity=0.8, hoverinfo='skip', showlegend=False
            ))
        else:
            edge_x_dim.extend([x0, x1, None])
            edge_y_dim.extend([y0, y1, None])

    # Draw dimmed background edges as a single trace (efficient)
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
    node_borders = []
    
    degrees = dict(G.degree(weight='weight'))
    max_deg = max(degrees.values()) if degrees else 1
    
    for node in G.nodes():
        if node not in pos: continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Sizing
        deg = degrees.get(node, 0)
        base_size = 8 + (deg / max_deg) * 15
        
        # Coloring Logic
        if selected_builder:
            if node in highlight_nodes:
                role = role_map.get(node, 'dim')
                c = ROLE_COLORS.get(role, ROLE_COLORS['dim'])
                s = base_size + 5 if node == selected_builder else base_size + 2
                op = 1.0
                border = 'white' if node == selected_builder else c
                
                # Descriptive text
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
                op = 0.3
                border = 'white'
                txt = f"{node}"
        else:
            # Global View: Color by Cluster
            cid = cluster_map.get(node, 0)
            c = CLUSTER_COLORS[(cid - 1) % len(CLUSTER_COLORS)] if cid > 0 else '#9ca3af'
            s = base_size
            op = 0.9
            border = 'white'
            txt = f"<b>{node}</b><br>Cluster {cid}<br>Volume: {int(deg)}"

        node_colors.append(c)
        node_sizes.append(s)
        node_texts.append(txt)
        node_borders.append(border)

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
    
    # Layout
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode='closest',
        annotations=[
            dict(
                text="<b>Network Map</b>" if not selected_builder else f"<b>Ego Network: {selected_builder}</b>",
                x=0.01, y=0.99, xref="paper", yref="paper",
                showarrow=False, align="left", font=dict(size=14, color="#374151")
            )
        ]
    )
    
    return fig

def render_builder_panel(builder, connections, shortfall_df, leverage_df):
    """Refined builder detail panel."""
    
    # 1. Header & Risk Status
    row = shortfall_df[shortfall_df['BuilderRegionKey'] == builder]
    risk_score = 0
    shortfall = 0
    
    if not row.empty:
        risk_score = row['Risk_Score'].iloc[0]
        shortfall = row['Projected_Shortfall'].iloc[0]
    
    # Risk Badge
    if risk_score > 50:
        badge = f'<span class="badge badge-red">Risk: {int(risk_score)}</span>'
        status_text = f"CRITICAL GAP: {int(shortfall):,} leads"
    elif risk_score > 20:
        badge = f'<span class="badge badge-yellow">Risk: {int(risk_score)}</span>'
        status_text = f"At Risk: {int(shortfall):,} gap"
    else:
        badge = f'<span class="badge badge-green">Safe</span>'
        status_text = "On Track"

    st.markdown(f"""
    <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <h3 style="margin:0; font-size:1.1rem; color:#111827;">{builder}</h3>
            {badge}
        </div>
        <div style="margin-top:0.5rem; font-size:0.9rem; color:#4b5563;">
            {status_text}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Commercial Data Tabs
    tab_in, tab_out, tab_partners = st.tabs(["📥 Supply (In)", "📤 Demand (Out)", "🤝 Partners"])
    
    with tab_in:
        if connections['inbound']:
            df = pd.DataFrame(connections['inbound'])
            df = df.rename(columns={'partner': 'Source', 'count': 'Vol', 'value': 'Spend'})
            
            # Show Efficiency if available
            lev_subset = leverage_df[
                (leverage_df['Dest_BuilderRegionKey'] == builder) & 
                (leverage_df['MediaPayer_BuilderRegionKey'].isin(df['Source']))
            ]
            if not lev_subset.empty:
                df = df.merge(lev_subset[['MediaPayer_BuilderRegionKey', 'eCPR']], 
                              left_on='Source', right_on='MediaPayer_BuilderRegionKey', how='left')
                df = df.drop(columns=['MediaPayer_BuilderRegionKey'])
            
            st.dataframe(
                df.style.format({'Spend': '${:,.0f}', 'eCPR': '${:,.0f}'})
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
                # Categorize efficiency
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
        shortfall_df = calculate_shortfalls(events_filtered, period_days=period_days)
        leverage_df = analyze_network_leverage(events_filtered)
        cluster_res = run_referral_clustering(events_filtered, target_max_clusters=12)
        G = cluster_res.get('graph', nx.Graph())
        builder_master = cluster_res.get('builder_master', pd.DataFrame())

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
            render_builder_panel(st.session_state.selected_builder, conns, shortfall_df, leverage_df)
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