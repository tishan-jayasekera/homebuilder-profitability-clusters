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

# Add project root to path
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
    /* Cards */
    .card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* Metrics */
    .metric-container {
        display: flex;
        flex-direction: column;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 4px;
    }
    
    /* Headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 1rem;
        border-bottom: 2px solid #f3f4f6;
        padding-bottom: 0.5rem;
    }
    
    /* Risk Tags */
    .tag-critical { background-color: #fef2f2; color: #dc2626; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; border: 1px solid #fecaca; }
    .tag-warning { background-color: #fffbeb; color: #d97706; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; border: 1px solid #fde68a; }
    .tag-safe { background-color: #f0fdf4; color: #16a34a; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; border: 1px solid #bbf7d0; }
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
def render_metric_card(label, value, sub="", color="text-gray-900"):
    """HTML Helper for metric cards."""
    st.markdown(f"""
    <div class="metric-container">
        <span class="metric-label">{label}</span>
        <span class="metric-value" style="color: {color}">{value}</span>
        <span class="metric-sub">{sub}</span>
    </div>
    """, unsafe_allow_html=True)

def render_network_map(G, pos, connections, selected_builder=None):
    fig = go.Figure()
    
    # Theme Colors
    COL_TWO_WAY = '#3b82f6'    # Blue
    COL_INBOUND = '#10b981'    # Emerald
    COL_OUTBOUND = '#f59e0b'   # Amber
    COL_SELECTED = '#ef4444'   # Red
    COL_BG = '#f3f4f6'
    
    two_way_set, inbound_set, outbound_set = set(), set(), set()
    if selected_builder and connections:
        two_way_set = {c['partner'] for c in connections.get('two_way', [])}
        inbound_set = {c['partner'] for c in connections.get('inbound', [])}
        outbound_set = {c['partner'] for c in connections.get('outbound', [])}
    
    # 1. Background Edges (faded)
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos: continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        # Logic to dim non-relevant edges
        is_relevant = False
        if selected_builder:
            if u == selected_builder or v == selected_builder: is_relevant = True
        
        color = '#e5e7eb' if not is_relevant else '#9ca3af'
        opacity = 0.1 if not is_relevant else 0.0 # We draw relevant ones later
        
        if not is_relevant:
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines', line=dict(color=color, width=0.5),
                opacity=opacity, hoverinfo='skip', showlegend=False
            ))

    # 2. Highlighted Edges for Selected
    if selected_builder:
        def draw_link(target, color, width=1.5):
            if target not in pos: return
            x0, y0 = pos[selected_builder]
            x1, y1 = pos[target]
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines', line=dict(color=color, width=width),
                opacity=0.8, hoverinfo='skip', showlegend=False
            ))

        for n in two_way_set: draw_link(n, COL_TWO_WAY, 2.5)
        for n in inbound_set: draw_link(n, COL_INBOUND, 1.5)
        for n in outbound_set: draw_link(n, COL_OUTBOUND, 1.5)

    # 3. Nodes
    node_x, node_y, node_color, node_size = [], [], [], []
    hover_texts = []
    
    for node in G.nodes():
        if node not in pos: continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Color logic
        if node == selected_builder:
            c = COL_SELECTED
            s = 25
            txt = f"<b>{node}</b><br>SELECTED"
        elif node in two_way_set:
            c = COL_TWO_WAY
            s = 15
            txt = f"<b>{node}</b><br>↔ Partner"
        elif node in inbound_set:
            c = COL_INBOUND
            s = 12
            txt = f"<b>{node}</b><br>→ Supplier"
        elif node in outbound_set:
            c = COL_OUTBOUND
            s = 12
            txt = f"<b>{node}</b><br>← Receiver"
        else:
            c = '#d1d5db'
            s = 8
            txt = f"{node}"
            
        node_color.append(c)
        node_size.append(s)
        hover_texts.append(txt)

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(size=node_size, color=node_color, line=dict(width=1, color='white')),
        text=hover_texts, hoverinfo='text', showlegend=False
    ))
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=450,
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode='closest'
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
        badge = f'<span class="tag-critical">Risk: {int(risk_score)}</span>'
        status_color = "#dc2626"
        status_text = f"CRITICAL SHORTFALL ({int(shortfall):,} leads)"
    elif risk_score > 20:
        badge = f'<span class="tag-warning">Risk: {int(risk_score)}</span>'
        status_color = "#d97706"
        status_text = f"At Risk ({int(shortfall):,} leads)"
    else:
        badge = f'<span class="tag-safe">Risk: {int(risk_score)}</span>'
        status_color = "#16a34a"
        status_text = "On Track"

    st.markdown(f"""
    <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <h3 style="margin:0; font-size:1.2rem;">{builder}</h3>
            {badge}
        </div>
        <div style="margin-top:5px; font-size:0.9rem; color:{status_color}; font-weight:600;">
            {status_text}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Commercial Data Tabs
    tab_in, tab_out, tab_partners = st.tabs(["📥 Supply (In)", "📤 Demand (Out)", "🤝 Partners"])
    
    with tab_in:
        st.caption("Who sends leads TO this builder?")
        if connections['inbound']:
            df = pd.DataFrame(connections['inbound'])
            df = df.rename(columns={'partner': 'Source', 'count': 'Volume', 'value': 'Media Value'})
            
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
                df.style.format({'Media Value': '${:,.0f}', 'eCPR': '${:,.0f}'})
                .background_gradient(subset=['Volume'], cmap='Greens'),
                hide_index=True, use_container_width=True
            )
        else:
            st.info("No inbound lead sources found.")

    with tab_out:
        st.caption("Who does this builder send leads TO?")
        if connections['outbound']:
            df = pd.DataFrame(connections['outbound'])
            df = df.rename(columns={'partner': 'Destination', 'count': 'Volume'})
            st.dataframe(
                df.style.background_gradient(subset=['Volume'], cmap='Oranges'),
                hide_index=True, use_container_width=True
            )
        else:
            st.info("No outbound referrals found.")

    with tab_partners:
        st.caption("Two-way mutual referral partners.")
        if connections['two_way']:
            df = pd.DataFrame(connections['two_way'])
            df.columns = ['Partner', 'Received', 'Sent']
            st.dataframe(df, hide_index=True, use_container_width=True)
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
        st.sidebar.info("Add builders to cart to plan a campaign.")
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
    <div style="background:#f3f4f6; padding:15px; border-radius:8px; margin-bottom:15px; text-align:center;">
        <div style="font-size:0.8rem; color:#6b7280; text-transform:uppercase;">Total Shortfall</div>
        <div style="font-size:1.4rem; font-weight:700; color:#dc2626;">{int(total_gap):,}</div>
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
            st.markdown(f"<span style='font-size:0.8rem; color:#6b7280;'>{dot_color} Gap: {int(item['gap'])}</span>", unsafe_allow_html=True)
        
        with c2:
            if st.button("✕", key=f"rm_{item['name']}", help="Remove"):
                remove_from_cart(item['name'])
                st.rerun()
    
    if st.sidebar.button("Clear Cart", type="secondary", use_container_width=True):
        st.session_state.campaign_targets = []
        st.rerun()

# ==========================================
# CAMPAIGN PLANNER (WIZARD STYLE)
# ==========================================
def render_campaign_planner(targets, shortfall_df, leverage_df, G, pos):
    if not targets:
        st.info("👆 Please add builders to the campaign cart to unlock the planner.")
        return

    st.markdown("## 🚀 Campaign Planner")
    
    # Run analysis
    analysis = analyze_campaign_network(targets, leverage_df, shortfall_df)
    sources = analysis['sources']
    
    # --- STEP 1: SCOPE ---
    with st.expander("Step 1: Campaign Scope & Network", expanded=True):
        c1, c2, c3 = st.columns(3)
        
        total_shortfall = sum(
            shortfall_df[shortfall_df['BuilderRegionKey'] == t]['Projected_Shortfall'].iloc[0]
            for t in targets if not shortfall_df[shortfall_df['BuilderRegionKey'] == t].empty
        )
        
        render_metric_card("Total Shortfall", f"{int(total_shortfall):,}", "Leads needed", "text-red-600")
        with c2: render_metric_card("Available Sources", len(sources), "Network partners")
        with c3: render_metric_card("Capture Rate", f"{analysis['stats']['target_capture_rate']:.0%}", "Efficiency")
    
    if not sources:
        st.warning("No historical sources found for these targets. Cannot optimize media.")
        return

    # --- STEP 2: BUDGET & SIMULATION ---
    st.markdown("### Step 2: Budget Allocation")
    
    col_sim_left, col_sim_right = st.columns([1, 2])
    
    with col_sim_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("⚙️ Simulation Params")
        budget = st.slider("Total Budget ($)", 5000, 500000, 50000, step=5000)
        
        if st.button("🔄 Run Simulation", type="primary", use_container_width=True):
            sim = simulate_campaign_spend(targets, budget, sources, shortfall_df)
            st.session_state.campaign_simulation = sim
        st.markdown("</div>", unsafe_allow_html=True)

    with col_sim_right:
        if 'campaign_simulation' in st.session_state:
            sim = st.session_state.campaign_simulation
            summ = sim['summary']
            
            # Results Cards
            rc1, rc2, rc3 = st.columns(3)
            with rc1: render_metric_card("Leads Generated", f"{int(summ['leads_to_targets']):,}", "To Targets")
            with rc2: render_metric_card("Coverage", f"{summ['coverage_pct']:.1%}", "Of Shortfall")
            with rc3: render_metric_card("Effective CPR", f"${int(summ['effective_cpr'])}", "Cost per Lead")
            
            # Progress Bar
            st.markdown(f"""
            <div style="margin-top:15px;">
                <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#6b7280; margin-bottom:5px;">
                    <span>Gap Filled: {int(summ['shortfall_covered']):,}</span>
                    <span>Target: {int(summ['target_shortfall']):,}</span>
                </div>
                <div style="width:100%; background:#e5e7eb; height:8px; border-radius:4px;">
                    <div style="width:{min(summ['coverage_pct']*100, 100)}%; background:#10b981; height:8px; border-radius:4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👈 Set budget and click 'Run Simulation' to see results.")

    # --- STEP 3: PLAN DETAILS ---
    if 'campaign_simulation' in st.session_state:
        st.markdown("### Step 3: Execution Plan")
        
        sim = st.session_state.campaign_simulation
        allocs = pd.DataFrame(sim['allocations'])
        
        if not allocs.empty:
            allocs = allocs.rename(columns={
                'source': 'Source Builder',
                'budget': 'Budget Allocation',
                'leads_to_targets': 'Target Leads',
                'leads_leaked': 'Leakage',
                'effective_cpr': 'Eff. CPR'
            })
            
            st.dataframe(
                allocs[['Source Builder', 'Budget Allocation', 'Target Leads', 'Eff. CPR', 'Leakage']],
                column_config={
                    "Budget Allocation": st.column_config.ProgressColumn(
                        "Budget", format="$%d", min_value=0, max_value=int(allocs['Budget Allocation'].max())
                    ),
                    "Target Leads": st.column_config.NumberColumn("Target Leads", format="%d"),
                    "Eff. CPR": st.column_config.NumberColumn("CPR", format="$%d"),
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Export
            csv = allocs.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Campaign Plan (CSV)",
                csv,
                "campaign_plan.csv",
                "text/csv",
                key='download-csv'
            )

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
    events_filtered = events.copy() # (simplified filtering for demo)
    
    with st.spinner("Analyzing Ecosystem..."):
        shortfall_df = calculate_shortfalls(events_filtered, period_days=90)
        leverage_df = analyze_network_leverage(events_filtered)
        cluster_res = run_referral_clustering(events_filtered, target_max_clusters=12)
        G = cluster_res.get('graph', nx.Graph())

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
    if st.session_state.selected_builder:
        b = st.session_state.selected_builder
        
        c1, c2 = st.columns([1.8, 1])
        
        with c1:
            st.markdown(f"#### Network Map: {b}")
            pos = nx.spring_layout(G, seed=42, k=0.8)
            conns = get_builder_connections(events_filtered, b)
            fig = render_network_map(G, pos, conns, b)
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            render_builder_panel(b, conns, shortfall_df, leverage_df)

    else:
        st.info("Select a builder above to explore their network relationships.")

    st.markdown("---")
    
    # 3. Campaign Planner (Full Width)
    render_campaign_planner(
        st.session_state.campaign_targets, 
        shortfall_df, 
        leverage_df, 
        G, 
        nx.spring_layout(G, seed=42) # Re-calc pos for planner
    )

if __name__ == "__main__":
    main()