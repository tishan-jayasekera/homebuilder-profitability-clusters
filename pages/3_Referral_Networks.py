"""
Referral Networks Dashboard - Streamlit Page
Filename: pages/3_Referral_Networks.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

import sys
from pathlib import Path

root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data_loader import load_events, export_to_excel
from src.normalization import normalize_events
from src.builder_pnl import build_builder_pnl
from src.referral_clusters import run_referral_clustering, compute_network_metrics
from src.utils import fmt_currency, fmt_roas
from src.network_optimization import (
    calculate_shortfalls, 
    analyze_network_leverage, 
    generate_investment_strategies,
    generate_global_media_plan,
    compute_effective_network_cpr
)

st.set_page_config(page_title="Referral Networks", page_icon="🔗", layout="wide")

# ==========================================
# STYLING
# ==========================================
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #0F172A; }
    .metric-label { font-size: 14px; color: #64748B; margin-bottom: 5px; }
    .priority-critical { color: #dc2626; font-weight: bold; }
    .priority-high { color: #ea580c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🔗 Referral Network & Optimization Engine")
st.markdown("Leverage network effects to close lead gaps efficiently.")


def load_data():
    if 'events_file' not in st.session_state:
        return None
    events = load_events(st.session_state['events_file'])
    if events is None:
        return None
    return normalize_events(events)

def get_date_range_info(events):
    """Identify date column and range."""
    date_col = 'lead_date' if 'lead_date' in events.columns else 'RefDate'
    if date_col not in events.columns:
        return None, [], None
        
    dates = pd.to_datetime(events[date_col], errors='coerce').dropna()
    if dates.empty:
        return date_col, [], None
        
    min_date = dates.min()
    max_date = dates.max()
    
    # Get list of month starts for slider
    month_starts = pd.date_range(start=min_date.replace(day=1), end=max_date.replace(day=1), freq='MS')
    return date_col, month_starts

def render_risk_matrix(shortfall_df):
    """Scatter plot: Urgency vs Shortfall Size."""
    if shortfall_df.empty:
        st.info("No shortfall data available.")
        return

    df = shortfall_df[shortfall_df['Projected_Shortfall'] > 0].copy()
    if df.empty:
        st.info("No builders currently at risk.")
        return

    # Create detailed label for granular data
    df['Label'] = df['BuilderRegionKey']
    if 'WIP_JOB_MATCHED' in df.columns:
        df['Label'] = df['Label'] + " (" + df['WIP_JOB_MATCHED'].astype(str).fillna('') + ")"
    elif 'Suburb' in df.columns:
        df['Label'] = df['Label'] + " (" + df['Suburb'].astype(str).fillna('') + ")"

    # Ensure Risk_Score is numeric for plotting
    df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce').fillna(0)

    fig = px.scatter(
        df,
        x="Days_Remaining",
        y="Projected_Shortfall",
        size="Risk_Score",
        color="Risk_Score",
        hover_name="Label",
        color_continuous_scale="RdYlGn_r",
        text="Label", # Use granular label
        title="Risk Matrix: Shortfall Size vs. Time Remaining",
        labels={"Days_Remaining": "Days Until Campaign End", "Projected_Shortfall": "Projected Lead Deficit"}
    )
    
    fig.update_traces(textposition='top center')
    fig.update_layout(height=400, plot_bgcolor='white')
    fig.add_vline(x=30, line_dash="dash", line_color="red", annotation_text="Critical Urgency")
    
    st.plotly_chart(fig, use_container_width=True)

def render_media_plan_table(plan_df):
    """Detailed action plan table."""
    if plan_df.empty:
        st.success("✅ No interventions required. All builders on track.")
        return

    # Metrics
    total_inv = plan_df['Est_Investment'].sum()
    leads_closed = plan_df['Gap_Leads'].sum()
    avg_cpr = total_inv / leads_closed if leads_closed > 0 else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Investment Required", fmt_currency(total_inv))
    c2.metric("Total Lead Gap to Close", f"{int(leads_closed):,}")
    c3.metric("Blended eCPR", fmt_currency(avg_cpr))

    st.markdown("### 📋 Master Media Plan")
    st.markdown("Recommended interventions to close projected gaps.")

    # Prepare display columns (dynamic based on granularity)
    cols = ['Priority', 'Target_Builder']
    if 'Project' in plan_df.columns: cols.append('Project')
    if 'Suburb' in plan_df.columns: cols.append('Suburb')
    cols.extend(['Gap_Leads', 'Recommended_Source', 'Action', 'Est_Investment', 'Effective_CPR', 'Strategy_Note'])
    
    # Filter only columns that actually exist
    cols = [c for c in cols if c in plan_df.columns]
    
    display_df = plan_df[cols].copy()
    
    # Format and Style - Safe approach
    styler = display_df.style.format({
        "Est_Investment": "${:,.0f}",
        "Effective_CPR": "${:,.0f}",
        "Gap_Leads": "{:,.0f}"
    }, na_rep="—")
    
    # Apply gradient only if column exists and has non-null values
    if 'Est_Investment' in display_df.columns:
        # Check if there's any non-null data to gradient
        if display_df['Est_Investment'].notna().any():
            styler = styler.background_gradient(subset=['Est_Investment'], cmap="Reds")
    
    # Table
    st.dataframe(
        styler,
        use_container_width=True,
        hide_index=True
    )
    
    # Download
    st.download_button(
        "📥 Download Media Plan CSV",
        plan_df.to_csv(index=False),
        "media_optimization_plan.csv",
        "text/csv"
    )

def main():
    events_full = load_data()
    
    if events_full is None:
        st.warning("⚠️ Please upload events data on the Home page first.")
        st.page_link("app.py", label="← Go to Home", icon="🏠")
        return

    # --- SIDEBAR CONTROLS ---
    date_col, available_months = get_date_range_info(events_full)
    
    with st.sidebar:
        st.header("📅 Time Period")
        st.caption("Adjusting the period recalculates clustering and lead velocity to account for recent network changes.")
        
        use_all_time = st.checkbox("Use All Time", value=True)
        
        if use_all_time or not len(available_months):
            events_selected = events_full
            # Default period days for velocity calc if using all time (approx)
            if date_col and not events_full.empty:
                dates = pd.to_datetime(events_full[date_col], errors='coerce')
                period_days = max((dates.max() - dates.min()).days, 1)
            else:
                period_days = 90
        else:
            # Range Slider
            start_month, end_month = st.select_slider(
                "Select Analysis Range",
                options=available_months,
                value=(available_months[0], available_months[-1]),
                format_func=lambda x: x.strftime("%b %Y")
            )
            
            # Filter Data
            end_date_filter = end_month + pd.offsets.MonthEnd(1)
            
            mask = (pd.to_datetime(events_full[date_col]) >= start_month) & (pd.to_datetime(events_full[date_col]) <= end_date_filter)
            events_selected = events_full.loc[mask].copy()
            
            period_days = max((end_date_filter - start_month).days, 1)
            
            st.info(f"Analyzing {period_days} days: {start_month.strftime('%b %Y')} to {end_month.strftime('%b %Y')}")
            
        st.metric("Events in Period", f"{len(events_selected):,}")
        st.divider()

    # --- TOP LEVEL CALCULATIONS ---
    # 1. Demand & Risk Analysis
    shortfall_data = calculate_shortfalls(
        events_df=events_selected,
        targets_df=None, 
        period_days=period_days,
        total_events_df=events_full
    )
    
    # 2. Leverage Analysis (Uses selected period to reflect CURRENT network health/turnover)
    leverage_data = analyze_network_leverage(events_selected)
    
    # 3. Global Plan Generation
    media_plan = generate_global_media_plan(shortfall_data, leverage_data)

    # --- TABS ---
    tab_plan, tab_explorer, tab_details = st.tabs(["🚀 Optimization Plan", "🕸️ Network Explorer", "🔍 Builder Deep Dive"])

    # ----------------------------------------
    # TAB 1: OPTIMIZATION PLAN
    # ----------------------------------------
    with tab_plan:
        st.header("Strategic Media Allocation")
        
        if shortfall_data.empty:
            st.info("No shortfall data available for the selected parameters.")
        else:
            # 1. Summary Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_shortfall = shortfall_data['Projected_Shortfall'].sum()
            builders_at_risk = shortfall_data[shortfall_data['Projected_Shortfall'] > 0]['BuilderRegionKey'].nunique()
            total_surplus = shortfall_data['Projected_Surplus'].sum()
            
            col1.metric("Total Projected Shortfall", f"{int(total_shortfall):,}", help="Leads needed across all builders")
            col2.metric("Builders At Risk", f"{builders_at_risk}", help="Count of builders missing targets")
            col3.metric("Available Surplus", f"{int(total_surplus):,}", help="Excess leads projected at other builders")
            
            # 2. Risk Matrix
            st.divider()
            col_risk, col_surplus = st.columns([2, 1])
            
            with col_risk:
                render_risk_matrix(shortfall_data)
                
            with col_surplus:
                st.subheader("Over-Serviced Targets")
                st.caption("Potential to reduce spend or shift media from these areas:")
                
                surplus_cols = ['BuilderRegionKey']
                if 'WIP_JOB_MATCHED' in shortfall_data.columns: surplus_cols.append('WIP_JOB_MATCHED')
                if 'Suburb' in shortfall_data.columns: surplus_cols.append('Suburb')
                surplus_cols.extend(['Projected_Surplus', 'LeadTarget'])
                
                surplus_df = shortfall_data[shortfall_data['Projected_Surplus'] > 0].sort_values('Projected_Surplus', ascending=False).head(10)
                
                # Check columns exist
                display_cols = [c for c in surplus_cols if c in surplus_df.columns]
                
                if not surplus_df.empty:
                    surplus_style = surplus_df[display_cols].style.format({
                        'Projected_Surplus': "{:,.0f}", 
                        'LeadTarget': "{:,.0f}"
                    }, na_rep="—")
                    
                    if 'Projected_Surplus' in display_cols:
                        surplus_style = surplus_style.background_gradient(cmap="Greens", subset=['Projected_Surplus'])
                        
                    st.dataframe(
                        surplus_style,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No over-serviced targets found.")

            # 3. The Plan
            st.divider()
            render_media_plan_table(media_plan)

    # ----------------------------------------
    # TAB 2: NETWORK EXPLORER
    # ----------------------------------------
    with tab_explorer:
        st.subheader("Ecosystem Clustering")
        
        # Clustering Controls
        c1, c2 = st.columns(2)
        with c1:
            resolution = st.slider("Resolution", 0.5, 2.5, 1.5, 0.1)
        with c2:
            target_clusters = st.slider("Max Clusters", 3, 25, 15, 1)
        
        # Run Clustering ON SELECTED PERIOD
        with st.spinner("Clustering network..."):
            results = run_referral_clustering(events_selected, resolution=resolution, target_max_clusters=target_clusters)
            
        edges_clean = results.get('edges_clean', pd.DataFrame())
        builder_master = results.get('builder_master', pd.DataFrame())
        cluster_summary = results.get('cluster_summary', pd.DataFrame())
        G = results.get('graph', nx.Graph())
        
        # Visualize Graph
        if not G.nodes:
            st.warning(f"No network data found in the selected period ({len(events_selected)} events).")
        else:
            # Cluster Select
            cluster_labels = {int(r.ClusterId): f"Cluster {int(r.ClusterId)} ({int(r.N_builders)} builders)" for r in cluster_summary.itertuples()}
            sel_cluster_id = st.selectbox("Select Ecosystem to Visualize", options=list(cluster_labels.keys()), format_func=lambda x: cluster_labels[x])
            
            # Filter Graph Data
            sub_builders = builder_master[builder_master["ClusterId"] == sel_cluster_id]
            sub_edges = edges_clean[(edges_clean["Cluster_origin"] == sel_cluster_id) & (edges_clean["Cluster_dest"] == sel_cluster_id)]
            
            if not sub_edges.empty:
                pos = nx.spring_layout(G, weight="weight", seed=42) # simple layout
                
                st.info(f"Visualizing Cluster {sel_cluster_id} with {len(sub_builders)} builders and {len(sub_edges)} connections.")
                
                # Simple Plotly Graph
                edge_x = []
                edge_y = []
                for _, row in sub_edges.iterrows():
                    if row['Origin_builder'] in pos and row['Dest_builder'] in pos:
                        x0, y0 = pos[row['Origin_builder']]
                        x1, y1 = pos[row['Dest_builder']]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])

                edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

                node_x = []
                node_y = []
                for node in sub_builders['BuilderRegionKey']:
                    if node in pos:
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)

                node_trace = go.Scatter(
                    x=node_x, y=node_y, mode='markers', hoverinfo='text',
                    marker=dict(showscale=True, colorscale='YlGnBu', size=10, color=sub_builders['ClusterId']),
                    text=sub_builders['BuilderRegionKey']
                )

                fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40)))
                st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------
    # TAB 3: BUILDER DEEP DIVE
    # ----------------------------------------
    with tab_details:
        st.header("Single Builder Analysis")
        
        if shortfall_data.empty:
            st.info("No data available.")
        else:
            # Select Builder
            at_risk = shortfall_data[shortfall_data['Projected_Shortfall'] > 0]['BuilderRegionKey'].tolist()
            all_builders = sorted(shortfall_data['BuilderRegionKey'].unique())
            
            # prioritize at risk in dropdown (deduplicate list)
            at_risk = sorted(list(set(at_risk)))
            sorted_opts = at_risk + [b for b in all_builders if b not in at_risk]
            
            sel_builder = st.selectbox("Select Builder", sorted_opts)
            
            if sel_builder:
                # Aggregate data for the selected builder across all their projects
                builder_subset = shortfall_data[shortfall_data['BuilderRegionKey'] == sel_builder]
                
                # Weighted average for Days Remaining? Just take max or min? Min is safest (most urgent).
                days_rem = builder_subset['Days_Remaining'].min()
                total_target = builder_subset['LeadTarget'].sum()
                total_proj = builder_subset['Projected_Total'].sum()
                total_gap = builder_subset['Net_Gap'].sum()
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Lead Target (All Projects)", int(total_target))
                c2.metric("Projected Total", int(total_proj), delta=int(total_gap))
                c3.metric("Min Days Remaining", int(days_rem))
                
                # Show breakdown if multiple projects
                if len(builder_subset) > 1:
                    st.subheader("Project/Suburb Breakdown")
                    disp_cols = ['LeadTarget', 'Actual_Referrals', 'Velocity_LeadsPerDay', 'Projected_Total', 'Projected_Shortfall']
                    if 'WIP_JOB_MATCHED' in builder_subset.columns: disp_cols.insert(0, 'WIP_JOB_MATCHED')
                    if 'Suburb' in builder_subset.columns: disp_cols.insert(1, 'Suburb')
                    
                    # Filter existing columns
                    disp_cols = [c for c in disp_cols if c in builder_subset.columns]
                    
                    st.dataframe(
                        builder_subset[disp_cols]
                        .style.format({'Velocity_LeadsPerDay': '{:.2f}', 'Projected_Total': '{:.1f}', 'Projected_Shortfall': '{:.1f}'})
                        .background_gradient(subset=['Projected_Shortfall'], cmap="Reds"),
                        use_container_width=True
                    )
                
                st.divider()
                
                # Strategies for this specific builder
                strats = generate_investment_strategies(sel_builder, shortfall_data, leverage_data, events_selected)
                
                if not strats.empty:
                    st.subheader("Available Inbound Pathways (Selected Period)")
                    st.dataframe(strats.style.format({'Effective_CPR': '${:,.0f}', 'Investment_Required': '${:,.0f}'}))
                else:
                    st.info("No inbound referral history found for this builder in the selected period.")

if __name__ == "__main__":
    main()