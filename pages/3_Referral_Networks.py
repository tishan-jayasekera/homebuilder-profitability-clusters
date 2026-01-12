"""
Referral Networks Dashboard - Streamlit Page
Filename: pages/3_Referral_Networks.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go

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
# NEW: Import optimization logic
from src.network_optimization import calculate_shortfalls, compute_effective_network_cpr

st.set_page_config(page_title="Referral Networks", page_icon="🔗", layout="wide")

st.title("🔗 Referral Network Explorer")
st.markdown("Discover referral ecosystems and media efficiency pathways.")


def run_clustering_for_period(events_df, resolution, target_clusters, period_key):
    """Run clustering - period_key ensures recalculation when month changes."""
    return run_referral_clustering(events_df, resolution=resolution, target_max_clusters=target_clusters)


def load_data():
    if 'events_file' not in st.session_state:
        return None
    events = load_events(st.session_state['events_file'])
    if events is None:
        return None
    return normalize_events(events)


def get_available_months(events):
    """Extract available months from events data."""
    months = []
    date_col = None
    
    if "lead_date" in events.columns:
        date_col = "lead_date"
    elif "RefDate" in events.columns:
        date_col = "RefDate"
    
    if date_col:
        dates = pd.to_datetime(events[date_col], errors="coerce")
        month_starts = dates.dt.to_period("M").dt.start_time.dropna().unique()
        months = sorted(month_starts)
    
    return months, date_col


def filter_events_by_month(events, selected_month, date_col):
    """Filter events to a specific month."""
    if selected_month is None or date_col is None:
        return events
    
    dates = pd.to_datetime(events[date_col], errors="coerce")
    month_starts = dates.dt.to_period("M").dt.start_time
    
    return events[month_starts == selected_month].copy()


def compute_cpr_recommendations(edges_df, builders_df):
    """
    Compute Cost Per Referral (CPR) for each builder and generate recommendations.
    CPR = MediaCost / Referrals_out (cost to generate one referral)
    """
    if builders_df.empty or edges_df.empty:
        return pd.DataFrame()
    
    # Calculate referrals sent by each builder
    referrals_out = edges_df.groupby("Origin_builder")["Referrals"].sum().reset_index()
    referrals_out.columns = ["BuilderRegionKey", "Total_Referrals_Sent"]
    
    # Merge with builder data
    recs = builders_df.merge(referrals_out, on="BuilderRegionKey", how="left")
    recs["Total_Referrals_Sent"] = recs["Total_Referrals_Sent"].fillna(0)
    
    # Calculate CPR (Cost Per Referral)
    recs["CPR"] = np.where(
        recs["Total_Referrals_Sent"] > 0,
        recs["MediaCost"] / recs["Total_Referrals_Sent"],
        np.nan
    )
    
    # Only include builders who send referrals and have media cost
    recs = recs[(recs["Total_Referrals_Sent"] > 0) & (recs["MediaCost"] > 0)].copy()
    
    # Sort by CPR (lowest = most efficient)
    recs = recs.sort_values("CPR", ascending=True)
    
    return recs


def analyze_gap_and_investment(focus_builder, events_df, builder_master):
    """
    Advanced Logic: Gap Analysis & Network Investment Engine.
    
    1. Calculate Demand (Gap)
    2. Analyze Supply (Transfer Rates, eCPR)
    3. Calculate Investment & Externalities (Spillover)
    """
    
    # --- 1. Calculate the Gap (The Demand) ---
    
    # Extract Target/Deadline info for the Focus Builder
    # Note: Assuming these columns exist in events_df or builder_master. 
    # If not, we default to 0/NaT as per constraints.
    
    # Try to get target info from events (often repeated on rows) or master
    target_val = 0
    deadline_val = pd.NaT
    
    # Check events first (looking for rows where Dest = Focus)
    focus_events = events_df[events_df['Dest_BuilderRegionKey'] == focus_builder]
    
    if not focus_events.empty:
        if 'LeadTarget_from_job' in focus_events.columns:
            # Take the max found, assuming target is constant or we want the latest
            target_val = focus_events['LeadTarget_from_job'].max()
        if 'WIP_JOB_LIVE_END' in focus_events.columns:
            deadline_val = pd.to_datetime(focus_events['WIP_JOB_LIVE_END'].max())
            
    # Handle missing/NaN
    if pd.isna(target_val): target_val = 0
    
    # Calculate Current Leads (Inbound Referrals)
    current_leads = len(focus_events[focus_events['is_referral'] == True])
    
    shortfall = max(0, target_val - current_leads)
    
    # --- 2. Analyze Supply Sources (The Leverage) ---
    
    # Identify sources (S) who send to Focus Builder (B_target)
    # Filter to referrals only
    all_referrals = events_df[events_df['is_referral'] == True].copy()
    
    # Group by Source (MediaPayer) -> Dest
    # We use MediaPayer as the source of the "Media Investment"
    
    # First, calculate Source-level stats (Total Outbound, Media Spend)
    source_stats = all_referrals.groupby('MediaPayer_BuilderRegionKey').agg(
        Total_Referrals_Sent=('LeadId', 'count'),
        Total_Media_Spend=('MediaCost_referral_event', 'sum')
    ).reset_index()
    
    # Calculate Base CPR (Cost per generic referral)
    source_stats['CPR_base'] = np.where(
        source_stats['Total_Referrals_Sent'] > 0,
        source_stats['Total_Media_Spend'] / source_stats['Total_Referrals_Sent'],
        np.inf
    )
    
    # Calculate specific flow to Focus Builder
    flow_to_target = all_referrals[all_referrals['Dest_BuilderRegionKey'] == focus_builder].groupby(
        'MediaPayer_BuilderRegionKey'
    ).size().reset_index(name='Referrals_to_Target')
    
    # Merge
    analysis = flow_to_target.merge(source_stats, on='MediaPayer_BuilderRegionKey', how='left')
    
    # Calculate Transfer Rate (TR)
    analysis['Transfer_Rate'] = analysis['Referrals_to_Target'] / analysis['Total_Referrals_Sent']
    
    # Calculate Effective CPR (eCPR)
    # eCPR = CPR_base / TR
    analysis['eCPR'] = np.where(
        analysis['Transfer_Rate'] > 0,
        analysis['CPR_base'] / analysis['Transfer_Rate'],
        np.inf
    )
    
    # --- 3. Calculate Investment & Externalities ---
    
    results = []
    
    for _, row in analysis.iterrows():
        source = row['MediaPayer_BuilderRegionKey']
        tr = row['Transfer_Rate']
        ecpr = row['eCPR']
        
        # Scenario: Plug the Shortfall
        if shortfall > 0:
            investment_required = shortfall * ecpr
            total_leads_generated = shortfall / tr if tr > 0 else 0
            excess_leads = total_leads_generated - shortfall
        else:
            investment_required = 0
            total_leads_generated = 0
            excess_leads = 0
            
        # Externalities: Where do the excess leads go?
        # We need the distribution of THIS source's referrals to OTHER builders
        source_flows = all_referrals[all_referrals['MediaPayer_BuilderRegionKey'] == source]
        
        # Calculate spillover impact
        # Simplified: Just count unique other builders receiving leads
        other_recipients = source_flows[source_flows['Dest_BuilderRegionKey'] != focus_builder]['Dest_BuilderRegionKey'].unique()
        spillover_desc = f"{len(other_recipients)} other builders"
        
        results.append({
            'Source Builder': source,
            'Transfer Rate': tr,
            'Base CPR': row['CPR_base'],
            'Effective CPR': ecpr,
            'Investment Needed': investment_required,
            'Total Leads Generated': total_leads_generated,
            'Spillover (Excess Leads)': excess_leads,
            'Spillover Targets': spillover_desc
        })
        
    results_df = pd.DataFrame(results)
    
    # Package gap info
    gap_info = {
        'Target': target_val,
        'Actual': current_leads,
        'Shortfall': shortfall,
        'Deadline': deadline_val
    }
    
    return gap_info, results_df


def main():
    events_full = load_data()
    
    if events_full is None:
        st.warning("⚠️ Please upload events data on the Home page first.")
        st.page_link("app.py", label="← Go to Home", icon="🏠")
        return
    
    # Get available months
    available_months, date_col = get_available_months(events_full)
    
    # Sidebar controls
    with st.sidebar:
        st.header("📅 Time Period")
        
        if available_months:
            month_options = ["All Time"] + [m.strftime("%Y-%m") for m in available_months]
            selected_month_str = st.selectbox(
                "Select Month",
                options=month_options,
                index=0,
                help="Filter network analysis to a specific month. Clustering will be recalculated."
            )
            
            if selected_month_str == "All Time":
                selected_month = None
                events = events_full.copy()
                period_key = "all_time"
            else:
                selected_month = pd.Timestamp(selected_month_str + "-01")
                events = filter_events_by_month(events_full, selected_month, date_col)
                period_key = selected_month_str
            
            # Show filtered stats
            st.metric("Events in Period", f"{len(events):,}")
            if selected_month:
                st.caption(f"🔄 Clustering for **{selected_month_str}** only")
        else:
            events = events_full.copy()
            selected_month = None
            period_key = "all_time"
            st.info("No date column found")
        
        st.divider()
        
        # --- NEW: Optimization Mode Selector ---
        st.header("🚀 Optimization Mode")
        opt_mode = st.radio(
            "Recommendation Engine",
            ["Standard (Lowest CPR)", "Advanced (Shortfall Targeting)"],
            help="Standard: Minimizes cost per referral.\nAdvanced: Minimizes cost to fulfill specific builder lead shortfalls."
        )
        
        st.divider()
        st.header("🎛️ Clustering Parameters")
        
        resolution = st.slider("Resolution", min_value=0.5, max_value=2.5, value=1.5, step=0.1)
        target_clusters = st.slider("Max Clusters", min_value=3, max_value=25, value=15, step=1)
        
        st.divider()
        st.subheader("🎨 Graph Settings")
        show_labels = st.checkbox("Show Node Labels", value=False)
        edge_style = st.selectbox("Edge Style", ["Curved Arrows", "Straight Lines", "Curved Lines"])
        node_size_factor = st.slider("Node Size", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
    
    # Check minimum events
    if len(events) < 10:
        st.error(f"⚠️ Only {len(events)} events for selected period. Need at least 10 events.")
        return
    
    # Run clustering (NOT cached - recalculates each time)
    with st.spinner(f"🔄 Running network clustering for {period_key}..."):
        results = run_clustering_for_period(events, resolution, target_clusters, period_key)
    
    edges_clean = results.get('edges_clean', pd.DataFrame())
    builder_master = results.get('builder_master', pd.DataFrame())
    cluster_summary = results.get('cluster_summary', pd.DataFrame())
    G = results.get('graph', nx.Graph())
    
    if builder_master.empty:
        st.warning("No referral patterns found for the selected period.")
        return
    
    # Period banner
    if selected_month:
        st.success(f"📅 **Analysis Period: {selected_month.strftime('%B %Y')}** — Clustering recalculated for this month")
    else:
        st.info("📅 **Analysis Period: All Time**")
    
    # Build P&L for the filtered period
    pnl_recipient = build_builder_pnl(events, lens="recipient", date_basis="lead_date", freq="ALL")
    pnl_recipient = pnl_recipient.drop(columns=["period_start"], errors="ignore")
    
    builder_master = builder_master.merge(pnl_recipient, on="BuilderRegionKey", how="left")
    for col in ["Revenue", "MediaCost", "Profit", "ROAS"]:
        if col in builder_master.columns:
            builder_master[col] = builder_master[col].fillna(0)
    
    builder_master = compute_network_metrics(G, builder_master)
    
    # ==========================================
    # 💡 RECOMMENDATION ENGINE
    # ==========================================
    st.header("💡 Investment Recommendations")
    
    cpr_recs = pd.DataFrame()
    
    if opt_mode == "Standard (Lowest CPR)":
        st.markdown("**Goal:** Minimize `Media Cost / Total Referrals`.")
        
        cpr_recs = compute_cpr_recommendations(edges_clean, builder_master)
        
        if not cpr_recs.empty:
            # Top 5 recommendations (Standard)
            top_5 = cpr_recs.head(5)
            rec_cols = st.columns(5)
            for i, (_, row) in enumerate(top_5.iterrows()):
                with rec_cols[i]:
                    st.metric(
                        label=f"#{i+1} {row['BuilderRegionKey'][:20]}...",
                        value=f"${row['CPR']:,.0f}",
                        delta=f"{int(row['Total_Referrals_Sent']):,} refs",
                        delta_color="off"
                    )
            
            # Detailed Table for Standard
            with st.expander("📊 Full CPR Ranking (Standard)", expanded=False):
                display_recs = cpr_recs[["BuilderRegionKey", "ClusterId", "Total_Referrals_Sent", "MediaCost", "CPR", "ROAS", "Profit"]].copy()
                st.dataframe(
                    display_recs.style.format({
                        "Total_Referrals_Sent": "{:,.0f}", "MediaCost": "${:,.0f}", 
                        "CPR": "${:,.2f}", "ROAS": "{:.2f}", "Profit": "${:,.0f}"
                    }).background_gradient(subset=["CPR"], cmap="RdYlGn_r"),
                    use_container_width=True
                )
        
    else:
        # --- ADVANCED MODE (Shortfall Targeting) ---
        st.markdown("**Goal:** Minimize `Media Cost / Needed Referrals`. Prioritizes builders sending to those with **Shortfalls** & **Tight Deadlines**.")
        
        # 1. Define Shortfalls (Simulation inputs)
        with st.expander("⚙️ Configure Demand / Targets", expanded=False):
            st.info("In a production environment, this data would load from `LeadTarget_from_job` and `WIP_JOB_LIVE_END`.")
            
            # Use module to calc shortfalls (with mocking)
            shortfall_data = calculate_shortfalls(events)
            
            # Interactive simulation
            builders_list = sorted(shortfall_data['BuilderRegionKey'].unique())
            target_builder = st.selectbox("Simulate Critical Need For:", ["(None)"] + builders_list)
            
            if target_builder != "(None)":
                mask = shortfall_data['BuilderRegionKey'] == target_builder
                shortfall_data.loc[mask, 'Shortfall'] = 50
                shortfall_data.loc[mask, 'Urgency_Weight'] = 2.0
                shortfall_data.loc[mask, 'Weighted_Demand'] = 100
                st.success(f"Simulating 50 lead shortfall for {target_builder}!")

            st.dataframe(
                shortfall_data[shortfall_data['Shortfall']>0].sort_values('Weighted_Demand', ascending=False).head(10)
                .style.format({'Urgency_Weight': '{:.1f}x'}),
                height=150,
                use_container_width=True
            )

        # 2. Run Advanced Optimization
        recs = compute_effective_network_cpr(events, shortfall_data)
        
        # Map back to cpr_recs format for compatibility with search features below
        if not recs.empty:
            # We rename columns to match what the UI expects later (CPR -> Raw CPR mostly)
            # But the 'Effective_CPR' is the real metric here.
            
            best_rec = recs.iloc[0]
            
            # Metric Cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Top Investment", best_rec['BuilderRegionKey'])
            c2.metric("Effective CPR", fmt_currency(best_rec['Effective_CPR']), help="Cost per USEFUL referral")
            c3.metric("Raw CPR", fmt_currency(best_rec['Raw_CPR']), help="Cost per GENERIC referral")
            c4.metric("Efficiency Gain", f"{best_rec['Raw_CPR']/best_rec['Effective_CPR']:.0%}" if best_rec['Effective_CPR'] > 0 else "N/A", delta_color="inverse")
            
            st.subheader(f"Why invest in {best_rec['BuilderRegionKey']}?")
            st.markdown(f"""
            - **Utility:** {best_rec['Useful_Share']:.1%} of their referrals go to builders with active shortfalls.
            - **Leverage:** They help **{best_rec['Beneficiaries_Count']}** distinct builders who are behind target.
            - **Primary Beneficiary:** {best_rec['Top_Beneficiary']}
            """)
            
            with st.expander("📊 Full Effective CPR Ranking", expanded=True):
                st.dataframe(
                    recs[['BuilderRegionKey', 'Raw_CPR', 'Effective_CPR', 'Useful_Share', 'Beneficiaries_Count', 'Top_Beneficiary']]
                    .style.format({
                        'Raw_CPR': '${:,.0f}',
                        'Effective_CPR': '${:,.0f}',
                        'Useful_Share': '{:.1%}'
                    }).background_gradient(subset=['Effective_CPR'], cmap='RdYlGn_r'),
                    use_container_width=True
                )
            
            # Prepare cpr_recs for downstream compatibility
            cpr_recs = recs.rename(columns={'Raw_CPR': 'CPR', 'Effective_CPR': 'Effective_CPR'})
            # Merge cluster IDs back in
            cpr_recs = cpr_recs.merge(builder_master[['BuilderRegionKey', 'ClusterId']], on='BuilderRegionKey', how='left')

    
    st.divider()
    
    # ==========================================
    # GLOBAL BUILDER SEARCH
    # ==========================================
    st.subheader("🔍 Find a Builder")
    
    all_builders = sorted(builder_master["BuilderRegionKey"].dropna().unique().tolist())
    builder_to_cluster = dict(zip(builder_master["BuilderRegionKey"], builder_master["ClusterId"]))
    
    builder_search_options = ["(Select a builder to find their cluster)"] + [
        f"{b}  →  Cluster {int(builder_to_cluster.get(b, 0))}" for b in all_builders
    ]
    
    selected_search = st.selectbox("Search all builders", options=builder_search_options, key="global_builder_search")
    
    search_builder = None
    auto_cluster = None
    if selected_search != "(Select a builder to find their cluster)":
        search_builder = selected_search.split("  →  ")[0]
        auto_cluster = builder_to_cluster.get(search_builder)
        
        # Show this builder's CPR if available
        if not cpr_recs.empty and "BuilderRegionKey" in cpr_recs.columns:
            builder_cpr = cpr_recs[cpr_recs["BuilderRegionKey"] == search_builder]
            if not builder_cpr.empty:
                val = builder_cpr.iloc[0]["CPR"]
                rank = cpr_recs.index.get_loc(builder_cpr.index[0]) + 1
                
                if opt_mode == "Advanced (Shortfall Targeting)" and "Effective_CPR" in builder_cpr.columns:
                     eff_val = builder_cpr.iloc[0]["Effective_CPR"]
                     st.info(f"✅ **{search_builder}** → Cluster {int(auto_cluster)} | Effective CPR: **${eff_val:,.2f}** | Raw CPR: ${val:,.2f}")
                else:
                    st.info(f"✅ **{search_builder}** → Cluster {int(auto_cluster)} | CPR: **${val:,.2f}** (Rank #{rank})")
            else:
                st.info(f"✅ **{search_builder}** → Cluster {int(auto_cluster)} | No CPR data (no referrals sent)")
        else:
             st.info(f"✅ **{search_builder}** → Cluster {int(auto_cluster)}")
    
    st.divider()
    
    # ==========================================
    # CLUSTER SELECTOR
    # ==========================================
    cluster_options = []
    cluster_id_map = {}
    for _, row in cluster_summary.sort_values("ClusterId").iterrows():
        cid = int(row["ClusterId"])
        n_b = int(row["N_builders"])
        t_ref = int(row["Total_referrals_in"] + row["Total_referrals_out"])
        label = f"Cluster {cid} — {n_b} builders, ~{t_ref:,} referrals"
        cluster_options.append(label)
        cluster_id_map[label] = cid
    
    default_idx = 0
    if auto_cluster is not None:
        for i, label in enumerate(cluster_options):
            if cluster_id_map[label] == auto_cluster:
                default_idx = i
                break
    
    col_cluster, col_builder = st.columns([2, 2])
    
    with col_cluster:
        selected_label = st.selectbox("Select Ecosystem", options=cluster_options, index=default_idx)
    
    selected_cluster = cluster_id_map[selected_label]
    
    cluster_builders = builder_master[builder_master["ClusterId"] == selected_cluster].copy()
    cluster_edges = edges_clean[
        (edges_clean["Cluster_origin"] == selected_cluster) &
        (edges_clean["Cluster_dest"] == selected_cluster)
    ].copy()
    
    with col_builder:
        builder_list = sorted(cluster_builders["BuilderRegionKey"].dropna().unique().tolist())
        builder_options = ["(None - show all)"] + builder_list
        
        default_builder_idx = 0
        if search_builder and search_builder in builder_list:
            default_builder_idx = builder_list.index(search_builder) + 1
        
        selected_builder = st.selectbox("🎯 Focus on Builder", options=builder_options, index=default_builder_idx)
        focus_builder = None if selected_builder == "(None - show all)" else selected_builder
    
    # Overview metrics
    period_label = selected_month.strftime('%B %Y') if selected_month else "All Time"
    st.header(f"🌐 Cluster {selected_cluster} Overview ({period_label})")
    
    total_profit = cluster_builders["Profit"].sum()
    total_media = cluster_builders["MediaCost"].sum()
    total_rev = cluster_builders["Revenue"].sum()
    roas = total_rev / total_media if total_media > 0 else np.nan
    n_builders = len(cluster_builders)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Builders", n_builders)
    col2.metric("Revenue", fmt_currency(total_rev))
    col3.metric("Media Cost", fmt_currency(total_media))
    col4.metric("Gross Profit", fmt_currency(total_profit))
    col5.metric("ROAS", fmt_roas(roas))
    
    # --- NEW: Gap Analysis Container ---
    if focus_builder:
        st.markdown("---")
        st.subheader(f"⚡ Gap Analysis: {focus_builder}")
        
        # Run advanced analysis
        gap, pathways = analyze_gap_and_investment(focus_builder, events, builder_master)
        
        # Top Metrics
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Target Leads", int(gap['Target']))
        g2.metric("Current Leads", int(gap['Actual']))
        g3.metric("Shortfall", int(gap['Shortfall']), delta="Fully Met" if gap['Shortfall'] <= 0 else f"-{int(gap['Shortfall'])}", delta_color="normal" if gap['Shortfall'] <= 0 else "inverse")
        
        days_rem = (gap['Deadline'] - pd.Timestamp.now()).days if pd.notnull(gap['Deadline']) else None
        deadline_str = f"{days_rem} days" if days_rem is not None else "No Deadline"
        g4.metric("Time Remaining", deadline_str)
        
        if gap['Shortfall'] > 0:
            st.markdown("#### 🎯 Best Pathways to Close Gap")
            if not pathways.empty:
                # Sort by Investment Needed (efficiency)
                pathways = pathways.sort_values("Investment Needed")
                
                # Format for display
                display_path = pathways.copy()
                
                st.dataframe(
                    display_path[['Source Builder', 'Transfer Rate', 'Base CPR', 'Effective CPR', 'Investment Needed', 'Spillover (Excess Leads)', 'Spillover Targets']]
                    .style.format({
                        'Transfer Rate': '{:.1%}',
                        'Base CPR': '${:,.2f}',
                        'Effective CPR': '${:,.2f}',
                        'Investment Needed': '${:,.0f}',
                        'Spillover (Excess Leads)': '{:,.1f}'
                    })
                    .background_gradient(subset=['Investment Needed'], cmap='RdYlGn_r'),
                    use_container_width=True
                )
                
                # Tooltip logic via expander
                best_source = pathways.iloc[0]
                with st.expander(f"💡 Recommendation: Invest in {best_source['Source Builder']}", expanded=True):
                    st.info(f"""
                    To get **{int(gap['Shortfall'])} leads** for **{focus_builder}**:
                    1.  Invest **${best_source['Investment Needed']:,.0f}** in **{best_source['Source Builder']}**.
                    2.  This generates **{int(best_source['Total Leads Generated'])} total leads** at {best_source['Source Builder']}.
                    3.  **{best_source['Transfer Rate']:.1%}** flow to {focus_builder} (closing the gap).
                    4.  **{int(best_source['Spillover (Excess Leads)'])} excess leads** spill over to **{best_source['Spillover Targets']}**.
                    """)
            else:
                st.warning("No existing referral pathways found. Consider direct media.")
        else:
            st.success("🎉 Target fulfilled! No gap investment required.")
            
        st.markdown("---")

    
    # Cluster-specific CPR recommendations
    if not cpr_recs.empty:
        cluster_cpr = cpr_recs[cpr_recs["ClusterId"] == selected_cluster].head(3)
        if not cluster_cpr.empty:
            st.markdown(f"**🎯 Top Efficiency in Cluster {selected_cluster}:**")
            
            if opt_mode == "Advanced (Shortfall Targeting)" and "Effective_CPR" in cluster_cpr.columns:
                 cpr_text = " | ".join([f"**{r['BuilderRegionKey']}**: ${r['Effective_CPR']:,.0f} (Eff.)" for _, r in cluster_cpr.iterrows()])
            else:
                 cpr_text = " | ".join([f"**{r['BuilderRegionKey']}**: ${r['CPR']:,.0f}" for _, r in cluster_cpr.iterrows()])
            st.markdown(cpr_text)
    
    if focus_builder:
        st.info(f"🎯 Focused on: **{focus_builder}**")
    
    # Network visualization
    st.subheader("🕸️ Network Graph")
    
    with st.expander("📖 Graph Legend", expanded=False):
        leg1, leg2, leg3, leg4, leg5 = st.columns(5)
        leg1.markdown("🟠 **Target** - Selected")
        leg2.markdown("🟢 **Inbound** - Sends TO")
        leg3.markdown("🔵 **Outbound** - Receives FROM")
        leg4.markdown("🟣 **Two-way** - Both")
        leg5.markdown("⚪ **Other** - Not connected")
    
    render_network_graph(cluster_builders, cluster_edges, G, focus_builder, show_labels, edge_style, node_size_factor)
    
    if focus_builder:
        render_focus_analysis(focus_builder, cluster_builders, cluster_edges, cpr_recs)
    
    # Builder table
    st.subheader("📊 Builder Details")
    
    display_cols = ["BuilderRegionKey", "Role", "Referrals_in", "Referrals_out", "Revenue", "MediaCost", "Profit", "ROAS"]
    display_cols = [c for c in display_cols if c in cluster_builders.columns]
    
    st.dataframe(
        cluster_builders[display_cols].sort_values("Profit", ascending=False)
        .style.format({
            "Referrals_in": "{:,.0f}",
            "Referrals_out": "{:,.0f}",
            "Revenue": "${:,.0f}",
            "MediaCost": "${:,.0f}",
            "Profit": "${:,.0f}",
            "ROAS": "{:.2f}"
        }),
        hide_index=True,
        use_container_width=True,
        height=400
    )


def render_network_graph(builders, edges, G, focus_builder, show_labels, edge_style, node_size_factor):
    if len(G.nodes) == 0:
        st.info("No network edges to display.")
        return
    
    pos = nx.spring_layout(G, weight="weight", seed=42, k=1.5, iterations=50)
    
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    xo, yo = np.mean(xs), np.mean(ys)
    rmax = max(max(abs(np.array(xs) - xo)), max(abs(np.array(ys) - yo)), 1e-9)
    scale = 0.9 / rmax
    
    for n, (x, y) in pos.items():
        pos[n] = ((x - xo) * scale, (y - yo) * scale)
    
    if focus_builder:
        inbound_nodes = set(edges.loc[edges["Dest_builder"] == focus_builder, "Origin_builder"])
        outbound_nodes = set(edges.loc[edges["Origin_builder"] == focus_builder, "Dest_builder"])
    else:
        inbound_nodes, outbound_nodes = set(), set()
    
    two_way = inbound_nodes & outbound_nodes
    inbound_only = inbound_nodes - outbound_nodes
    outbound_only = outbound_nodes - inbound_nodes
    
    traces = []
    max_weight = edges["Referrals"].max() if not edges.empty else 1
    min_weight = edges["Referrals"].min() if not edges.empty else 1
    
    for _, row in edges.iterrows():
        u, v = row["Origin_builder"], row["Dest_builder"]
        if u not in pos or v not in pos:
            continue
        
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        weight = row["Referrals"]
        norm_weight = 1 + 5 * (weight - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 3
        
        is_focus_edge = focus_builder and (u == focus_builder or v == focus_builder)
        edge_color = "#1E40AF" if is_focus_edge else "#94A3B8"
        edge_width = norm_weight * 1.5 if is_focus_edge else norm_weight * 0.8
        edge_opacity = 1.0 if is_focus_edge else (0.4 if focus_builder else 0.6)
        
        if edge_style in ["Curved Arrows", "Curved Lines"]:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            dx, dy = x1 - x0, y1 - y0
            length = np.sqrt(dx**2 + dy**2) + 1e-9
            cx = mx - dy / length * 0.15 * length
            cy = my + dx / length * 0.15 * length
            
            t_vals = np.linspace(0, 1, 20)
            curve_x = (1-t_vals)**2 * x0 + 2*(1-t_vals)*t_vals * cx + t_vals**2 * x1
            curve_y = (1-t_vals)**2 * y0 + 2*(1-t_vals)*t_vals * cy + t_vals**2 * y1
            
            traces.append(go.Scatter(
                x=curve_x.tolist(), y=curve_y.tolist(), mode="lines",
                line=dict(width=edge_width, color=edge_color), opacity=edge_opacity,
                hoverinfo="text", hovertext=f"{u} → {v}<br>Referrals: {int(weight):,}", showlegend=False
            ))
            
            if edge_style == "Curved Arrows":
                t_arrow, t_before = 0.8, 0.75
                ax = (1-t_arrow)**2 * x0 + 2*(1-t_arrow)*t_arrow * cx + t_arrow**2 * x1
                ay = (1-t_arrow)**2 * y0 + 2*(1-t_arrow)*t_arrow * cy + t_arrow**2 * y1
                bx = (1-t_before)**2 * x0 + 2*(1-t_before)*t_before * cx + t_before**2 * x1
                by = (1-t_before)**2 * y0 + 2*(1-t_before)*t_before * cy + t_before**2 * y1
                
                traces.append(go.Scatter(
                    x=[ax], y=[ay], mode="markers",
                    marker=dict(symbol="triangle-up", size=8+norm_weight, color=edge_color,
                               angle=np.degrees(np.arctan2(ay-by, ax-bx))-90, opacity=edge_opacity),
                    hoverinfo="skip", showlegend=False
                ))
        else:
            traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None], mode="lines",
                line=dict(width=edge_width, color=edge_color), opacity=edge_opacity,
                hoverinfo="text", hovertext=f"{u} → {v}<br>Referrals: {int(weight):,}", showlegend=False
            ))
    
    bidx = builders.set_index("BuilderRegionKey")
    categories = {
        "Target": {"x": [], "y": [], "txt": [], "size": [], "color": "#F97316", "symbol": "star"},
        "Inbound": {"x": [], "y": [], "txt": [], "size": [], "color": "#22C55E", "symbol": "circle"},
        "Outbound": {"x": [], "y": [], "txt": [], "size": [], "color": "#3B82F6", "symbol": "circle"},
        "Two-way": {"x": [], "y": [], "txt": [], "size": [], "color": "#A855F7", "symbol": "diamond"},
        "Other": {"x": [], "y": [], "txt": [], "size": [], "color": "#CBD5E1", "symbol": "circle"}
    }
    
    for node in G.nodes():
        if node not in pos:
            continue
        x, y = pos[node]
        
        if node in bidx.index:
            b = bidx.loc[node]
            txt = f"<b>{node}</b><br>Profit: ${b.get('Profit',0):,.0f}<br>In: {int(b.get('Referrals_in',0)):,} | Out: {int(b.get('Referrals_out',0)):,}"
            size = min(max(20 + (b.get("Referrals_in",0) + b.get("Referrals_out",0)) * 0.08, 15), 60) * node_size_factor
        else:
            txt, size = node, 15 * node_size_factor
        
        cat = "Target" if node == focus_builder else ("Two-way" if node in two_way else ("Inbound" if node in inbound_only else ("Outbound" if node in outbound_only else "Other")))
        if cat == "Target": size *= 1.5
        
        categories[cat]["x"].append(x)
        categories[cat]["y"].append(y)
        categories[cat]["txt"].append(txt)
        categories[cat]["size"].append(size)
    
    for name in ["Other", "Two-way", "Outbound", "Inbound", "Target"]:
        cat = categories[name]
        if not cat["x"]: continue
        traces.append(go.Scatter(
            x=cat["x"], y=cat["y"], mode="markers+text" if show_labels else "markers",
            text=[t.split("<br>")[0].replace("<b>","").replace("</b>","") for t in cat["txt"]],
            textposition="top center", textfont=dict(size=10, color="#1F2937"),
            hovertext=cat["txt"], hoverinfo="text", name=name,
            marker=dict(size=cat["size"], color=cat["color"], symbol=cat["symbol"], opacity=0.9, line=dict(width=2, color="#1F2937"))
        ))
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        height=650, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        hovermode="closest", dragmode="pan", paper_bgcolor="#F8FAFC", plot_bgcolor="#F8FAFC"
    )
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})


def render_focus_analysis(focus_builder, builders, edges, cpr_recs):
    st.subheader(f"🔍 {focus_builder} - Flow Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**⬅️ Inbound (who sends TO this builder)**")
        inbound = edges[edges["Dest_builder"] == focus_builder].groupby("Origin_builder", as_index=False)["Referrals"].sum().sort_values("Referrals", ascending=False)
        if not inbound.empty:
            inbound = inbound.merge(cpr_recs, left_on="Origin_builder", right_on="BuilderRegionKey", how="left")
            
            # Defensive check for Effective_CPR column (only in Advanced mode)
            display_cols = ["Origin_builder", "Referrals", "CPR"]
            if "Effective_CPR" in inbound.columns:
                display_cols.append("Effective_CPR")
                
            st.dataframe(
                inbound[display_cols].style.format({"Referrals": "{:,.0f}", "CPR": "${:,.2f}", "Effective_CPR": "${:,.0f}"}, na_rep="-"), 
                hide_index=True, 
                use_container_width=True
            )
        else:
            st.info("No inbound referrals")
    with col2:
        st.markdown("**➡️ Outbound (where this builder sends)**")
        outbound = edges[edges["Origin_builder"] == focus_builder].groupby("Dest_builder", as_index=False)["Referrals"].sum().sort_values("Referrals", ascending=False)
        if not outbound.empty:
            st.dataframe(outbound.style.format({"Referrals": "{:,.0f}"}), hide_index=True, use_container_width=True)
        else:
            st.info("No outbound referrals")


if __name__ == "__main__":
    main()