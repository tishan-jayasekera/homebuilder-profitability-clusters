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
from src.network_optimization import (
    calculate_shortfalls, 
    get_targeted_fulfillment_strategies, 
    generate_network_fulfillment_plan,
    compute_effective_network_cpr
)

st.set_page_config(page_title="Referral Networks", page_icon="🔗", layout="wide")

st.title("🔗 Referral Network Explorer")
st.markdown("Discover referral ecosystems and media efficiency pathways.")


def run_clustering_for_period(events_df, resolution, target_clusters, period_key):
    return run_referral_clustering(events_df, resolution=resolution, target_max_clusters=target_clusters)


def load_data():
    if 'events_file' not in st.session_state:
        return None
    events = load_events(st.session_state['events_file'])
    if events is None:
        return None
    return normalize_events(events)


def get_available_months(events):
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
    if selected_month is None or date_col is None:
        return events
    dates = pd.to_datetime(events[date_col], errors="coerce")
    month_starts = dates.dt.to_period("M").dt.start_time
    return events[month_starts == selected_month].copy()


def compute_cpr_recommendations(edges_df, builders_df):
    if builders_df.empty or edges_df.empty:
        return pd.DataFrame()
    
    referrals_out = edges_df.groupby("Origin_builder")["Referrals"].sum().reset_index()
    referrals_out.columns = ["BuilderRegionKey", "Total_Referrals_Sent"]
    
    recs = builders_df.merge(referrals_out, on="BuilderRegionKey", how="left")
    recs["Total_Referrals_Sent"] = recs["Total_Referrals_Sent"].fillna(0)
    
    recs["CPR"] = np.where(
        recs["Total_Referrals_Sent"] > 0,
        recs["MediaCost"] / recs["Total_Referrals_Sent"],
        np.nan
    )
    
    recs = recs[(recs["Total_Referrals_Sent"] > 0) & (recs["MediaCost"] > 0)].copy()
    recs = recs.sort_values("CPR", ascending=True)
    return recs

def render_fulfillment_sankey(plan_df, strategies_df):
    """
    Visualize the flow from Recommended Payer -> Strategy -> Target Builder
    """
    if plan_df.empty or strategies_df.empty:
        return None

    # Filter to only actionable items
    actionable_plan = plan_df[plan_df['Status'] == 'Fulfillable'].copy()
    if actionable_plan.empty:
        return None

    # We want flow: Payer -> Target Builder
    # Get the Payer for each Target from the plan (which picked the best strategy)
    # The plan_df has 'Recommended_Action' which is "Invest in {Payer}"
    
    # Extract Payer from Recommended Action string or join back with strategies
    # Easier to join back. For each target in plan, find the payer we selected.
    # The plan used the best strategy. Let's re-find that best strategy to be sure.
    
    sankey_data = []
    
    for _, row in actionable_plan.iterrows():
        target = row['Target_Builder']
        shortfall = row['Shortfall']
        
        # Find the strategy used (lowest cost per target lead)
        opts = strategies_df[strategies_df['Dest_Builder'] == target].sort_values('Cost_Per_Target_Lead')
        if not opts.empty:
            payer = opts.iloc[0]['MediaPayer_BuilderRegionKey']
            # Flow: Payer -> Target, value = Shortfall (leads needed)
            sankey_data.append({'Source': payer, 'Target': target, 'Value': shortfall})

    sankey_df = pd.DataFrame(sankey_data)
    
    # Aggregate flows (multiple targets might use same payer)
    sankey_df = sankey_df.groupby(['Source', 'Target']).sum().reset_index()

    # Create nodes
    all_nodes = list(pd.concat([sankey_df['Source'], sankey_df['Target']]).unique())
    node_map = {node: i for i, node in enumerate(all_nodes)}
    
    # Create links
    links = {
        'source': [node_map[src] for src in sankey_df['Source']],
        'target': [node_map[tgt] for tgt in sankey_df['Target']],
        'value': sankey_df['Value'],
        'color': 'rgba(31, 119, 180, 0.3)' # Light blue links
    }

    # Node colors - distinguish Payer (Source) vs Target
    node_colors = []
    for node in all_nodes:
        if node in sankey_df['Source'].values:
            node_colors.append('#1F77B4') # Blue for Payers
        else:
            node_colors.append('#FF7F0E') # Orange for Targets

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=node_colors
        ),
        link=links
    )])
    
    fig.update_layout(title_text="Fulfillment Flow: Payer → Target Builder (Leads)", font_size=10, height=400)
    return fig


def main():
    events_full = load_data()
    
    if events_full is None:
        st.warning("⚠️ Please upload events data on the Home page first.")
        st.page_link("app.py", label="← Go to Home", icon="🏠")
        return
    
    # Get available months
    available_months, date_col = get_available_months(events_full)
    
    # Sidebar
    with st.sidebar:
        st.header("📅 Time Period")
        if available_months:
            month_options = ["All Time"] + [m.strftime("%Y-%m") for m in available_months]
            selected_month_str = st.selectbox("Select Month", options=month_options, index=0)
            
            if selected_month_str == "All Time":
                selected_month = None
                events = events_full.copy()
                period_key = "all_time"
            else:
                selected_month = pd.Timestamp(selected_month_str + "-01")
                events = filter_events_by_month(events_full, selected_month, date_col)
                period_key = selected_month_str
            
            st.metric("Events in Period", f"{len(events):,}")
        else:
            events = events_full.copy()
            selected_month = None
            period_key = "all_time"
        
        st.divider()
        st.header("🚀 Optimization Mode")
        opt_mode = st.radio(
            "Recommendation Engine",
            ["Standard (Lowest CPR)", "Advanced (Shortfall Targeting)"],
            help="Standard: Minimizes generic cost per referral.\nAdvanced: Identifies lead shortfalls and targets specific payers to fill them."
        )
        
        st.divider()
        st.header("🎛️ Clustering")
        resolution = st.slider("Resolution", 0.5, 2.5, 1.5, 0.1)
        target_clusters = st.slider("Max Clusters", 3, 25, 15, 1)
        
        st.divider()
        st.subheader("🎨 Graph")
        show_labels = st.checkbox("Show Labels", False)
        edge_style = st.selectbox("Edge Style", ["Curved Arrows", "Straight Lines", "Curved Lines"])
        node_size_factor = st.slider("Node Size", 0.5, 2.0, 1.0, 0.1)
    
    # Run clustering
    if len(events) < 10:
        st.error(f"⚠️ Only {len(events)} events. Need at least 10.")
        return
        
    with st.spinner(f"🔄 Running clustering..."):
        results = run_clustering_for_period(events, resolution, target_clusters, period_key)
    
    edges_clean = results.get('edges_clean', pd.DataFrame())
    builder_master = results.get('builder_master', pd.DataFrame())
    cluster_summary = results.get('cluster_summary', pd.DataFrame())
    G = results.get('graph', nx.Graph())
    
    # P&L Data
    pnl_recipient = build_builder_pnl(events, lens="recipient", date_basis="lead_date", freq="ALL")
    pnl_recipient = pnl_recipient.drop(columns=["period_start"], errors="ignore")
    builder_master = builder_master.merge(pnl_recipient, on="BuilderRegionKey", how="left")
    builder_master = compute_network_metrics(G, builder_master)

    # ==========================================
    # 💡 INVESTMENT RECOMMENDATIONS
    # ==========================================
    st.header("💡 Investment Recommendations")
    
    cpr_recs = pd.DataFrame()
    
    if opt_mode == "Standard (Lowest CPR)":
        st.markdown("### 📊 Generic Efficiency View")
        st.caption("Ranking builders by lowest cost per referral generated, regardless of destination.")
        
        cpr_recs = compute_cpr_recommendations(edges_clean, builder_master)
        
        if not cpr_recs.empty:
            cols = st.columns(4)
            for i, (_, row) in enumerate(cpr_recs.head(4).iterrows()):
                cols[i].metric(
                    f"#{i+1} {row['BuilderRegionKey']}",
                    f"${row['CPR']:,.0f} CPR",
                    f"{int(row['Total_Referrals_Sent'])} refs"
                )
            
            with st.expander("View Full Ranking", expanded=True):
                st.dataframe(
                    cpr_recs[['BuilderRegionKey', 'MediaCost', 'Total_Referrals_Sent', 'CPR', 'ROAS']]
                    .style.format({'MediaCost': '${:,.0f}', 'CPR': '${:,.2f}', 'ROAS': '{:.2f}'})
                    .background_gradient(subset=['CPR'], cmap='RdYlGn_r'),
                    use_container_width=True
                )
    
    else:
        # --- ADVANCED MODE: SHORTFALL TARGETING ---
        st.markdown("### 🎯 Operational Strategy View")
        st.caption("Identifying builders with lead shortfalls and mapping the most efficient media partners to fulfill them.")
        
        # 1. Calc Shortfalls
        shortfall_data = calculate_shortfalls(events)
        
        # Simulation UI
        with st.expander("⚙️ Simulation Controls (Mock Data)", expanded=False):
            st.info("Simulate a critical lead shortage to test the strategy engine.")
            builders = sorted(shortfall_data['BuilderRegionKey'].unique())
            sim_target = st.selectbox("Simulate Critical Need For:", ["(None)"] + builders)
            
            if sim_target != "(None)":
                mask = shortfall_data['BuilderRegionKey'] == sim_target
                shortfall_data.loc[mask, 'Shortfall'] = 50
                shortfall_data.loc[mask, 'Urgency_Weight'] = 2.0
                shortfall_data.loc[mask, 'Weighted_Demand'] = 100
        
        # 2. Network Diagnosis
        st.subheader("1. Network Diagnosis")
        
        critical_shortfalls = shortfall_data[shortfall_data['Shortfall'] > 0].copy()
        
        if critical_shortfalls.empty:
            st.success("✅ No critical lead shortfalls detected.")
        else:
            total_gap = critical_shortfalls['Shortfall'].sum()
            critical_count = len(critical_shortfalls)
            high_urgency = len(critical_shortfalls[critical_shortfalls['Urgency_Weight'] >= 1.5])
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Lead Gap", f"{int(total_gap)}")
            m2.metric("Builders w/ Shortfalls", f"{critical_count}")
            m3.metric("Critical Urgency (<60 days)", f"{high_urgency}", delta_color="inverse")
            
            # 3. Fulfillment Planning
            st.divider()
            st.subheader("2. Fulfillment Plan & Reconciliation")
            
            strategies = get_targeted_fulfillment_strategies(events, shortfall_data)
            full_plan = generate_network_fulfillment_plan(shortfall_data, strategies)
            
            if not full_plan.empty:
                # Summary of the Plan
                total_budget = full_plan['Est_Budget_Required'].sum()
                fill_count = full_plan[full_plan['Status'] == 'Fulfillable'].shape[0]
                gap_count = full_plan[full_plan['Status'] != 'Fulfillable'].shape[0]
                
                # Sankey Diagram of the Plan
                sankey_fig = render_fulfillment_sankey(full_plan, strategies)
                if sankey_fig:
                    st.plotly_chart(sankey_fig, use_container_width=True)
                
                p1, p2, p3 = st.columns(3)
                p1.metric("Est. Media Requirement", fmt_currency(total_budget))
                p2.metric("Solvable Builders", fill_count)
                p3.metric("Unserviceable Gaps", gap_count, delta_color="inverse")
                
                st.dataframe(
                    full_plan[['Priority', 'Target_Builder', 'Shortfall', 'Status', 'Recommended_Action', 'Est_Budget_Required']]
                    .rename(columns={'Est_Budget_Required': 'Est. Cost'})
                    .style.format({'Est. Cost': '${:,.0f}'})
                    .applymap(lambda v: 'color: red' if v == 'Under-Serviced' else 'color: green', subset=['Status']),
                    use_container_width=True, hide_index=True
                )
            else:
                 st.info("No actionable fulfillment plan could be generated.")
            
            # Compatibility for map
            cpr_recs = compute_cpr_recommendations(edges_clean, builder_master)

    st.divider()
    
    # ==========================================
    # GLOBAL SEARCH & GRAPH
    # ==========================================
    st.subheader("🔍 Network Explorer")
    
    all_builders = sorted(builder_master["BuilderRegionKey"].unique())
    builder_to_cluster = dict(zip(builder_master["BuilderRegionKey"], builder_master["ClusterId"]))
    
    search_options = ["(Select Builder)"] + [f"{b} (Cluster {int(builder_to_cluster.get(b,0))})" for b in all_builders]
    search_sel = st.selectbox("Find Builder", search_options)
    
    focus_builder = None
    search_cluster = None
    
    if search_sel != "(Select Builder)":
        focus_builder = search_sel.split(" (Cluster")[0]
        search_cluster = builder_to_cluster.get(focus_builder)
        
        # Show stats
        if not cpr_recs.empty:
            stats = cpr_recs[cpr_recs['BuilderRegionKey'] == focus_builder]
            if not stats.empty:
                st.info(f"**{focus_builder}**: CPR ${stats.iloc[0]['CPR']:,.0f}")
    
    # Cluster Select
    cluster_labels = {int(r.ClusterId): f"Cluster {int(r.ClusterId)} ({int(r.N_builders)} builders)" for r in cluster_summary.itertuples()}
    
    default_ix = 0
    if search_cluster is not None:
        valid_ids = list(cluster_labels.keys())
        if search_cluster in valid_ids:
            default_ix = valid_ids.index(search_cluster)
            
    sel_cluster_id = st.selectbox("Select Ecosystem", options=list(cluster_labels.keys()), format_func=lambda x: cluster_labels[x], index=default_ix)
    
    # Filter Graph Data
    sub_builders = builder_master[builder_master["ClusterId"] == sel_cluster_id]
    sub_edges = edges_clean[(edges_clean["Cluster_origin"] == sel_cluster_id) & (edges_clean["Cluster_dest"] == sel_cluster_id)]
    
    render_network_graph(sub_builders, sub_edges, G, focus_builder, show_labels, edge_style, node_size_factor)
    
    if focus_builder:
        render_focus_analysis(focus_builder, sub_builders, sub_edges, cpr_recs)


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
            st.dataframe(inbound[["Origin_builder", "Referrals", "CPR"]].style.format({"Referrals": "{:,.0f}", "CPR": "${:,.2f}"}), hide_index=True, use_container_width=True)
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