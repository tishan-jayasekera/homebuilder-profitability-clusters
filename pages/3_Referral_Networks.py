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
    # 💡 RECOMMENDATION ENGINE - TOP OF PAGE
    # ==========================================
    st.header("💡 CPR Recommendations")
    st.markdown("**Lowest Cost Per Referral (CPR)** = Most efficient builders to invest media spend in")
    
    cpr_recs = compute_cpr_recommendations(edges_clean, builder_master)
    
    if not cpr_recs.empty:
        # Top 5 recommendations
        top_5 = cpr_recs.head(5)
        
        rec_cols = st.columns(5)
        for i, (_, row) in enumerate(top_5.iterrows()):
            with rec_cols[i]:
                st.metric(
                    label=f"#{i+1} {row['BuilderRegionKey'][:20]}...",
                    value=f"${row['CPR']:,.0f}",
                    delta=f"{int(row['Total_Referrals_Sent']):,} refs sent",
                    delta_color="off"
                )
        
        # Full recommendation table
        with st.expander("📊 Full CPR Ranking (All Builders)", expanded=False):
            display_recs = cpr_recs[["BuilderRegionKey", "ClusterId", "Total_Referrals_Sent", "MediaCost", "CPR", "ROAS", "Profit"]].copy()
            display_recs.columns = ["Builder", "Cluster", "Referrals Sent", "Media Cost", "CPR", "ROAS", "Profit"]
            
            st.dataframe(
                display_recs.style.format({
                    "Referrals Sent": "{:,.0f}",
                    "Media Cost": "${:,.0f}",
                    "CPR": "${:,.2f}",
                    "ROAS": "{:.2f}",
                    "Profit": "${:,.0f}"
                }).background_gradient(subset=["CPR"], cmap="RdYlGn_r"),
                hide_index=True,
                use_container_width=True,
                height=300
            )
            
            st.download_button(
                "📥 Download CPR Recommendations",
                data=export_to_excel(cpr_recs, "cpr_recommendations.xlsx"),
                file_name=f"cpr_recommendations_{period_key}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Best overall recommendation
        best = cpr_recs.iloc[0]
        st.success(f"""
        🏆 **Top Recommendation:** Increase media spend on **{best['BuilderRegionKey']}** (Cluster {int(best['ClusterId'])})
        - **CPR:** ${best['CPR']:,.2f} per referral
        - **Referrals Sent:** {int(best['Total_Referrals_Sent']):,}
        - **Current Media Cost:** ${best['MediaCost']:,.0f}
        - **ROAS:** {best['ROAS']:.2f}x
        """)
    else:
        st.warning("Not enough data to generate CPR recommendations.")
    
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
        builder_cpr = cpr_recs[cpr_recs["BuilderRegionKey"] == search_builder]
        if not builder_cpr.empty:
            cpr_val = builder_cpr.iloc[0]["CPR"]
            rank = cpr_recs.index.get_loc(builder_cpr.index[0]) + 1
            st.info(f"✅ **{search_builder}** → Cluster {int(auto_cluster)} | CPR: **${cpr_val:,.2f}** (Rank #{rank})")
        else:
            st.info(f"✅ **{search_builder}** → Cluster {int(auto_cluster)} | No CPR data (no referrals sent)")
    
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
    
    # Cluster-specific CPR recommendations
    cluster_cpr = cpr_recs[cpr_recs["ClusterId"] == selected_cluster].head(3)
    if not cluster_cpr.empty:
        st.markdown(f"**🎯 Top CPR in Cluster {selected_cluster}:**")
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
            # Merge with CPR data
            inbound = inbound.merge(cpr_recs[["BuilderRegionKey", "CPR", "MediaCost"]], left_on="Origin_builder", right_on="BuilderRegionKey", how="left")
            
            st.dataframe(
                inbound[["Origin_builder", "Referrals", "MediaCost", "CPR"]]
                .style.format({"Referrals": "{:,.0f}", "MediaCost": "${:,.0f}", "CPR": "${:,.2f}"}),
                hide_index=True, use_container_width=True
            )
            
            # Best lever = lowest CPR among inbound
            valid_cpr = inbound[inbound["CPR"].notna()]
            if not valid_cpr.empty:
                best = valid_cpr.loc[valid_cpr["CPR"].idxmin()]
                st.success(f"💡 **Best lever to increase referrals TO {focus_builder}:** Invest in **{best['Origin_builder']}** @ **${best['CPR']:,.2f} CPR**")
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