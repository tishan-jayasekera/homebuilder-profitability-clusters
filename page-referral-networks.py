"""
Referral Networks Dashboard - Streamlit Page
"""
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_events, export_to_excel
from src.normalization import normalize_events
from src.builder_pnl import build_builder_pnl
from src.referral_clusters import run_referral_clustering, compute_network_metrics
from src.utils import fmt_currency, fmt_percent, fmt_roas, get_status_color

st.set_page_config(page_title="Referral Networks", page_icon="🔗", layout="wide")

st.title("🔗 Referral Network Explorer")
st.markdown("Discover referral ecosystems and media efficiency pathways.")


@st.cache_data(show_spinner="Running network clustering...")
def run_clustering(_events, resolution, target_clusters):
    """Cached clustering computation."""
    return run_referral_clustering(
        _events,
        resolution=resolution,
        target_max_clusters=target_clusters
    )


def load_data():
    """Load and normalize events data."""
    if 'events_file' not in st.session_state:
        return None
    
    events = load_events(st.session_state['events_file'])
    if events is None:
        return None
    
    return normalize_events(events)


def main():
    events = load_data()
    
    if events is None:
        st.warning("⚠️ Please upload events data on the Home page first.")
        st.page_link("app.py", label="← Go to Home", icon="🏠")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Clustering Parameters")
        
        resolution = st.slider(
            "Resolution",
            min_value=0.5, max_value=2.5, value=1.5, step=0.1,
            help="Higher = more clusters, Lower = fewer clusters"
        )
        
        target_clusters = st.slider(
            "Max Clusters",
            min_value=3, max_value=25, value=15, step=1
        )
        
        st.divider()
        
        show_labels = st.checkbox("Show Node Labels", value=False)
        
        st.divider()
        
        # Builder search
        st.subheader("🔍 Find Builder")
        builder_search = st.text_input("Search", placeholder="Type builder name...")
    
    # Run clustering
    results = run_clustering(events, resolution, target_clusters)
    
    edges_clean = results.get('edges_clean', pd.DataFrame())
    builder_master = results.get('builder_master', pd.DataFrame())
    cluster_summary = results.get('cluster_summary', pd.DataFrame())
    G = results.get('graph', nx.Graph())
    
    if builder_master.empty:
        st.warning("No referral patterns found in the data.")
        return
    
    # Build P&L for overlay
    pnl_recipient = build_builder_pnl(events, lens="recipient", date_basis="lead_date", freq="ALL")
    pnl_recipient = pnl_recipient.drop(columns=["period_start"], errors="ignore")
    
    # Merge P&L with builder master
    builder_master = builder_master.merge(pnl_recipient, on="BuilderRegionKey", how="left")
    for col in ["Revenue", "MediaCost", "Profit", "ROAS"]:
        if col in builder_master.columns:
            builder_master[col] = builder_master[col].fillna(0)
    
    # Compute network metrics
    builder_master = compute_network_metrics(G, builder_master)
    
    # Cluster selector
    cluster_options = []
    for _, row in cluster_summary.sort_values("ClusterId").iterrows():
        cid = int(row["ClusterId"])
        n_b = int(row["N_builders"])
        t_ref = int(row["Total_referrals_in"] + row["Total_referrals_out"])
        label = f"Cluster {cid} — {n_b} builders, ~{t_ref:,} referrals"
        cluster_options.append((label, cid))
    
    selected_label = st.selectbox(
        "Select Ecosystem",
        options=[opt[0] for opt in cluster_options]
    )
    selected_cluster = dict(cluster_options)[selected_label]
    
    # Filter to selected cluster
    cluster_builders = builder_master[builder_master["ClusterId"] == selected_cluster].copy()
    cluster_edges = edges_clean[
        (edges_clean["Cluster_origin"] == selected_cluster) &
        (edges_clean["Cluster_dest"] == selected_cluster)
    ].copy()
    
    # Overview metrics
    st.header(f"🌐 Cluster {selected_cluster} Overview")
    
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
    
    # Focus builder
    focus_builder = None
    if builder_search:
        matches = cluster_builders[
            cluster_builders["BuilderRegionKey"].str.lower().str.contains(builder_search.lower(), na=False)
        ]
        if len(matches) > 0:
            focus_builder = matches.iloc[0]["BuilderRegionKey"]
            st.info(f"📍 Focused on: **{focus_builder}**")
    
    # Network visualization
    st.subheader("🕸️ Network Graph")
    
    render_network_graph(cluster_builders, cluster_edges, G, focus_builder, show_labels)
    
    # Flow analysis for focus builder
    if focus_builder:
        render_focus_analysis(focus_builder, cluster_builders, cluster_edges)
    
    # Builder table
    st.subheader("📊 Builder Details")
    
    display_cols = [
        "BuilderRegionKey", "Role", "Referrals_in", "Referrals_out",
        "Revenue", "MediaCost", "Profit", "ROAS", "Referral_Gap"
    ]
    display_cols = [c for c in display_cols if c in cluster_builders.columns]
    
    st.dataframe(
        cluster_builders[display_cols]
        .sort_values("Profit", ascending=False)
        .style.format({
            "Referrals_in": "{:,.0f}",
            "Referrals_out": "{:,.0f}",
            "Revenue": "${:,.0f}",
            "MediaCost": "${:,.0f}",
            "Profit": "${:,.0f}",
            "ROAS": "{:.2f}",
            "Referral_Gap": "{:.1f}"
        }),
        hide_index=True,
        use_container_width=True,
        height=400
    )
    
    # Download
    st.download_button(
        "📥 Download Cluster Data",
        data=export_to_excel(cluster_builders, "cluster_builders.xlsx"),
        file_name=f"cluster_{selected_cluster}_builders.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


def render_network_graph(builders, edges, G, focus_builder, show_labels):
    """Render interactive network graph."""
    if len(G.nodes) == 0:
        st.info("No network edges to display.")
        return
    
    # Layout
    pos = nx.spring_layout(G, weight="weight", seed=42, k=0.8)
    
    # Normalize positions
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    xo, yo = np.mean(xs), np.mean(ys)
    rmax = max(max(abs(np.array(xs) - xo)), max(abs(np.array(ys) - yo)), 1e-9)
    scale = 0.9 / rmax
    
    for n, (x, y) in pos.items():
        pos[n] = ((x - xo) * scale, (y - yo) * scale)
    
    # Determine inbound/outbound to focus
    if focus_builder:
        inbound_nodes = set(edges.loc[edges["Dest_builder"] == focus_builder, "Origin_builder"])
        outbound_nodes = set(edges.loc[edges["Origin_builder"] == focus_builder, "Dest_builder"])
    else:
        inbound_nodes = set()
        outbound_nodes = set()
    
    two_way = inbound_nodes & outbound_nodes
    inbound_only = inbound_nodes - outbound_nodes
    outbound_only = outbound_nodes - inbound_nodes
    
    # Edge traces
    base_x, base_y, hi_x, hi_y = [], [], [], []
    
    for _, row in edges.iterrows():
        u, v = row["Origin_builder"], row["Dest_builder"]
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        if focus_builder and (u == focus_builder or v == focus_builder):
            hi_x += [x0, x1, None]
            hi_y += [y0, y1, None]
        else:
            base_x += [x0, x1, None]
            base_y += [y0, y1, None]
    
    traces = []
    
    if base_x:
        traces.append(go.Scatter(
            x=base_x, y=base_y, mode="lines",
            line=dict(width=1, color="#CBD5E1"),
            hoverinfo="none", showlegend=False
        ))
    
    if hi_x:
        traces.append(go.Scatter(
            x=hi_x, y=hi_y, mode="lines",
            line=dict(width=2.5, color="#1E40AF"),
            name="Focus edges", hoverinfo="none"
        ))
    
    # Node traces by category
    bidx = builders.set_index("BuilderRegionKey")
    
    categories = {
        "Target": {"x": [], "y": [], "txt": [], "size": [], "color": "#F97316"},
        "Inbound": {"x": [], "y": [], "txt": [], "size": [], "color": "#22C55E"},
        "Outbound": {"x": [], "y": [], "txt": [], "size": [], "color": "#3B82F6"},
        "Two-way": {"x": [], "y": [], "txt": [], "size": [], "color": "#A855F7"},
        "Other": {"x": [], "y": [], "txt": [], "size": [], "color": "#9CA3AF"}
    }
    
    for node in G.nodes():
        if node not in pos:
            continue
        
        x, y = pos[node]
        
        if node in bidx.index:
            b = bidx.loc[node]
            txt = f"<b>{node}</b><br>Profit: ${b.get('Profit', 0):,.0f}<br>In: {int(b.get('Referrals_in', 0)):,} | Out: {int(b.get('Referrals_out', 0)):,}"
            size = 15 + (b.get("Referrals_in", 0) + b.get("Referrals_out", 0)) * 0.05
            size = min(max(size, 12), 50)
        else:
            txt = node
            size = 12
        
        if node == focus_builder:
            cat = "Target"
            size *= 1.3
        elif node in two_way:
            cat = "Two-way"
        elif node in inbound_only:
            cat = "Inbound"
        elif node in outbound_only:
            cat = "Outbound"
        else:
            cat = "Other"
        
        categories[cat]["x"].append(x)
        categories[cat]["y"].append(y)
        categories[cat]["txt"].append(txt)
        categories[cat]["size"].append(size)
    
    for name, cat in categories.items():
        if not cat["x"]:
            continue
        
        traces.append(go.Scatter(
            x=cat["x"], y=cat["y"],
            mode="markers+text" if show_labels else "markers",
            text=[t.split("<br>")[0].replace("<b>", "").replace("</b>", "") for t in cat["txt"]],
            textposition="top center",
            hovertext=cat["txt"],
            hoverinfo="text",
            name=name,
            marker=dict(
                size=cat["size"],
                color=cat["color"],
                opacity=0.85,
                line=dict(width=1, color="#1F2937")
            )
        ))
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        hovermode="closest",
        dragmode="pan",
        paper_bgcolor="#F8FAFC",
        plot_bgcolor="#F8FAFC"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_focus_analysis(focus_builder, builders, edges):
    """Render focus builder flow analysis."""
    st.subheader(f"📍 {focus_builder} - Flow Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Inbound Referrals (who sends to focus)**")
        inbound = (
            edges[edges["Dest_builder"] == focus_builder]
            .groupby("Origin_builder", as_index=False)["Referrals"]
            .sum()
            .sort_values("Referrals", ascending=False)
        )
        
        if not inbound.empty:
            inbound = inbound.merge(
                builders[["BuilderRegionKey", "ROAS", "MediaCost"]],
                left_on="Origin_builder",
                right_on="BuilderRegionKey",
                how="left"
            )
            inbound["Eff_Cost_per_Ref"] = np.where(
                inbound["Referrals"] > 0,
                inbound["MediaCost"] / inbound["Referrals"] / inbound["ROAS"].replace(0, np.nan),
                np.nan
            )
            
            st.dataframe(
                inbound[["Origin_builder", "Referrals", "MediaCost", "ROAS", "Eff_Cost_per_Ref"]]
                .style.format({
                    "Referrals": "{:,.0f}",
                    "MediaCost": "${:,.0f}",
                    "ROAS": "{:.2f}",
                    "Eff_Cost_per_Ref": "${:,.0f}"
                }),
                hide_index=True,
                use_container_width=True
            )
            
            # Best lever recommendation
            best = inbound.loc[inbound["Eff_Cost_per_Ref"].idxmin()] if inbound["Eff_Cost_per_Ref"].notna().any() else None
            if best is not None:
                st.success(f"💡 **Best lever:** {best['Origin_builder']} @ ${best['Eff_Cost_per_Ref']:,.0f}/effective referral")
        else:
            st.info("No inbound referrals")
    
    with col2:
        st.markdown("**Outbound Referrals (where focus sends)**")
        outbound = (
            edges[edges["Origin_builder"] == focus_builder]
            .groupby("Dest_builder", as_index=False)["Referrals"]
            .sum()
            .sort_values("Referrals", ascending=False)
        )
        
        if not outbound.empty:
            st.dataframe(
                outbound.style.format({"Referrals": "{:,.0f}"}),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No outbound referrals")


if __name__ == "__main__":
    main()
