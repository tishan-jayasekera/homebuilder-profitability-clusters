import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go

# Add parent directory to path for imports
root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.network_optimization import (
    OptimConfig,
    calculate_shortfalls,
    analyze_network_leverage,
    generate_investment_strategies,
    optimise_portfolio_media_plan,
)

# =============================================================================
# Streamlit config
# =============================================================================

st.set_page_config(page_title="Referral Network Explorer", page_icon="🔗", layout="wide")


# =============================================================================
# Utilities
# =============================================================================

def fmt_currency(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"${float(x):,.0f}"
    except Exception:
        return str(x)


def fmt_num(x, d=0):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{float(x):,.{d}f}"
    except Exception:
        return str(x)


def detect_cols(df: pd.DataFrame):
    def first(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    return {
        "source": first(["MediaPayer_BuilderRegionKey", "Origin_builder", "Source_BuilderRegionKey", "Source"]),
        "dest": first(["Dest_BuilderRegionKey", "Dest_builder", "DestBuilderRegionKey", "Dest"]),
        "date": first(["lead_date", "RefDate", "ref_date", "LeadDate", "date"]),
        "is_ref": first(["is_referral", "IsReferral", "isReferral"]),
        "spend": first(["MediaCost_referral_event", "MediaCost", "media_cost", "Spend", "Cost"]),
        "target": first(["LeadTarget_from_job", "LeadTarget", "lead_target", "Target"]),
        "end": first(["WIP_JOB_LIVE_END", "JobLiveEnd", "campaign_end", "EndDate"]),
    }


def to_excel_bytes(tables: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in tables.items():
            if df is None:
                continue
            if not isinstance(df, pd.DataFrame):
                continue
            safe = name[:31]
            df.to_excel(writer, sheet_name=safe, index=False)
    buf.seek(0)
    return buf.read()


def build_edges(events: pd.DataFrame, source_col: str, dest_col: str, is_ref_col: str | None):
    df = events.copy()
    if is_ref_col and is_ref_col in df.columns:
        if df[is_ref_col].dtype == object:
            df[is_ref_col] = df[is_ref_col].astype(str).str.lower().isin(["true", "1", "yes", "y"])
        df = df[df[is_ref_col] == True].copy()

    if source_col not in df.columns or dest_col not in df.columns:
        return pd.DataFrame()

    edges = (
        df.groupby([source_col, dest_col])
        .size()
        .reset_index(name="Referrals")
        .rename(columns={source_col: "Source", dest_col: "Dest"})
    )
    return edges


def build_graph(edges: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    if edges is None or edges.empty:
        return G
    for _, r in edges.iterrows():
        s, d, w = r["Source"], r["Dest"], float(r["Referrals"])
        if pd.isna(s) or pd.isna(d) or w <= 0:
            continue
        G.add_edge(str(s), str(d), weight=w)
    return G


def compute_communities(G: nx.DiGraph):
    UG = G.to_undirected()
    if UG.number_of_nodes() == 0:
        return {}, pd.DataFrame()

    try:
        from networkx.algorithms.community import louvain_communities
        comms = louvain_communities(UG, seed=42)
    except Exception:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = greedy_modularity_communities(UG)

    node_to_cluster = {}
    for i, cset in enumerate(comms, start=1):
        for n in cset:
            node_to_cluster[n] = i

    summary = (
        pd.Series(node_to_cluster)
        .reset_index()
        .rename(columns={"index": "Builder", 0: "ClusterId"})
        .groupby("ClusterId")
        .agg(N_builders=("Builder", "nunique"))
        .reset_index()
        .sort_values("N_builders", ascending=False)
    )
    return node_to_cluster, summary


def plot_network(G: nx.DiGraph, node_meta: pd.DataFrame, highlight: dict[str, str] | None = None, show_labels=False):
    if G.number_of_nodes() == 0:
        st.info("No network edges to display.")
        return

    pos = nx.spring_layout(G.to_undirected(), weight="weight", seed=42, k=1.2, iterations=70)

    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=1),
                hoverinfo="none",
                opacity=0.35,
            )
        )

    node_x, node_y, text, size, symbol = [], [], [], [], []
    default_size = 10

    meta = node_meta.set_index("BuilderRegionKey") if (node_meta is not None and not node_meta.empty) else None

    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)

        if meta is not None and n in meta.index:
            shortfall = float(meta.loc[n, "Shortfall"]) if "Shortfall" in meta.columns else 0.0
            demand = float(meta.loc[n, "DemandScore"]) if "DemandScore" in meta.columns else 0.0
            size.append(default_size + min(18, np.sqrt(max(0.0, demand)) / 2.0))
            text.append(f"{n}<br>Shortfall: {fmt_num(shortfall,0)}<br>DemandScore: {fmt_num(demand,0)}")
        else:
            size.append(default_size)
            text.append(str(n))

        if highlight and n in highlight:
            symbol.append("diamond")
        else:
            symbol.append("circle")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text" if show_labels else "markers",
        text=[n for n in G.nodes()] if show_labels else None,
        hovertext=text,
        hoverinfo="text",
        marker=dict(size=size, symbol=symbol, line=dict(width=0.5)),
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True)


def sankey_for_strategy(focus_builder: str, strategy_row: pd.Series):
    source = strategy_row["Source_Builder"]
    transfer = float(strategy_row["Transfer_Rate"])
    inv = float(strategy_row["Investment_Required"])
    ecpr = float(strategy_row["Effective_CPR"])

    delivered = inv / ecpr if (np.isfinite(ecpr) and ecpr > 0) else 0.0
    total_leads = delivered / transfer if (np.isfinite(transfer) and transfer > 0) else delivered
    spill = max(0.0, total_leads - delivered)

    labels = [f"{source}<br>(Source)", f"{focus_builder}<br>(Deficit)", "Other Builders<br>(Spillover)"]
    sources = [0, 0]
    targets = [1, 2]
    values = [delivered, spill]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(pad=12, thickness=18, label=labels),
                link=dict(source=sources, target=targets, value=values),
            )
        ]
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=25, b=10), title=f"Spend {fmt_currency(inv)} → Delivered leads vs spillover")
    return fig


# =============================================================================
# Data Loading (uses session state from Home page)
# =============================================================================

def load_events_data():
    """Load events data from session state (uploaded on Home page)."""
    if 'events_file' not in st.session_state:
        return None
    
    try:
        st.session_state['events_file'].seek(0)  # Reset file pointer
        df = pd.read_excel(st.session_state['events_file'])
        return df
    except Exception as e:
        st.error(f"Error reading events file: {e}")
        return None


# =============================================================================
# App
# =============================================================================

st.title("🔗 Referral Network Explorer")
st.caption("Operational optimiser: identifies urgent lead gaps (pace + runway), uses network leverage to build a low-CPR media plan, and outputs a full audit trail.")

# Load data from session state
events = load_events_data()

if events is None:
    st.warning("⚠️ Please upload events data on the Home page first.")
    st.page_link("app.py", label="← Go to Home", icon="🏠")
    st.stop()

events.columns = [c.strip() for c in events.columns]

det = detect_cols(events)

with st.sidebar:
    st.header("🧩 Column Mapping (auto-detected)")
    source_col = st.selectbox("Source (media payer) column", options=[c for c in events.columns], index=events.columns.get_loc(det["source"]) if det["source"] in events.columns else 0)
    dest_col = st.selectbox("Destination (builder) column", options=[c for c in events.columns], index=events.columns.get_loc(det["dest"]) if det["dest"] in events.columns else 0)

    date_col = st.selectbox(
        "Date column",
        options=["(None)"] + list(events.columns),
        index=(1 + events.columns.get_loc(det["date"])) if det["date"] in events.columns else 0,
    )
    is_ref_col = st.selectbox(
        "is_referral column",
        options=["(None)"] + list(events.columns),
        index=(1 + events.columns.get_loc(det["is_ref"])) if det["is_ref"] in events.columns else 0,
    )
    spend_col = st.selectbox(
        "Spend column",
        options=["(None)"] + list(events.columns),
        index=(1 + events.columns.get_loc(det["spend"])) if det["spend"] in events.columns else 0,
    )
    target_col = st.selectbox(
        "Target column",
        options=["(None)"] + list(events.columns),
        index=(1 + events.columns.get_loc(det["target"])) if det["target"] in events.columns else 0,
    )
    end_col = st.selectbox(
        "Campaign end column",
        options=["(None)"] + list(events.columns),
        index=(1 + events.columns.get_loc(det["end"])) if det["end"] in events.columns else 0,
    )

    st.divider()
    st.header("📅 Analysis as-of")

    asof = pd.Timestamp.today().normalize()
    if date_col != "(None)":
        events[date_col] = pd.to_datetime(events[date_col], errors="coerce")
        months = sorted(events[date_col].dropna().dt.to_period("M").dt.to_timestamp().unique())
        month_options = ["(Latest in data)"] + [pd.Timestamp(m).strftime("%Y-%m") for m in months]
        sel = st.selectbox("As-of month", options=month_options, index=0)

        if sel == "(Latest in data)":
            max_dt = events[date_col].dropna().max()
            asof = pd.to_datetime(max_dt).normalize()
        else:
            asof = pd.Timestamp(sel + "-01") + pd.offsets.MonthEnd(0)
            asof = pd.to_datetime(asof).normalize()
    else:
        st.caption("No date column selected; using today as as-of.")

    st.divider()
    st.header("🧠 Optimiser settings")
    pace_lookback = st.slider("Pace lookback (days)", 7, 60, 14, 1)
    transfer_lookback = st.slider("Transfer lookback (days)", 30, 365, 90, 5)
    prior_strength = st.slider("Transfer smoothing strength", 0.0, 100.0, 25.0, 5.0)
    min_transfer_events = st.slider("Min events per (source→dest)", 0, 30, 5, 1)
    overserve_tol = st.slider("Overserve tolerance", 0.0, 0.30, 0.10, 0.01)

    st.divider()
    st.header("💰 Budget settings")
    max_mult = st.slider("Max spend multiplier", 0.0, 3.0, 1.0, 0.1)
    new_money = st.number_input("New money ($)", min_value=0.0, value=0.0, step=10_000.0)
    overserve_penalty = st.slider("Overserve penalty", 0.0, 1.0, 0.25, 0.05)
    max_step = st.number_input("Max step spend ($)", min_value=1_000.0, value=25_000.0, step=1_000.0)
    show_labels = st.checkbox("Show labels on network", value=False)

cfg = OptimConfig(
    pace_lookback_days=int(pace_lookback),
    transfer_lookback_days=int(transfer_lookback),
    min_transfer_events=int(min_transfer_events),
    prior_strength=float(prior_strength),
    overserve_tolerance=float(overserve_tol),
    max_step_spend=float(max_step),
)

_is_ref = None if is_ref_col == "(None)" else is_ref_col
_date = None if date_col == "(None)" else date_col
_spend = None if spend_col == "(None)" else spend_col
_target = None if target_col == "(None)" else target_col
_end = None if end_col == "(None)" else end_col

missing = [c for c in [source_col, dest_col] if c not in events.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

with st.spinner("Building demand model (pace + runway) ..."):
    demand = calculate_shortfalls(
        events,
        asof=asof,
        cfg=cfg,
        dest_col=dest_col,
        is_ref_col=_is_ref,
        date_col=_date,
        target_col=_target,
        end_col=_end,
    )

with st.spinner("Building network leverage (smoothed transfer rates + eCPR) ..."):
    leverage = analyze_network_leverage(
        events,
        asof=asof,
        cfg=cfg,
        source_col=source_col,
        dest_col=dest_col,
        is_ref_col=_is_ref,
        date_col=_date,
        spend_col=_spend,
    )

edges = build_edges(events, source_col, dest_col, _is_ref)
G = build_graph(edges)

node_to_cluster, cluster_summary = compute_communities(G)

if not demand.empty:
    demand["ClusterId"] = demand["BuilderRegionKey"].map(lambda x: node_to_cluster.get(str(x), np.nan))

# =============================================================================
# Top KPIs
# =============================================================================

k1, k2, k3, k4 = st.columns(4)
k1.metric("As-of", asof.strftime("%Y-%m-%d"))
k2.metric("Builders in deficit", int((demand["Shortfall"] > 0).sum()) if not demand.empty else 0)
k3.metric("Total forecast shortfall", f"{demand['Shortfall'].sum():,.0f} leads" if not demand.empty else "0")
k4.metric("Builders overserved", int((demand["ServiceFlag"] == "OVER").sum()) if not demand.empty else 0)

st.divider()

# =============================================================================
# Situation Room
# =============================================================================

st.header("🔥 Situation Room — highest-risk lead gaps")
top_def = demand[demand["Shortfall"] > 0].head(20).copy() if not demand.empty else pd.DataFrame()

left, right = st.columns([1.35, 1])

with left:
    if top_def.empty:
        st.success("✅ No forecast shortfalls detected (at current pace and runway).")
    else:
        st.dataframe(
            top_def[[
                "BuilderRegionKey", "LeadTarget", "Actual_Referrals",
                "Pace_Leads_per_Day", "Days_Remaining",
                "Projected_Finish", "Shortfall", "DemandScore", "ServiceFlag"
            ]].style.format({
                "LeadTarget": "{:,.0f}",
                "Actual_Referrals": "{:,.0f}",
                "Pace_Leads_per_Day": "{:.2f}",
                "Days_Remaining": "{:,.0f}",
                "Projected_Finish": "{:,.0f}",
                "Shortfall": "{:,.0f}",
                "DemandScore": "{:,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

with right:
    st.subheader("What's driving urgency?")
    st.caption("DemandScore weights forecast shortfall by (i) deadline proximity and (ii) required pace vs current pace.")
    if not top_def.empty:
        b = top_def.iloc[0]["BuilderRegionKey"]
        row = top_def.iloc[0]
        st.metric("Top deficit builder", str(b))
        st.metric("Shortfall", fmt_num(row["Shortfall"], 0))
        st.metric("Days remaining", fmt_num(row["Days_Remaining"], 0))
        st.metric("Required pace", fmt_num(row["Required_Pace"], 2) if "Required_Pace" in row else "—")
        st.metric("Current pace", fmt_num(row["Pace_Leads_per_Day"], 2))

st.divider()

# =============================================================================
# Portfolio plan
# =============================================================================

st.header("🎯 Portfolio Media Plan — close deficits at lowest expected cost")

plan = optimise_portfolio_media_plan(
    demand,
    leverage,
    cfg=cfg,
    max_spend_multiplier=float(max_mult),
    new_money=float(new_money),
    overserve_penalty=float(overserve_penalty),
)

if "error" in plan:
    st.error(plan["error"].iloc[0, 0])
    st.stop()

p1, p2 = st.columns([1.05, 1])

with p1:
    st.subheader("Plan by source (where money goes)")
    if plan["plan_by_source"].empty:
        st.info("No spend allocated (either no deficits, or no productive sources found).")
    else:
        st.dataframe(
            plan["plan_by_source"].style.format({
                "Spend": "${:,.0f}",
                "CPR_base": "${:,.2f}",
                "Leads_Generated": "{:,.0f}",
                "Budget": "${:,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=plan["plan_by_source"]["MediaPayer_BuilderRegionKey"],
            y=plan["plan_by_source"]["Spend"],
        ))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=35, b=10), title="Spend by source")
        st.plotly_chart(fig, use_container_width=True)

with p2:
    st.subheader("Reconciliation (pre vs post)")
    post = plan["post_state"].copy()
    view = post.sort_values("Shortfall", ascending=False).head(25)
    st.dataframe(
        view[[
            "BuilderRegionKey",
            "Shortfall", "Expected_Leads_Added", "Shortfall_Post", "Overserve_Post",
            "ServiceFlag"
        ]].style.format({
            "Shortfall": "{:,.0f}",
            "Expected_Leads_Added": "{:,.0f}",
            "Shortfall_Post": "{:,.0f}",
            "Overserve_Post": "{:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    closed = float((post["Shortfall"].sum() - post["Shortfall_Post"].sum()))
    st.metric("Total shortfall closed (expected)", f"{closed:,.0f} leads")

with st.expander("🔎 Audit trail (edge contributions + allocation log)", expanded=False):
    st.subheader("Edge plan (source → destination expected leads)")
    st.dataframe(plan["plan_edges"].head(400), use_container_width=True)
    st.subheader("Allocation log (step-by-step)")
    st.dataframe(plan["allocation_log"], use_container_width=True)

with st.expander("⬇️ Export optimiser outputs to Excel", expanded=False):
    tables = {
        "demand_table": plan["demand_table"],
        "source_budgets": plan["source_budgets"],
        "plan_by_source": plan["plan_by_source"],
        "plan_edges": plan["plan_edges"],
        "post_state": plan["post_state"],
        "allocation_log": plan["allocation_log"],
        "leverage_table": leverage,
    }
    xbytes = to_excel_bytes(tables)
    st.download_button(
        "Download Excel audit pack",
        data=xbytes,
        file_name=f"network_media_plan_{asof.strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.divider()

# =============================================================================
# Focus builder view
# =============================================================================

st.header("🎯 Focus Builder — best pathways to close one gap (traceable)")

deficit_builders = demand[demand["Shortfall"] > 0]["BuilderRegionKey"].astype(str).tolist() if not demand.empty else []
if not deficit_builders:
    st.info("No deficit builders to analyse.")
else:
    focus = st.selectbox("Select focus builder", options=deficit_builders, index=0)

    focus_row = demand[demand["BuilderRegionKey"].astype(str) == str(focus)].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target", fmt_num(focus_row["LeadTarget"], 0))
    c2.metric("Actual", fmt_num(focus_row["Actual_Referrals"], 0))
    c3.metric("Forecast shortfall", fmt_num(focus_row["Shortfall"], 0))
    c4.metric("Days remaining", fmt_num(focus_row["Days_Remaining"], 0))

    strategies = generate_investment_strategies(str(focus), demand, leverage)

    if strategies.empty:
        st.warning("No historical network pathways found for this builder (in the lookback window).")
    else:
        st.subheader("Ranked strategies (lowest Effective CPR first)")
        st.dataframe(
            strategies.head(25).style.format({
                "Transfer_Rate": "{:.2%}",
                "Base_CPR": "${:,.2f}",
                "Effective_CPR": "${:,.2f}",
                "Investment_Required": "${:,.0f}",
                "Total_Leads_Generated": "{:,.0f}",
                "Excess_Leads": "{:,.0f}",
                "Confidence": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        pick = st.selectbox("Visualise strategy source", options=strategies["Source_Builder"].astype(str).tolist(), index=0)
        sel = strategies[strategies["Source_Builder"].astype(str) == str(pick)].iloc[0]

        st.plotly_chart(sankey_for_strategy(str(focus), sel), use_container_width=True)

st.divider()

# =============================================================================
# Network explorer
# =============================================================================

st.header("🕸️ Network Graph Explorer")

if cluster_summary is not None and not cluster_summary.empty:
    clusters = cluster_summary["ClusterId"].astype(int).tolist()
    labels = {int(r.ClusterId): f"Cluster {int(r.ClusterId)} ({int(r.N_builders)} builders)" for r in cluster_summary.itertuples()}
    sel_cluster = st.selectbox("Filter to cluster (optional)", options=["(All)"] + clusters, format_func=lambda x: labels.get(x, "(All)") if x != "(All)" else "(All)", index=0)
else:
    sel_cluster = "(All)"

highlight = {str(b): "UNDER" for b in demand[demand["Shortfall"] > 0]["BuilderRegionKey"].astype(str).tolist()} if not demand.empty else {}

if sel_cluster != "(All)":
    keep = set([n for n, cid in node_to_cluster.items() if cid == int(sel_cluster)])
    subG = G.subgraph(keep).copy()
    sub_demand = demand[demand["ClusterId"] == int(sel_cluster)].copy() if not demand.empty else pd.DataFrame()
else:
    subG = G
    sub_demand = demand.copy() if not demand.empty else pd.DataFrame()

plot_network(subG, node_meta=sub_demand, highlight=highlight, show_labels=show_labels)

st.caption("Tip: deficits are implicitly highlighted (diamond markers) when they exist. Increase lookback windows if the graph is sparse.")