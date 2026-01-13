"""
Network Intelligence Engine — V2 (Operational + Traceable)
Commercial focus:
- Who is under/over-served (forecast-aware)
- Which network sources can close gaps at lowest effective CPR
- Portfolio media plan that reconciles overserve vs underserve, fully auditable
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go


# ──────────────────────────────────────────────────────────────────────────────
# Repo imports
# ──────────────────────────────────────────────────────────────────────────────
root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data_loader import load_events  # keep existing
from src.normalization import normalize_events
from src.referral_clusters import run_referral_clustering
from src.builder_pnl import build_builder_pnl

# Optimiser: prefer src, fallback to root module
try:
    from src.network_optimization import (
        OptimConfig,
        calculate_shortfalls,
        analyze_network_leverage,
        generate_investment_strategies,
        optimise_portfolio_media_plan,
    )
except Exception:
    from network_optimization import (
        OptimConfig,
        calculate_shortfalls,
        analyze_network_leverage,
        generate_investment_strategies,
        optimise_portfolio_media_plan,
    )

# Formatting helper
try:
    from src.utils import fmt_currency
except Exception:
    def fmt_currency(x):
        try:
            return f"${float(x):,.0f}"
        except Exception:
            return "—"


# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Network Intelligence", page_icon="🔗", layout="wide")


# ──────────────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
/* Layout polish */
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
hr { margin: 1.1rem 0; }

/* Cards */
.ni-card{
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 16px 16px 14px 16px;
  box-shadow: 0 8px 20px -14px rgba(0,0,0,0.25);
}
.ni-card:hover{ border-color:#d1d5db; }

.ni-kpi-label{
  font-size: 0.74rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 700;
  color: #6b7280;
}
.ni-kpi-value{
  font-size: 1.55rem;
  font-weight: 800;
  color: #111827;
  line-height: 1.05;
  margin-top: 2px;
}
.ni-kpi-sub{
  font-size: 0.86rem;
  color: #6b7280;
  margin-top: 4px;
}

/* Badges */
.ni-badge{
  display:inline-flex; align-items:center;
  padding: 4px 10px; border-radius: 999px;
  font-size: 0.78rem; font-weight: 800;
  border:1px solid transparent;
}
.ni-red{ background:#fef2f2; color:#991b1b; border-color:#fecaca; }
.ni-amber{ background:#fffbeb; color:#92400e; border-color:#fde68a; }
.ni-green{ background:#f0fdf4; color:#166534; border-color:#bbf7d0; }
.ni-blue{ background:#eff6ff; color:#1e40af; border-color:#dbeafe; }

/* Section headers */
.ni-section{
  display:flex; align-items:center; justify-content:space-between;
  margin: 0.4rem 0 0.8rem 0;
}
.ni-section-title{
  font-size:1.15rem; font-weight:900; color:#111827;
}
.ni-section-sub{ font-size:0.92rem; color:#6b7280; }

/* Table helpers */
.small-note { font-size: 0.88rem; color: #6b7280; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────
if "campaign_targets" not in st.session_state:
    st.session_state.campaign_targets = []
if "selected_builder" not in st.session_state:
    st.session_state.selected_builder = None
if "plan_result" not in st.session_state:
    st.session_state.plan_result = None


def _add_target(b: str):
    if b and b not in st.session_state.campaign_targets:
        st.session_state.campaign_targets.append(b)


def _remove_target(b: str):
    if b in st.session_state.campaign_targets:
        st.session_state.campaign_targets.remove(b)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_process_from_upload(upload_key: str) -> Optional[pd.DataFrame]:
    """Upload key ensures cache busts when file changes."""
    if "events_file" not in st.session_state:
        return None
    events = load_events(st.session_state["events_file"])
    if events is None or len(events) == 0:
        return None
    return normalize_events(events)


def _first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _required_cols_check(events: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Minimal viable columns for full engine; we’ll still run partial if missing."""
    missing = []
    if _first_col(events, ["lead_date", "RefDate", "ref_date", "LeadDate", "date"]) is None:
        missing.append("lead_date (or RefDate/ref_date)")
    if _first_col(events, ["is_referral", "IsReferral", "isReferral"]) is None:
        missing.append("is_referral")
    if _first_col(events, ["Dest_BuilderRegionKey", "Dest_builder", "DestBuilderRegionKey", "Dest"]) is None:
        missing.append("Dest_BuilderRegionKey")
    if _first_col(events, ["MediaPayer_BuilderRegionKey", "Origin_builder", "Source_BuilderRegionKey", "Source"]) is None:
        missing.append("MediaPayer_BuilderRegionKey")
    return (len(missing) == 0), missing


def _filter_by_date(events: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    date_col = _first_col(events, ["lead_date", "RefDate", "ref_date", "LeadDate", "date"])
    if not date_col or start is None or end is None:
        return events
    df = events.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df[(df[date_col].notna()) & (df[date_col] >= start) & (df[date_col] <= end)].copy()


def _get_all_builders(events: pd.DataFrame) -> List[str]:
    b = set()
    for col in ["Dest_BuilderRegionKey", "MediaPayer_BuilderRegionKey", "Origin_builder", "Dest_builder"]:
        if col in events.columns:
            b.update(events[col].dropna().astype(str).unique())
    return sorted(b)


def _get_builder_connections(events: pd.DataFrame, builder: str) -> Dict[str, List[Dict]]:
    """Direct connections (inbound/outbound/mutual), robust to column names."""
    if not builder:
        return {"inbound": [], "outbound": [], "two_way": []}

    is_ref = _first_col(events, ["is_referral", "IsReferral", "isReferral"])
    src = _first_col(events, ["MediaPayer_BuilderRegionKey", "Origin_builder", "Source_BuilderRegionKey", "Source"])
    dst = _first_col(events, ["Dest_BuilderRegionKey", "Dest_builder", "DestBuilderRegionKey", "Dest"])
    spend = _first_col(events, ["MediaCost_referral_event", "MediaCost", "media_cost", "Spend", "Cost"])
    leadid = _first_col(events, ["LeadId", "lead_id", "id"])

    df = events.copy()
    if is_ref is not None:
        if df[is_ref].dtype == object:
            df[is_ref] = df[is_ref].astype(str).str.lower().isin(["true", "1", "yes", "y"])
        df = df[df[is_ref] == True].copy()

    if df.empty or src is None or dst is None:
        return {"inbound": [], "outbound": [], "two_way": []}

    # inbound: src -> builder
    inbound = df[df[dst].astype(str) == str(builder)].copy()
    if leadid and leadid in inbound.columns:
        agg_in = inbound.groupby(src).agg(
            refs_in=(leadid, "count"),
            spend_in=(spend, "sum") if spend else (leadid, "count"),
        ).reset_index()
    else:
        agg_in = inbound.groupby(src).size().reset_index(name="refs_in")
        agg_in["spend_in"] = np.nan

    agg_in = agg_in.rename(columns={src: "partner"})

    # outbound: builder -> dst
    outbound = df[df[src].astype(str) == str(builder)].copy()
    if leadid and leadid in outbound.columns:
        agg_out = outbound.groupby(dst).agg(refs_out=(leadid, "count")).reset_index()
    else:
        agg_out = outbound.groupby(dst).size().reset_index(name="refs_out")
    agg_out = agg_out.rename(columns={dst: "partner"})

    merged = pd.merge(agg_in, agg_out, on="partner", how="outer").fillna(0)
    two = merged[(merged["refs_in"] > 0) & (merged["refs_out"] > 0)].copy()
    ino = merged[(merged["refs_in"] > 0) & (merged["refs_out"] == 0)].copy()
    outo = merged[(merged["refs_in"] == 0) & (merged["refs_out"] > 0)].copy()

    return {
        "two_way": two.sort_values(["refs_in", "refs_out"], ascending=False).to_dict("records"),
        "inbound": ino.sort_values("refs_in", ascending=False).to_dict("records"),
        "outbound": outo.sort_values("refs_out", ascending=False).to_dict("records"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Viz helpers
# ──────────────────────────────────────────────────────────────────────────────
def kpi_card(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="ni-card">
          <div class="ni-kpi-label">{label}</div>
          <div class="ni-kpi-value">{value}</div>
          <div class="ni-kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge_for_service(flag: str) -> str:
    if flag == "UNDER":
        return '<span class="ni-badge ni-red">UNDER</span>'
    if flag == "OVER":
        return '<span class="ni-badge ni-amber">OVER</span>'
    return '<span class="ni-badge ni-green">ON TRACK</span>'


def render_network_map(G: nx.Graph, pos: Dict, selected: Optional[str] = None, connections: Optional[Dict] = None) -> go.Figure:
    fig = go.Figure()

    highlight = set()
    role = {}
    if selected and connections:
        highlight.add(selected)
        role[selected] = "selected"
        for r in connections.get("inbound", []):
            highlight.add(r["partner"]); role[r["partner"]] = "in"
        for r in connections.get("outbound", []):
            highlight.add(r["partner"]); role[r["partner"]] = "out"
        for r in connections.get("two_way", []):
            highlight.add(r["partner"]); role[r["partner"]] = "mutual"

    ROLE_COL = {
        "selected": "#ef4444",
        "in": "#10b981",
        "out": "#f59e0b",
        "mutual": "#3b82f6",
        "dim": "#e5e7eb",
    }

    # Background edges (single trace)
    ex, ey = [], []
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ex += [x0, x1, None]
        ey += [y0, y1, None]
    fig.add_trace(go.Scattergl(
        x=ex, y=ey, mode="lines",
        line=dict(color="#e5e7eb", width=0.6),
        opacity=0.35 if selected else 0.55,
        hoverinfo="skip", showlegend=False
    ))

    # Highlight edges around selected
    if selected:
        for u, v, data in G.edges(data=True):
            if u not in pos or v not in pos:
                continue
            if u != selected and v != selected:
                continue
            other = v if u == selected else u
            col = ROLE_COL.get(role.get(other, "dim"), "#9ca3af")
            w = 1.8 + 0.15 * float(data.get("weight", 1))
            x0, y0 = pos[u]; x1, y1 = pos[v]
            fig.add_trace(go.Scattergl(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(color=col, width=w),
                opacity=0.95, hoverinfo="skip", showlegend=False
            ))

    # Nodes
    node_x, node_y, node_color, node_size, node_text, node_op = [], [], [], [], [], []
    deg = dict(G.degree(weight="weight"))
    mx = max(deg.values()) if deg else 1.0

    for n in G.nodes():
        if n not in pos:
            continue
        x, y = pos[n]
        node_x.append(x); node_y.append(y)

        base = 10 + 22 * (deg.get(n, 0) / mx)
        if selected:
            if n == selected:
                c = ROLE_COL["selected"]; s = base + 12; op = 1.0
                t = f"<b>{n}</b><br>FOCUS<br>Weighted degree: {deg.get(n,0):.0f}"
            elif n in highlight:
                r = role.get(n, "dim")
                c = ROLE_COL.get(r, "#9ca3af"); s = base + 5; op = 0.95
                lab = {"in":"Inbound source","out":"Outbound dest","mutual":"Mutual partner"}.get(r,"")
                t = f"<b>{n}</b><br>{lab}<br>Weighted degree: {deg.get(n,0):.0f}"
            else:
                c = "#9ca3af"; s = base * 0.65; op = 0.20
                t = f"{n}"
        else:
            c = "#3b82f6"; s = base; op = 0.90
            t = f"<b>{n}</b><br>Weighted degree: {deg.get(n,0):.0f}"

        node_color.append(c); node_size.append(s); node_text.append(t); node_op.append(op)

    fig.add_trace(go.Scattergl(
        x=node_x, y=node_y, mode="markers",
        marker=dict(size=node_size, color=node_color, opacity=node_op, line=dict(color="white", width=1.2)),
        text=node_text, hoverinfo="text", showlegend=False
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=620,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode="closest",
        dragmode="pan",
    )
    return fig


def sankey_from_plan_edges(plan_edges: pd.DataFrame, targets_set: set, min_leads: float = 2.0) -> Optional[go.Figure]:
    if plan_edges is None or plan_edges.empty:
        return None

    df = plan_edges.copy()
    df = df[df["Expected_Leads"] >= float(min_leads)].copy()
    if df.empty:
        return None

    # Keep only sources that actually hit targets OR show all with highlight
    df["IsTarget"] = df["Dest"].astype(str).isin({str(x) for x in targets_set})

    # Nodes
    sources = df["Source"].astype(str).unique().tolist()
    targets = df["Dest"].astype(str).unique().tolist()

    labels = sources + targets
    idx = {k: i for i, k in enumerate(labels)}

    s_idx = df["Source"].astype(str).map(idx).tolist()
    t_idx = df["Dest"].astype(str).map(idx).tolist()
    vals = df["Expected_Leads"].astype(float).tolist()

    # Color targets lightly
    node_colors = []
    for lab in labels:
        if lab in sources:
            node_colors.append("#3b82f6")
        elif lab in targets_set:
            node_colors.append("#10b981")
        else:
            node_colors.append("#9ca3af")

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=14, thickness=18,
            line=dict(color="rgba(0,0,0,0.15)", width=0.8),
            label=labels, color=node_colors
        ),
        link=dict(
            source=s_idx, target=t_idx, value=vals,
            color="rgba(148,163,184,0.35)",
            hovertemplate="Leads: %{value:.1f}<extra></extra>"
        )
    )])
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def export_audit_pack_xlsx(
    out_path: str,
    sheets: Dict[str, pd.DataFrame],
) -> bytes:
    """Return XLSX bytes for st.download_button."""
    import io
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in sheets.items():
            if df is None:
                continue
            if not isinstance(df, pd.DataFrame):
                continue
            df.to_excel(w, sheet_name=str(name)[:31], index=False)
    buf.seek(0)
    return buf.read()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown(
        """
        <div class="ni-section">
          <div>
            <div class="ni-section-title">🔗 Network Intelligence OS</div>
            <div class="ni-section-sub">Forecast-aware shortfalls → network leverage → lowest-eCPR media plan (audited)</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    events = load_and_process_from_upload("events_file")
    if events is None:
        st.warning("⚠️ Please upload Events on the Home page first.")
        return

    ok, missing = _required_cols_check(events)
    if not ok:
        st.error("Your events file is missing required columns for the full engine:")
        st.write(missing)
        st.info("You can still browse clusters, but optimisation may be partial.")
        st.divider()

    # ──────────────────────────────────────────────────────────────────────────
    # Sidebar controls (period + engine params)
    # ──────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Controls")

        date_col = _first_col(events, ["lead_date", "RefDate", "ref_date", "LeadDate", "date"])
        if date_col:
            tmp = events.copy()
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            ds = tmp[date_col].dropna()
        else:
            ds = pd.Series([], dtype="datetime64[ns]")

        if not ds.empty:
            min_d, max_d = ds.min().date(), ds.max().date()
            dr = st.date_input("Analysis Period", value=(min_d, max_d))
            if dr and len(dr) == 2:
                start_date, end_date = pd.Timestamp(dr[0]), pd.Timestamp(dr[1])
            else:
                start_date, end_date = None, None
        else:
            st.caption("No date column found; using all rows.")
            start_date, end_date = None, None

        st.markdown("### 🧠 Engine")
        cfg = OptimConfig(
            pace_lookback_days=st.slider("Pace lookback (days)", 7, 60, 14, 1),
            transfer_lookback_days=st.slider("Transfer lookback (days)", 30, 365, 90, 5),
            overserve_tolerance=st.slider("Overserve tolerance", 0.0, 0.5, 0.10, 0.01),
            prior_strength=st.slider("Transfer smoothing (prior strength)", 5.0, 100.0, 25.0, 5.0),
            min_transfer_events=st.slider("Min transfer events", 1, 25, 5, 1),
            urgency_days_scale=st.slider("Urgency days scale", 7.0, 60.0, 21.0, 1.0),
            pace_gap_alpha=st.slider("Pace gap weight", 0.5, 3.0, 1.25, 0.05),
            max_step_spend=st.number_input("Max spend step", min_value=1_000, value=25_000, step=1_000),
            min_step_spend=st.number_input("Min spend step", min_value=100, value=500, step=100),
        )

        st.markdown("### 🧩 Clustering")
        max_clusters = st.slider("Max clusters", 3, 25, 12, 1)
        resolution = st.slider("Resolution", 0.5, 2.5, 1.5, 0.1)

        st.markdown("### 💸 Media plan")
        new_money = st.number_input("New money ($)", min_value=0, value=0, step=5_000)
        max_mult = st.slider("Max spend multiplier (vs baseline)", 0.5, 3.0, 1.0, 0.05)
        overserve_penalty = st.slider("Overserve penalty", 0.0, 1.0, 0.25, 0.05)

        st.divider()
        st.markdown("### 🛒 Campaign Cart")
        if st.session_state.campaign_targets:
            for t in list(st.session_state.campaign_targets):
                c1, c2 = st.columns([0.82, 0.18])
                with c1:
                    st.write(f"• {t}")
                with c2:
                    if st.button("✕", key=f"rm_{t}"):
                        _remove_target(t)
                        st.rerun()
            if st.button("Clear cart", use_container_width=True):
                st.session_state.campaign_targets = []
                st.rerun()
        else:
            st.caption("Add builders from Demand / Search.")

    # ──────────────────────────────────────────────────────────────────────────
    # Filtered events (period) + build model tables
    # ──────────────────────────────────────────────────────────────────────────
    events_f = _filter_by_date(events, start_date, end_date)
    asof = pd.Timestamp(end_date).normalize() if end_date is not None else pd.Timestamp.today().normalize()

    with st.spinner("Building network intelligence…"):
        # 1) Demand (forecast-aware)
        demand = calculate_shortfalls(events_f, asof=asof, cfg=cfg)

        # 2) Leverage (smoothed transfers + base CPR)
        leverage = analyze_network_leverage(events_f, asof=asof, cfg=cfg)

        # 3) Clusters + Graph
        clus = run_referral_clustering(events_f, resolution=resolution, target_max_clusters=max_clusters)
        G = clus.get("graph", nx.Graph())
        edges_clean = clus.get("edges_clean", pd.DataFrame())
        builder_master = clus.get("builder_master", pd.DataFrame())
        cluster_summary = clus.get("cluster_summary", pd.DataFrame())

        # 4) P&L enrich (optional)
        pnl = build_builder_pnl(events_f, lens="recipient", freq="ALL")
        if isinstance(pnl, pd.DataFrame) and not pnl.empty and "BuilderRegionKey" in pnl.columns and not builder_master.empty:
            keep = [c for c in ["BuilderRegionKey", "Profit", "ROAS", "MediaCost", "Referrals_in"] if c in pnl.columns]
            pnl_sub = pnl[keep].copy()
            builder_master = builder_master.merge(pnl_sub, on="BuilderRegionKey", how="left")
            for c in ["Profit", "ROAS", "MediaCost"]:
                if c in builder_master.columns:
                    builder_master[c] = builder_master[c].fillna(0.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Tabs (exec → demand → network → plan → export)
    # ──────────────────────────────────────────────────────────────────────────
    tab_exec, tab_demand, tab_network, tab_plan, tab_export = st.tabs(
        ["📌 Executive", "⚡ Demand", "🕸️ Network", "💸 Media Plan", "📦 Export"]
    )

    # ──────────────────────────────────────────────────────────────────────────
    # EXEC TAB
    # ──────────────────────────────────────────────────────────────────────────
    with tab_exec:
        st.markdown('<div class="ni-section"><div class="ni-section-title">Executive Snapshot</div><div class="ni-section-sub">What matters right now</div></div>', unsafe_allow_html=True)

        if demand is None or demand.empty:
            st.warning("No demand table generated.")
        else:
            under = demand[demand["ServiceFlag"] == "UNDER"].copy()
            over = demand[demand["ServiceFlag"] == "OVER"].copy()

            total_shortfall = under["Shortfall"].sum() if not under.empty else 0.0
            med_days = float(np.nanmedian(under["Days_Remaining_Fill"])) if not under.empty else np.nan

            # quick spend estimate: sum(best ecpr per under builder) * shortfall
            est = 0.0
            if leverage is not None and not leverage.empty and not under.empty:
                for b, sf in under.set_index("BuilderRegionKey")["Shortfall"].to_dict().items():
                    cand = leverage[leverage["Dest_BuilderRegionKey"].astype(str) == str(b)].copy()
                    cand = cand[cand.get("Pass_Min_Events", True) == True]
                    if cand.empty:
                        continue
                    best_ecpr = float(cand["eCPR"].replace([np.inf, -np.inf], np.nan).dropna().min()) if cand["eCPR"].notna().any() else np.nan
                    if np.isfinite(best_ecpr):
                        est += float(sf) * best_ecpr

            c1, c2, c3, c4 = st.columns(4)
            with c1: kpi_card("Builders under-served", f"{len(under):,}", "Forecast shortfall > 0")
            with c2: kpi_card("Total shortfall", f"{total_shortfall:,.0f}", "Leads required to hit target")
            with c3: kpi_card("Median days remaining", "—" if not np.isfinite(med_days) else f"{int(med_days):,}", "Campaign urgency signal")
            with c4: kpi_card("Rough close-cost", "—" if est <= 0 else fmt_currency(est), "Using best historical eCPR per target")

            st.markdown("#### 🔥 Top at-risk builders")
            hot = demand.copy()
            hot["ServiceBadge"] = hot["ServiceFlag"].apply(badge_for_service)
            cols = ["BuilderRegionKey", "ServiceBadge", "LeadTarget", "Actual_Referrals", "Pace_Leads_per_Day", "Days_Remaining", "Projected_Finish", "Shortfall", "Surplus", "DemandScore"]
            cols = [c for c in cols if c in hot.columns]
            hot_view = hot.loc[:, cols].head(30).copy()

            st.dataframe(
                hot_view,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ServiceBadge": st.column_config.TextColumn("Service", width="small"),
                    "Pace_Leads_per_Day": st.column_config.NumberColumn("Pace/day", format="%.2f"),
                    "DemandScore": st.column_config.NumberColumn("DemandScore", format="%.1f"),
                },
            )
            st.caption("DemandScore = Shortfall × Urgency × Pace gap. Use it to prioritise where the next dollar goes.")

    # ──────────────────────────────────────────────────────────────────────────
    # DEMAND TAB
    # ──────────────────────────────────────────────────────────────────────────
    with tab_demand:
        st.markdown('<div class="ni-section"><div class="ni-section-title">Demand & Coverage</div><div class="ni-section-sub">Forecast-aware: pace + days remaining</div></div>', unsafe_allow_html=True)

        if demand is None or demand.empty:
            st.warning("No demand table available.")
        else:
            under = demand[demand["ServiceFlag"] == "UNDER"].copy()
            over = demand[demand["ServiceFlag"] == "OVER"].copy()

            d1, d2, d3 = st.columns([1.2, 1.0, 0.8])
            with d1:
                default_pick = under["BuilderRegionKey"].iloc[0] if not under.empty else demand["BuilderRegionKey"].iloc[0]
                focus = st.selectbox("Focus builder", options=demand["BuilderRegionKey"].astype(str).tolist(), index=int(demand.index[demand["BuilderRegionKey"] == default_pick][0]) if default_pick in demand["BuilderRegionKey"].values else 0)
            with d2:
                if st.button("➕ Add focus to cart", use_container_width=True):
                    _add_target(str(focus)); st.rerun()
            with d3:
                topn = st.number_input("Auto-target Top N UNDER", min_value=0, value=0, step=5, help="Set >0 to auto-populate cart from the demand table (Top N by DemandScore).")
                if topn and topn > 0:
                    picks = under["BuilderRegionKey"].astype(str).head(int(topn)).tolist()
                    for p in picks:
                        _add_target(p)
                    st.success(f"Added {len(picks)} builders to cart.")
                    st.rerun()

            # Focus row summary
            fr = demand[demand["BuilderRegionKey"].astype(str) == str(focus)].head(1)
            if not fr.empty:
                r = fr.iloc[0]
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Service", r.get("ServiceFlag", "—"))
                m2.metric("Target", f"{r.get('LeadTarget', 0):,.0f}")
                m3.metric("Actual", f"{r.get('Actual_Referrals', 0):,.0f}")
                m4.metric("Shortfall", f"{r.get('Shortfall', 0):,.0f}")
                m5.metric("Days remaining", "—" if pd.isna(r.get("Days_Remaining", np.nan)) else f"{int(r.get('Days_Remaining')):,}")

            st.divider()

            st.markdown("#### 🎯 Best sources to close the focus gap (lowest eCPR)")
            if leverage is None or leverage.empty:
                st.info("No leverage table (transfers) available.")
            else:
                strat = generate_investment_strategies(str(focus), demand, leverage)
                if strat is None or strat.empty:
                    st.info("No usable historical paths to this builder (or shortfall is 0).")
                else:
                    st.dataframe(
                        strat.head(20),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Transfer_Rate": st.column_config.NumberColumn("Transfer", format="%.2%"),
                            "Base_CPR": st.column_config.NumberColumn("Base CPR", format="$%.0f"),
                            "Effective_CPR": st.column_config.NumberColumn("eCPR", format="$%.0f"),
                            "Investment_Required": st.column_config.NumberColumn("Investment", format="$%.0f"),
                            "Confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
                        },
                    )
                    st.caption("eCPR = CPR_base / smoothed transfer rate. Confidence rises with more source volume.")

    # ──────────────────────────────────────────────────────────────────────────
    # NETWORK TAB
    # ──────────────────────────────────────────────────────────────────────────
    with tab_network:
        st.markdown('<div class="ni-section"><div class="ni-section-title">Network Map & Builder Drilldown</div><div class="ni-section-sub">Explore clusters, partnerships, and flow concentration</div></div>', unsafe_allow_html=True)

        all_builders = _get_all_builders(events_f)
        cA, cB = st.columns([1.3, 0.7])
        with cA:
            selected = st.selectbox("Search builder", options=[""] + all_builders, index=0)
        with cB:
            if st.button("➕ Add to cart", use_container_width=True, disabled=(not selected)):
                _add_target(str(selected))
                st.rerun()

        if selected:
            st.session_state.selected_builder = str(selected)

        c1, c2 = st.columns([1.65, 0.95])

        with c1:
            if len(G.nodes) == 0:
                st.info("No graph to render for this period.")
            else:
                pos = nx.spring_layout(G, seed=42, k=0.65, iterations=50, weight="weight")
                conns = _get_builder_connections(events_f, st.session_state.selected_builder) if st.session_state.selected_builder else None
                fig = render_network_map(G, pos, st.session_state.selected_builder, conns)
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            b = st.session_state.selected_builder
            if not b:
                st.info("Select a builder to see the drilldown.")
            else:
                st.markdown("#### Builder panel")
                # Demand summary
                if demand is not None and not demand.empty:
                    row = demand[demand["BuilderRegionKey"].astype(str) == str(b)].head(1)
                    if not row.empty:
                        rr = row.iloc[0]
                        st.markdown(f"**Service:** {rr.get('ServiceFlag','—')}  \n"
                                    f"**Shortfall:** {rr.get('Shortfall',0):,.0f}  \n"
                                    f"**Pace/day:** {rr.get('Pace_Leads_per_Day',0):.2f}  \n"
                                    f"**Days remaining:** {rr.get('Days_Remaining','—')}")
                # Connections
                conns = _get_builder_connections(events_f, b)
                t_in, t_out, t_mut = st.tabs(["📥 Inbound", "📤 Outbound", "🤝 Mutual"])
                with t_in:
                    df = pd.DataFrame(conns.get("inbound", []))
                    if df.empty:
                        st.caption("No inbound sources.")
                    else:
                        st.dataframe(df.rename(columns={"partner":"Source","refs_in":"Refs","spend_in":"Spend"}), use_container_width=True, hide_index=True)
                with t_out:
                    df = pd.DataFrame(conns.get("outbound", []))
                    if df.empty:
                        st.caption("No outbound destinations.")
                    else:
                        st.dataframe(df.rename(columns={"partner":"Dest","refs_out":"Refs"}), use_container_width=True, hide_index=True)
                with t_mut:
                    df = pd.DataFrame(conns.get("two_way", []))
                    if df.empty:
                        st.caption("No mutual partners.")
                    else:
                        st.dataframe(df.rename(columns={"partner":"Partner","refs_in":"In","refs_out":"Out"}), use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────────────
    # MEDIA PLAN TAB
    # ──────────────────────────────────────────────────────────────────────────
    with tab_plan:
        st.markdown('<div class="ni-section"><div class="ni-section-title">Portfolio Media Plan</div><div class="ni-section-sub">Close forecast shortfalls at lowest cost (network-aware)</div></div>', unsafe_allow_html=True)

        if demand is None or demand.empty:
            st.warning("No demand table available.")
        elif leverage is None or leverage.empty:
            st.warning("No leverage table available.")
        else:
            # Target set selection
            under = demand[demand["ServiceFlag"] == "UNDER"].copy()
            mode = st.radio("Target set", ["Cart", "All UNDER builders"], horizontal=True)

            if mode == "Cart":
                targets = [str(x) for x in st.session_state.campaign_targets]
            else:
                targets = under["BuilderRegionKey"].astype(str).tolist()

            if not targets:
                st.info("Add builders to cart (or choose All UNDER builders).")
            else:
                # Slice demand to include all builders, but focus deficits weights on targets
                demand_seen = demand.copy()
                # Build weights: boost target builders in DemandScore, keep others for overserve penalties
                demand_seen["IsTarget"] = demand_seen["BuilderRegionKey"].astype(str).isin(set(targets))
                demand_seen.loc[~demand_seen["IsTarget"], "DemandScore"] = 0.0

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    kpi_card("Targets", f"{len(targets):,}", "Builders in optimisation set")
                with c2:
                    kpi_card("Total shortfall", f"{under[under['BuilderRegionKey'].astype(str).isin(set(targets))]['Shortfall'].sum():,.0f}", "Forecast shortfalls only")
                with c3:
                    kpi_card("New money", fmt_currency(new_money), "Added to baseline budgets")
                with c4:
                    kpi_card("Max spend mult", f"{max_mult:.2f}×", "Budget guardrail")

                if st.button("⚡ Generate portfolio plan", type="primary", use_container_width=True):
                    res = optimise_portfolio_media_plan(
                        demand_seen,
                        leverage,
                        cfg=cfg,
                        max_spend_multiplier=float(max_mult),
                        new_money=float(new_money),
                        overserve_penalty=float(overserve_penalty),
                    )
                    st.session_state.plan_result = res

                res = st.session_state.plan_result
                if res is None:
                    st.info("Generate a plan to see allocations and reconciliation.")
                else:
                    if "error" in res:
                        st.error("Plan engine error:")
                        st.dataframe(res["error"], use_container_width=True, hide_index=True)
                    else:
                        plan_by_source = res.get("plan_by_source", pd.DataFrame())
                        plan_edges = res.get("plan_edges", pd.DataFrame())
                        post = res.get("post_state", pd.DataFrame())
                        log = res.get("allocation_log", pd.DataFrame())
                        budgets = res.get("source_budgets", pd.DataFrame())

                        # Summary
                        tgt_set = set(targets)
                        pre_tgt = demand[demand["BuilderRegionKey"].astype(str).isin(tgt_set)].copy()
                        post_tgt = post[post["BuilderRegionKey"].astype(str).isin(tgt_set)].copy()

                        pre_sf = float(pre_tgt["Shortfall"].sum()) if not pre_tgt.empty else 0.0
                        post_sf = float(post_tgt["Shortfall_Post"].sum()) if not post_tgt.empty else 0.0
                        covered = max(0.0, pre_sf - post_sf)
                        spend = float(plan_by_source["Spend"].sum()) if (plan_by_source is not None and not plan_by_source.empty) else 0.0
                        eff_cpr = (spend / covered) if covered > 0 else np.nan
                        overserve_added = float(post["Overserve_Post"].sum()) if post is not None and not post.empty else 0.0

                        s1, s2, s3, s4 = st.columns(4)
                        with s1: kpi_card("Spend", fmt_currency(spend), "Total planned spend")
                        with s2: kpi_card("Shortfall covered", f"{covered:,.0f}", f"Pre {pre_sf:,.0f} → Post {post_sf:,.0f}")
                        with s3: kpi_card("Effective CPR", "—" if not np.isfinite(eff_cpr) else fmt_currency(eff_cpr), "Spend / covered shortfall")
                        with s4: kpi_card("Overserve (post)", f"{overserve_added:,.0f}", "Penalty-managed spillover")

                        st.divider()

                        # Sankey
                        st.markdown("#### Flow map (Expected leads)")
                        fig_s = sankey_from_plan_edges(plan_edges, tgt_set, min_leads=2.0)
                        if fig_s:
                            st.plotly_chart(fig_s, use_container_width=True)
                        else:
                            st.caption("No meaningful flows to chart (try reducing min transfer events or increasing budget).")

                        # Tables
                        st.markdown("#### Buy list (by source)")
                        if plan_by_source is None or plan_by_source.empty:
                            st.info("No sources were selected (budget or leverage constraints).")
                        else:
                            st.dataframe(
                                plan_by_source,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Spend": st.column_config.NumberColumn("Spend", format="$%.0f"),
                                    "CPR_base": st.column_config.NumberColumn("Base CPR", format="$%.0f"),
                                    "Leads_Generated": st.column_config.NumberColumn("Leads", format="%.1f"),
                                    "Budget": st.column_config.NumberColumn("Budget cap", format="$%.0f"),
                                },
                            )

                        st.markdown("#### Attribution (source → dest)")
                        if plan_edges is None or plan_edges.empty:
                            st.caption("No edge attribution table.")
                        else:
                            # highlight targets first
                            pe = plan_edges.copy()
                            pe["IsTarget"] = pe["Dest"].astype(str).isin(tgt_set)
                            pe = pe.sort_values(["IsTarget", "Expected_Leads"], ascending=[False, False])
                            st.dataframe(
                                pe.head(250),
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Expected_Leads": st.column_config.NumberColumn("Expected leads", format="%.1f"),
                                    "Spend_on_Source": st.column_config.NumberColumn("Spend", format="$%.0f"),
                                    "Effective_CPR_to_Dest": st.column_config.NumberColumn("eCPR(dest)", format="$%.0f"),
                                },
                            )

                        st.markdown("#### Reconciliation (pre → post)")
                        if post is not None and not post.empty:
                            post_view = post.copy()
                            post_view["IsTarget"] = post_view["BuilderRegionKey"].astype(str).isin(tgt_set)
                            post_view = post_view.sort_values(["IsTarget", "Shortfall_Post", "Overserve_Post"], ascending=[False, False, False])
                            keep = [
                                "BuilderRegionKey","IsTarget","ServiceFlag","LeadTarget","Projected_Finish","Expected_Leads_Added",
                                "Projected_Finish_Post","Shortfall","Shortfall_Post","Overserve_Post","DemandScore"
                            ]
                            keep = [c for c in keep if c in post_view.columns]
                            st.dataframe(post_view[keep].head(300), use_container_width=True, hide_index=True)

                        with st.expander("Allocation log (debug/trace)", expanded=False):
                            st.dataframe(log, use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────────────
    # EXPORT TAB
    # ──────────────────────────────────────────────────────────────────────────
    with tab_export:
        st.markdown('<div class="ni-section"><div class="ni-section-title">Audit Pack</div><div class="ni-section-sub">Download every table behind the plan</div></div>', unsafe_allow_html=True)

        res = st.session_state.plan_result
        if res is None or ("error" in (res or {})):
            st.info("Generate a plan first (Media Plan tab) to export a complete audit pack.")
        else:
            sheets = {
                "Demand": res.get("demand_table", pd.DataFrame()),
                "Leverage": leverage if leverage is not None else pd.DataFrame(),
                "SourceBudgets": res.get("source_budgets", pd.DataFrame()),
                "PlanBySource": res.get("plan_by_source", pd.DataFrame()),
                "PlanEdges": res.get("plan_edges", pd.DataFrame()),
                "PostState": res.get("post_state", pd.DataFrame()),
                "AllocationLog": res.get("allocation_log", pd.DataFrame()),
                "ClusterSummary": cluster_summary if isinstance(cluster_summary, pd.DataFrame) else pd.DataFrame(),
                "EdgesClean": edges_clean if isinstance(edges_clean, pd.DataFrame) else pd.DataFrame(),
            }
            xbytes = export_audit_pack_xlsx("network_audit_pack.xlsx", sheets)
            st.download_button(
                "📥 Download Network Audit Pack (XLSX)",
                data=xbytes,
                file_name=f"network_audit_pack_{pd.Timestamp.today().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            st.caption("Includes: demand model, smoothed transfers, budgets, allocations, attribution, and reconciliation.")


if __name__ == "__main__":
    main()
