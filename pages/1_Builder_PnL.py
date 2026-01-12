"""
Builder P&L Dashboard — Executive P&L Review (McKinsey-style)
Filename: pages/1_Builder_PnL.py

Design goals:
- Pyramid principle: Headline → Support → Implications → Actions
- Exec-grade P&L visuals: Waterfall, contribution, concentration, trajectory
- Decision layer: segmentation + ranked actions + scenario planning (directional)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
from pathlib import Path
from datetime import datetime

root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data_loader import load_events, export_to_excel
from src.normalization import normalize_events
from src.builder_pnl import build_builder_pnl, apply_status_bands, compute_paid_share
from src.utils import fmt_currency, fmt_percent, fmt_roas


# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Builder Economics — Executive P&L",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOT_BG = "rgba(0,0,0,0)"
GRID = "#eef2f7"
INK = "#0f172a"
MUTED = "#64748b"

POS = "#16a34a"
NEG = "#dc2626"
NEU = "#334155"

# Streamlit 2026+: replace use_container_width with width
STRETCH = "stretch"

# ──────────────────────────────────────────────────────────────────────────────
# STYLE (clean, exec)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

:root{
  --ink: #0f172a;
  --muted: #64748b;
  --card: #ffffff;
  --line: #e5e7eb;
  --soft: #f8fafc;
}

.exec-header {
  padding: 0.25rem 0 0.75rem 0;
  border-bottom: 1px solid var(--line);
  margin-bottom: 0.75rem;
}
.exec-title {
  font-size: 1.9rem;
  font-weight: 600;
  letter-spacing: -0.02em;
  color: var(--ink);
  margin: 0;
}
.exec-subtitle {
  font-size: 0.9rem;
  color: var(--muted);
  margin: 0.35rem 0 0 0;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 0.75rem;
  margin-top: 0.75rem;
  margin-bottom: 0.5rem;
}
.kpi-card{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 0.85rem 0.9rem;
}
.kpi-label{
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin-bottom: 0.35rem;
}
.kpi-value{
  font-size: 1.2rem;
  font-weight: 600;
  color: var(--ink);
  line-height: 1.15;
}
.kpi-sub{
  margin-top: 0.3rem;
  font-size: 0.78rem;
  color: var(--muted);
}

.story-card{
  background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 1rem 1.1rem;
  margin: 0.75rem 0 0.5rem 0;
}
.story-title{
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.10em;
  color: var(--muted);
  margin: 0 0 0.5rem 0;
}
.story-headline{
  font-size: 1.15rem;
  font-weight: 650;
  color: var(--ink);
  margin: 0 0 0.5rem 0;
}
.story-bullets{
  margin: 0;
  color: #111827;
  font-size: 0.92rem;
  line-height: 1.45;
}
.story-bullets li { margin: 0.2rem 0; }

.callout {
  background: #0b1220;
  color: #e5e7eb;
  border-radius: 14px;
  padding: 1rem 1.1rem;
  margin: 0.75rem 0;
}
.callout .label{
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.10em;
  color: #94a3b8;
  margin-bottom: 0.4rem;
}
.callout .text{
  font-size: 0.95rem;
  line-height: 1.5;
  color: #e5e7eb;
}

.section-h{
  margin: 0.75rem 0 0.25rem 0;
  font-weight: 650;
  color: var(--ink);
  letter-spacing: -0.01em;
}
.section-sub{
  margin: 0 0 0.5rem 0;
  color: var(--muted);
  font-size: 0.9rem;
}

.small-muted { color: var(--muted); font-size: 0.85rem; }

</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_data():
    if "events_file" not in st.session_state:
        return None
    events = load_events(st.session_state["events_file"])
    return normalize_events(events) if events is not None else None


@st.cache_data(show_spinner=False)
def cached_build_pnl(events: pd.DataFrame, lens: str, date_basis: str, freq: str) -> pd.DataFrame:
    return build_builder_pnl(events, lens=lens, date_basis=date_basis, freq=freq)


def _month_col(date_basis: str) -> str:
    return "lead_month_start" if date_basis == "lead_date" else "ref_month_start"


def _week_col(date_basis: str) -> str:
    # best-effort: if your pipeline has week starts, use them; otherwise derive from RefDate/lead_date later
    return "lead_week_start" if date_basis == "lead_date" else "ref_week_start"


# ──────────────────────────────────────────────────────────────────────────────
# METRICS / NARRATIVE
# ──────────────────────────────────────────────────────────────────────────────
def portfolio_metrics(pnl_snapshot: pd.DataFrame):
    total_rev = float(pnl_snapshot["Revenue"].sum())
    total_cost = float(pnl_snapshot["MediaCost"].sum())
    total_profit = float(pnl_snapshot["Profit"].sum())
    roas = (total_rev / total_cost) if total_cost > 0 else 0.0
    margin = (total_profit / total_rev) if total_rev > 0 else 0.0

    builders = len(pnl_snapshot)
    profitable = int((pnl_snapshot["Profit"] > 0).sum())
    profitable_pct = (profitable / builders) if builders else 0.0

    top5_profit = float(pnl_snapshot.nlargest(5, "Profit")["Profit"].sum()) if builders else 0.0
    concentration = (top5_profit / total_profit) if total_profit > 0 else 0.0

    loss_drag = float(pnl_snapshot.loc[pnl_snapshot["Profit"] < 0, "Profit"].sum())

    return {
        "rev": total_rev,
        "cost": total_cost,
        "profit": total_profit,
        "roas": roas,
        "margin": margin,
        "builders": builders,
        "profitable": profitable,
        "profitable_pct": profitable_pct,
        "top5_profit": top5_profit,
        "concentration": concentration,
        "loss_drag": loss_drag,
    }


def safe_pct(x):
    try:
        if x is None or np.isnan(x):
            return 0.0
    except Exception:
        pass
    return float(x)


def build_storyline(m, delta=None):
    """
    Pyramid principle narrative:
    - Headline
    - 3 supporting bullets (performance, drivers, risk, actions)
    """
    profit = m["profit"]
    rev = m["rev"]
    margin = m["margin"]
    roas = m["roas"]
    prof_pct = m["profitable_pct"]
    conc = m["concentration"]
    loss_drag = m["loss_drag"]

    # Headline classification
    if margin >= 0.25 and prof_pct >= 0.70:
        headline = f"Portfolio is strong: {fmt_currency(profit)} profit on {fmt_currency(rev)} revenue ({margin:.1%} margin), ROAS {roas:.2f}x."
    elif margin >= 0.10 and prof_pct >= 0.50:
        headline = f"Portfolio is mixed: {fmt_currency(profit)} profit on {fmt_currency(rev)} revenue ({margin:.1%} margin), ROAS {roas:.2f}x."
    else:
        headline = f"Portfolio is under pressure: {fmt_currency(profit)} profit on {fmt_currency(rev)} revenue ({margin:.1%} margin), ROAS {roas:.2f}x."

    bullets = []
    bullets.append(
        f"**Profitability breadth:** {m['profitable']}/{m['builders']} builders profitable ({prof_pct:.0%})."
    )
    if conc >= 0.60:
        bullets.append(f"**Concentration risk:** Top 5 builders drive **{conc:.0%}** of profit (single-point exposure).")
    else:
        bullets.append(f"**Concentration:** Top 5 builders drive {conc:.0%} of profit (moderate).")

    if loss_drag < 0:
        bullets.append(f"**Loss-maker drag:** Loss-making builders reduce profit by **{fmt_currency(abs(loss_drag))}**.")

    if delta is not None:
        d_profit = delta.get("profit", 0.0)
        d_rev = delta.get("rev", 0.0)
        d_roas = delta.get("roas", 0.0)
        sign_p = "+" if d_profit >= 0 else ""
        sign_r = "+" if d_rev >= 0 else ""
        sign_ro = "+" if d_roas >= 0 else ""
        bullets.append(
            f"**Change vs prior period:** Profit {sign_p}{fmt_currency(d_profit)}, Revenue {sign_r}{fmt_currency(d_rev)}, ROAS {sign_ro}{d_roas:.2f}x (directional)."
        )

    return headline, bullets


# ──────────────────────────────────────────────────────────────────────────────
# CHARTS
# ──────────────────────────────────────────────────────────────────────────────
def fig_pnl_waterfall(m):
    # Revenue (positive) -> Media (negative) -> Profit (total)
    vals = [m["rev"], -m["cost"], m["profit"]]
    cats = ["Revenue", "Media Cost", "Gross Profit"]

    fig = go.Figure(
        go.Waterfall(
            x=cats,
            y=vals,
            measure=["relative", "relative", "total"],
            connector={"line": {"color": GRID, "width": 1}},
            increasing={"marker": {"color": POS}},
            decreasing={"marker": {"color": NEG}},
            totals={"marker": {"color": INK}},
            text=[fmt_currency(v) for v in vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        yaxis=dict(gridcolor=GRID, tickformat="$,.0f", title=""),
        xaxis=dict(tickfont=dict(color=NEU)),
        showlegend=False,
    )
    return fig


def fig_profit_bridge_by_segment(pnl_snapshot: pd.DataFrame):
    # Top 5 + Other profitable + Loss makers + Total
    df = pnl_snapshot.copy()
    top = df[df["Profit"] > 0].nlargest(5, "Profit")
    other_prof = df[(df["Profit"] > 0) & (~df["BuilderRegionKey"].isin(top["BuilderRegionKey"]))]
    losses = df[df["Profit"] < 0]

    cats, vals = [], []

    for _, r in top.iterrows():
        name = r["BuilderRegionKey"]
        name = name[:18] + "…" if len(name) > 19 else name
        cats.append(name)
        vals.append(float(r["Profit"]))

    if len(other_prof) > 0:
        cats.append(f"Other profitable ({len(other_prof)})")
        vals.append(float(other_prof["Profit"].sum()))

    if len(losses) > 0:
        cats.append(f"Loss-making ({len(losses)})")
        vals.append(float(losses["Profit"].sum()))

    cats.append("NET PROFIT")
    vals.append(float(sum(vals)))

    fig = go.Figure(
        go.Waterfall(
            x=cats,
            y=vals,
            measure=["relative"] * (len(cats) - 1) + ["total"],
            connector={"line": {"color": GRID, "width": 1}},
            increasing={"marker": {"color": POS}},
            decreasing={"marker": {"color": NEG}},
            totals={"marker": {"color": INK}},
            text=[f"${v/1000:,.0f}K" if abs(v) >= 1000 else fmt_currency(v) for v in vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=10, b=70),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        yaxis=dict(gridcolor=GRID, tickformat="$,.0f", title=""),
        xaxis=dict(tickangle=-25),
        showlegend=False,
    )
    return fig


def fig_concentration_pareto(pnl_snapshot: pd.DataFrame):
    df = pnl_snapshot.sort_values("Profit", ascending=False).copy()
    if df.empty or df["Profit"].sum() == 0:
        return None, None

    df["cum_profit"] = df["Profit"].cumsum()
    df["cum_pct"] = df["cum_profit"] / df["Profit"].sum()
    df["builder_pct"] = (np.arange(len(df)) + 1) / len(df)

    # where 80% achieved
    idx80 = df.index[df["cum_pct"] >= 0.8]
    pct80 = float(df.loc[idx80[0], "builder_pct"] * 100) if len(idx80) else 100.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["builder_pct"] * 100,
            y=df["cum_pct"] * 100,
            fill="tozeroy",
            line=dict(color=INK, width=2),
            fillcolor="rgba(15, 23, 42, 0.08)",
            hovertemplate="% Builders: %{x:.0f}%<br>Cum Profit: %{y:.0f}%<extra></extra>",
        )
    )
    fig.add_hline(y=80, line_dash="dot", line_color="#94a3b8", line_width=1)
    fig.add_vline(x=pct80, line_dash="dot", line_color="#94a3b8", line_width=1)

    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        xaxis=dict(title="% of Builders", gridcolor=GRID, range=[0, 100]),
        yaxis=dict(title="% of Profit", gridcolor=GRID, range=[0, 105]),
        showlegend=False,
    )
    return fig, pct80


def fig_top_bottom_contributors(pnl_snapshot: pd.DataFrame, n=10):
    df = pnl_snapshot.sort_values("Profit", ascending=False).copy()
    top = df.head(n)
    bot = df.tail(n).sort_values("Profit", ascending=True)

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=(f"Top {n} Profit Contributors", f"Bottom {n} (Loss Drag)")
    )

    fig.add_trace(
        go.Bar(
            y=top["BuilderRegionKey"],
            x=top["Profit"],
            orientation="h",
            marker_color=POS,
            hovertemplate="<b>%{y}</b><br>Profit: %{x:$,.0f}<extra></extra>",
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            y=bot["BuilderRegionKey"],
            x=bot["Profit"],
            orientation="h",
            marker_color=NEG,
            hovertemplate="<b>%{y}</b><br>Profit: %{x:$,.0f}<extra></extra>",
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        xaxis=dict(gridcolor=GRID, tickformat="$,.0f"),
        xaxis2=dict(gridcolor=GRID, tickformat="$,.0f"),
        yaxis=dict(automargin=True),
        yaxis2=dict(automargin=True),
        showlegend=False,
    )
    return fig


def segment_action_map(pnl_snapshot: pd.DataFrame):
    df = pnl_snapshot[(pnl_snapshot["MediaCost"] > 0) & (pnl_snapshot["Revenue"] > 0)].copy()
    if df.empty:
        return None, None, None

    df["Margin"] = np.where(df["Revenue"] > 0, df["Profit"] / df["Revenue"], 0.0)

    roas_med = float(df["ROAS"].median())
    margin_med = float(df["Margin"].median())

    def classify(r):
        if r["ROAS"] >= roas_med and r["Margin"] >= margin_med:
            return "Scale"
        if r["ROAS"] >= roas_med and r["Margin"] < margin_med:
            return "Fix Margin"
        if r["ROAS"] < roas_med and r["Margin"] >= margin_med:
            return "Fix Efficiency"
        return "Review"

    df["Action"] = df.apply(classify, axis=1)

    color_map = {"Scale": POS, "Fix Margin": "#2563eb", "Fix Efficiency": "#f59e0b", "Review": NEG}

    fig = go.Figure()
    for a, c in color_map.items():
        sub = df[df["Action"] == a]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["ROAS"],
                y=sub["Margin"] * 100,
                mode="markers",
                name=a,
                marker=dict(
                    size=8 + np.log1p(sub["MediaCost"]) * 2,
                    color=c,
                    opacity=0.75,
                    line=dict(width=1, color="white"),
                ),
                text=sub["BuilderRegionKey"],
                hovertemplate="<b>%{text}</b><br>ROAS: %{x:.2f}x<br>Margin: %{y:.1f}%<extra></extra>",
            )
        )

    fig.add_hline(y=margin_med * 100, line_dash="dot", line_color="#94a3b8", line_width=1)
    fig.add_vline(x=roas_med, line_dash="dot", line_color="#94a3b8", line_width=1)

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        xaxis=dict(title="ROAS", gridcolor=GRID),
        yaxis=dict(title="Profit Margin (%)", gridcolor=GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    summary = (
        df.groupby("Action")
        .agg(
            Builders=("BuilderRegionKey", "count"),
            Revenue=("Revenue", "sum"),
            MediaCost=("MediaCost", "sum"),
            Profit=("Profit", "sum"),
        )
        .reset_index()
    )

    return fig, summary, df


def fig_trajectory(pnl_ts: pd.DataFrame, freq: str):
    # pnl_ts expected to have period_start column
    if pnl_ts is None or pnl_ts.empty or "period_start" not in pnl_ts.columns:
        return None

    ts = (
        pnl_ts.groupby("period_start")
        .agg(Revenue=("Revenue", "sum"), MediaCost=("MediaCost", "sum"), Profit=("Profit", "sum"))
        .reset_index()
    )
    ts["period_start"] = pd.to_datetime(ts["period_start"])
    ts = ts.sort_values("period_start")
    ts["ROAS"] = np.where(ts["MediaCost"] > 0, ts["Revenue"] / ts["MediaCost"], 0.0)
    ts["Margin"] = np.where(ts["Revenue"] > 0, ts["Profit"] / ts["Revenue"], 0.0)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.12,
    )

    fig.add_trace(
        go.Bar(
            x=ts["period_start"],
            y=ts["Revenue"],
            name="Revenue",
            marker_color="#e2e8f0",
            opacity=0.9,
            hovertemplate="Revenue: %{y:$,.0f}<extra></extra>",
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=ts["period_start"],
            y=ts["Profit"],
            name="Profit",
            mode="lines+markers",
            line=dict(color=POS, width=3),
            marker=dict(size=7),
            hovertemplate="Profit: %{y:$,.0f}<extra></extra>",
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=ts["period_start"],
            y=ts["Margin"] * 100,
            name="Margin %",
            mode="lines",
            line=dict(color="#2563eb", width=2),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.08)",
            hovertemplate="Margin: %{y:.1f}%<extra></extra>",
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_yaxes(gridcolor=GRID, row=1, col=1, tickformat="$,.0f")
    fig.update_yaxes(gridcolor=GRID, row=2, col=1, ticksuffix="%", rangemode="tozero")
    fig.update_xaxes(gridcolor=GRID)

    return fig


def reallocation_scenario(df_actions: pd.DataFrame, share_move: float):
    """
    Directional scenario:
    move X% of spend from Review -> Scale, assume revenue scales linearly with ROAS.
    Incremental profit ≈ moved_spend * (ROAS_scale - ROAS_review)
    """
    if df_actions is None or df_actions.empty:
        return None

    scale = df_actions[df_actions["Action"] == "Scale"].copy()
    review = df_actions[df_actions["Action"] == "Review"].copy()

    if scale.empty or review.empty:
        return {"moved_spend": 0.0, "delta_rev": 0.0, "delta_profit": 0.0, "note": "Not enough Scale/Review builders to run scenario."}

    spend_review = float(review["MediaCost"].sum())
    spend_scale = float(scale["MediaCost"].sum())

    moved = spend_review * share_move

    roas_scale = float(np.average(scale["ROAS"], weights=np.maximum(scale["MediaCost"], 1.0)))
    roas_review = float(np.average(review["ROAS"], weights=np.maximum(review["MediaCost"], 1.0)))

    # Directional incremental revenue/profit
    delta_rev = moved * (roas_scale - roas_review)
    delta_profit = moved * (roas_scale - roas_review)  # since media spend moved, net effect approximates revenue delta

    return {
        "moved_spend": moved,
        "delta_rev": delta_rev,
        "delta_profit": delta_profit,
        "note": f"Directional: assumes revenue scales linearly with spend at observed ROAS (use as a prioritisation signal, not a forecast).",
        "roas_scale": roas_scale,
        "roas_review": roas_review,
        "spend_review": spend_review,
        "spend_scale": spend_scale,
    }


def rank_action_list(df_actions: pd.DataFrame):
    """
    Create a ranked action list: where to scale / fix / review first.
    Score is intentionally simple + interpretable for execs.
    """
    df = df_actions.copy()

    # Normalize helpers
    spend = np.maximum(df["MediaCost"].astype(float), 0.0)
    profit = df["Profit"].astype(float)
    roas = np.maximum(df["ROAS"].astype(float), 0.0)
    margin = np.maximum(df["Margin"].astype(float), -2.0)

    # Priority score by action
    # - Review: prioritize biggest losses and biggest spend
    # - Scale: prioritize biggest profit and biggest spend (capacity to absorb budget)
    # - Fix: prioritize biggest spend + gap to median
    roas_med = float(df["ROAS"].median())
    margin_med = float(df["Margin"].median())

    score = np.zeros(len(df), dtype=float)
    for i, a in enumerate(df["Action"].values):
        if a == "Review":
            score[i] = (np.maximum(-profit.iloc[i], 0.0) * 0.7) + (spend.iloc[i] * 0.3)
        elif a == "Scale":
            score[i] = (np.maximum(profit.iloc[i], 0.0) * 0.6) + (spend.iloc[i] * 0.4)
        elif a == "Fix Margin":
            score[i] = (spend.iloc[i] * 0.6) + (np.maximum(margin_med - margin.iloc[i], 0.0) * 10000 * 0.4)
        else:  # Fix Efficiency
            score[i] = (spend.iloc[i] * 0.6) + (np.maximum(roas_med - roas.iloc[i], 0.0) * 10000 * 0.4)

    df["PriorityScore"] = score
    df["NowNext"] = df["Action"].map({
        "Scale": "Scale (add budget)",
        "Fix Margin": "Fix margin (pricing/cost)",
        "Fix Efficiency": "Fix efficiency (media mix)",
        "Review": "Review (stop/reshape)",
    })

    cols = [
        "BuilderRegionKey", "NowNext", "Revenue", "MediaCost", "Profit", "ROAS", "Margin", "PriorityScore"
    ]
    out = df[cols].sort_values("PriorityScore", ascending=False).copy()
    out.rename(columns={"BuilderRegionKey": "Builder"}, inplace=True)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    events = load_data()

    if events is None:
        st.warning("⚠️ Please upload events data on the Home page first.")
        st.page_link("app.py", label="← Go to Home", icon="🏠")
        return

    # ──────────────────────────────────────────────────────────────────────────
    # SIDEBAR — tight + decision-oriented
    # ──────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Controls")

        lens = st.selectbox(
            "Attribution lens",
            ["recipient", "payer", "origin"],
            format_func=lambda x: {"recipient": "Recipient", "payer": "Payer", "origin": "Origin"}[x],
        )

        date_basis = st.selectbox(
            "Date basis",
            ["lead_date", "RefDate"],
            format_func=lambda x: "Lead Date" if x == "lead_date" else "Referral Date",
        )

        freq = st.selectbox(
            "Trajectory granularity",
            ["ALL", "M", "W"],
            format_func=lambda x: {"ALL": "All time (no trajectory)", "M": "Monthly", "W": "Weekly"}[x],
        )

        st.markdown("---")
        st.markdown("#### Snapshot period")

        month_filter = None
        week_filter = None

        mcol = _month_col(date_basis)

        # default month = latest available
        if mcol in events.columns:
            months = sorted(pd.to_datetime(events[mcol].dropna().unique()))
        else:
            months = []

        if freq in ("M", "W") and months:
            month_labels = [pd.Timestamp(m).strftime("%Y-%m") for m in months]
            default_ix = len(month_labels) - 1
            sel = st.selectbox("Month", month_labels, index=default_ix)
            month_filter = pd.Timestamp(sel + "-01")

            if freq == "W":
                # Try to support a week_start column; if absent, derive from RefDate/lead_date
                wcol = _week_col(date_basis)
                if wcol in events.columns:
                    weeks = sorted(pd.to_datetime(events.loc[events[mcol] == month_filter, wcol].dropna().unique()))
                else:
                    # derive from actual date column
                    date_col = "lead_date" if date_basis == "lead_date" else "RefDate"
                    tmp = events.copy()
                    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
                    tmp = tmp[tmp[mcol] == month_filter]
                    weeks = sorted(tmp[date_col].dropna().dt.to_period("W").dt.start_time.unique())

                if weeks:
                    week_labels = [pd.Timestamp(w).strftime("%Y-%m-%d") for w in weeks]
                    default_w = len(week_labels) - 1
                    sel_w = st.selectbox("Week (start)", week_labels, index=default_w)
                    week_filter = pd.Timestamp(sel_w)

        st.markdown("---")
        st.markdown("#### Filters")
        min_revenue = st.number_input("Min Revenue (builder)", value=0, step=5000)
        min_media = st.number_input("Min Media (builder)", value=0, step=5000)

        st.markdown("---")
        st.markdown("#### Decision levers")
        move_share = st.slider("Reallocate spend: Review → Scale", 0, 60, 20, 5) / 100.0
        show_definitions = st.checkbox("Show metric definitions", value=False)

    # ──────────────────────────────────────────────────────────────────────────
    # Build snapshot events (for “point-in-time P&L”)
    # ──────────────────────────────────────────────────────────────────────────
    snapshot_events = events.copy()

    if month_filter is not None and mcol in snapshot_events.columns:
        snapshot_events = snapshot_events[pd.to_datetime(snapshot_events[mcol]) == pd.Timestamp(month_filter)]

    if week_filter is not None:
        # Apply week filter best-effort
        wcol = _week_col(date_basis)
        if wcol in snapshot_events.columns:
            snapshot_events = snapshot_events[pd.to_datetime(snapshot_events[wcol]) == pd.Timestamp(week_filter)]
        else:
            date_col = "lead_date" if date_basis == "lead_date" else "RefDate"
            snapshot_events[date_col] = pd.to_datetime(snapshot_events[date_col], errors="coerce")
            snapshot_events = snapshot_events[
                snapshot_events[date_col].dt.to_period("W").dt.start_time == pd.Timestamp(week_filter)
            ]

    if snapshot_events.empty:
        st.warning("No data for the selected snapshot period / filters.")
        return

    # ──────────────────────────────────────────────────────────────────────────
    # Snapshot P&L (aggregated to builder for exec table)
    # ──────────────────────────────────────────────────────────────────────────
    try:
        pnl_snapshot = cached_build_pnl(snapshot_events, lens=lens, date_basis=date_basis, freq="ALL")
    except Exception as e:
        st.error(f"Error building snapshot P&L: {e}")
        return

    if pnl_snapshot is None or pnl_snapshot.empty:
        st.warning("No P&L data available for snapshot.")
        return

    pnl_snapshot = apply_status_bands(pnl_snapshot)
    pnl_snapshot = compute_paid_share(snapshot_events, pnl_snapshot, lens=lens)

    # Apply filters (builder-level)
    if min_revenue > 0:
        pnl_snapshot = pnl_snapshot[pnl_snapshot["Revenue"] >= min_revenue]
    if min_media > 0:
        pnl_snapshot = pnl_snapshot[pnl_snapshot["MediaCost"] >= min_media]

    pnl_snapshot = pnl_snapshot[~((pnl_snapshot["Revenue"] == 0) & (pnl_snapshot["MediaCost"] == 0))]

    if pnl_snapshot.empty:
        st.warning("No builders match the filters.")
        return

    # ──────────────────────────────────────────────────────────────────────────
    # Trajectory P&L (portfolio trend) if freq enabled
    # ──────────────────────────────────────────────────────────────────────────
    pnl_ts = None
    if freq in ("M", "W"):
        try:
            pnl_ts = cached_build_pnl(events, lens=lens, date_basis=date_basis, freq=freq)
            pnl_ts = apply_status_bands(pnl_ts) if pnl_ts is not None and not pnl_ts.empty else pnl_ts
        except Exception:
            pnl_ts = None

    # ──────────────────────────────────────────────────────────────────────────
    # Prior period delta (directional) for snapshot
    # ──────────────────────────────────────────────────────────────────────────
    delta = None
    if freq in ("M", "W") and month_filter is not None:
        prior_events = events.copy()

        if freq == "M":
            prior_month = (pd.Timestamp(month_filter) - pd.offsets.MonthBegin(1))
            if mcol in prior_events.columns:
                prior_events = prior_events[pd.to_datetime(prior_events[mcol]) == pd.Timestamp(prior_month)]
        else:
            # weekly: prior week relative to selected week
            if week_filter is not None:
                prior_week = pd.Timestamp(week_filter) - pd.Timedelta(days=7)
                wcol = _week_col(date_basis)
                if wcol in prior_events.columns:
                    prior_events = prior_events[pd.to_datetime(prior_events[wcol]) == pd.Timestamp(prior_week)]
                else:
                    date_col = "lead_date" if date_basis == "lead_date" else "RefDate"
                    prior_events[date_col] = pd.to_datetime(prior_events[date_col], errors="coerce")
                    prior_events = prior_events[
                        prior_events[date_col].dt.to_period("W").dt.start_time == pd.Timestamp(prior_week)
                    ]

        if not prior_events.empty:
            try:
                pnl_prior = cached_build_pnl(prior_events, lens=lens, date_basis=date_basis, freq="ALL")
                pnl_prior = apply_status_bands(pnl_prior)
                m_now = portfolio_metrics(pnl_snapshot)
                m_prev = portfolio_metrics(pnl_prior)
                delta = {
                    "profit": m_now["profit"] - m_prev["profit"],
                    "rev": m_now["rev"] - m_prev["rev"],
                    "roas": m_now["roas"] - m_prev["roas"],
                }
            except Exception:
                delta = None

    # ──────────────────────────────────────────────────────────────────────────
    # Header
    # ──────────────────────────────────────────────────────────────────────────
    if freq == "ALL":
        period_label = "All time"
    else:
        if freq == "M":
            period_label = month_filter.strftime("%B %Y") if month_filter is not None else "Monthly"
        else:
            period_label = f"Week of {week_filter.strftime('%Y-%m-%d')}" if week_filter is not None else "Weekly"

    st.markdown(
        f"""
<div class="exec-header">
  <h1 class="exec-title">Builder Economics — Executive P&L Review</h1>
  <p class="exec-subtitle">{lens.title()} attribution • {period_label} • {len(pnl_snapshot)} builders in snapshot</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Core metrics + storyline (pyramid)
    # ──────────────────────────────────────────────────────────────────────────
    m = portfolio_metrics(pnl_snapshot)
    headline, bullets = build_storyline(m, delta=delta)

    # KPI row
    def kpi(label, value, sub=""):
        return f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """

    st.markdown(
        f"""
<div class="kpi-grid">
  {kpi("Revenue", fmt_currency(m["rev"]), "Gross revenue attributed")}
  {kpi("Media Cost", fmt_currency(m["cost"]), "Spend allocated")}
  {kpi("Gross Profit", fmt_currency(m["profit"]), "Revenue – Media cost")}
  {kpi("Margin", f"{m['margin']:.1%}", "Profit / Revenue")}
  {kpi("ROAS", f"{m['roas']:.2f}x", "Revenue / Media")}
  {kpi("Profitable", f"{m['profitable_pct']:.0%}", f"{m['profitable']}/{m['builders']} builders")}
</div>
""",
        unsafe_allow_html=True,
    )

    # Storyline card (Headline + bullets)
    bullet_html = "".join([f"<li>{b}</li>" for b in bullets])
    st.markdown(
        f"""
<div class="story-card">
  <div class="story-title">Key message (pyramid)</div>
  <div class="story-headline">{headline}</div>
  <ul class="story-bullets">{bullet_html}</ul>
</div>
""",
        unsafe_allow_html=True,
    )

    # Tabs for top-down journey
    tab_exec, tab_drivers, tab_actions, tab_detail, tab_data = st.tabs(
        ["01 Executive", "02 Drivers", "03 Actions", "04 Builder Detail", "05 Data"]
    )

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 01 — EXECUTIVE
    # ──────────────────────────────────────────────────────────────────────────
    with tab_exec:
        st.markdown('<div class="section-h">Executive view</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">P&L framing, composition, and trajectory (if available).</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([1.1, 1.4], vertical_alignment="top")
        with c1:
            st.markdown("**P&L Waterfall (Revenue → Media → Profit)**")
            st.plotly_chart(fig_pnl_waterfall(m), width=STRETCH)

            # Concentration quick view
            figc, pct80 = fig_concentration_pareto(pnl_snapshot)
            if figc is not None:
                st.markdown("**Concentration (Pareto)**")
                st.plotly_chart(figc, width=STRETCH)
                st.caption(f"~{pct80:.0f}% of builders generate 80% of profit (snapshot).")

        with c2:
            st.markdown("**Profit composition (Top performers vs drag)**")
            st.plotly_chart(fig_profit_bridge_by_segment(pnl_snapshot), width=STRETCH)

            if pnl_ts is not None and not pnl_ts.empty and "period_start" in pnl_ts.columns:
                st.markdown("**Trajectory (portfolio)**")
                figt = fig_trajectory(pnl_ts, freq=freq)
                if figt is not None:
                    st.plotly_chart(figt, width=STRETCH)
                else:
                    st.info("Trajectory not available for this selection.")

        # Executive “So what / Now what”
        st.markdown(
            f"""
<div class="callout">
  <div class="label">So what / Now what</div>
  <div class="text">
    <b>So what:</b> Profit is driven by a small set of builders; underperformers create material drag.<br/>
    <b>Now what:</b> Scale the “Scale” quadrant, fix unit economics for “Fix Margin/Efficiency”, and run hard stop/reshape decisions for “Review”.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 02 — DRIVERS
    # ──────────────────────────────────────────────────────────────────────────
    with tab_drivers:
        st.markdown('<div class="section-h">Drivers</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Where profit is created vs destroyed — contribution, drag, and exposure.</div>', unsafe_allow_html=True)

        st.plotly_chart(fig_top_bottom_contributors(pnl_snapshot, n=12), width=STRETCH)

        # Loss-maker deep list
        losses = pnl_snapshot[pnl_snapshot["Profit"] < 0].sort_values("Profit").copy()
        if not losses.empty:
            st.markdown("**Largest loss-makers (focus list)**")
            view = losses[["BuilderRegionKey", "Revenue", "MediaCost", "Profit", "ROAS", "Margin_pct"]].head(20).copy()
            view.rename(
                columns={
                    "BuilderRegionKey": "Builder",
                    "MediaCost": "Media Cost",
                    "Margin_pct": "Margin",
                },
                inplace=True,
            )

            st.dataframe(
                view.style.format(
                    {
                        "Revenue": "${:,.0f}",
                        "Media Cost": "${:,.0f}",
                        "Profit": "${:,.0f}",
                        "ROAS": "{:.2f}x",
                        "Margin": "{:.1%}",
                    }
                ),
                hide_index=True,
                width=STRETCH,
                height=420,
            )
        else:
            st.success("No loss-makers in the filtered snapshot.")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 03 — ACTIONS
    # ──────────────────────────────────────────────────────────────────────────
    with tab_actions:
        st.markdown('<div class="section-h">Actions</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Segment builders by efficiency & margin, then translate into a ranked operating agenda.</div>', unsafe_allow_html=True)

        fig_scatter, action_summary, df_actions = segment_action_map(pnl_snapshot)

        if fig_scatter is not None:
            c1, c2 = st.columns([1.45, 1.0], vertical_alignment="top")
            with c1:
                st.markdown("**Builder action map (ROAS × Margin)**")
                st.plotly_chart(fig_scatter, width=STRETCH)
                st.caption("Bubble size = media spend. Quadrants imply default action. Validate exceptions with context.")

            with c2:
                st.markdown("**Quadrant summary**")
                s = action_summary.copy()
                s["Revenue"] = s["Revenue"].map(lambda x: fmt_currency(x))
                s["MediaCost"] = s["MediaCost"].map(lambda x: fmt_currency(x))
                s["Profit"] = s["Profit"].map(lambda x: fmt_currency(x))
                st.dataframe(s, hide_index=True, width=STRETCH, height=220)

                # Scenario
                scen = reallocation_scenario(df_actions, move_share)
                st.markdown("**Reallocation scenario (directional)**")
                if scen is not None:
                    st.markdown(
                        f"""
                        <div class="story-card">
                          <div class="story-title">If we move {int(move_share*100)}% of Review spend</div>
                          <div class="story-headline">Move {fmt_currency(scen["moved_spend"])} from Review → Scale</div>
                          <ul class="story-bullets">
                            <li>Weighted ROAS: Scale {scen.get("roas_scale", 0):.2f}x vs Review {scen.get("roas_review", 0):.2f}x</li>
                            <li>Directional uplift: Revenue {fmt_currency(scen["delta_rev"])} • Profit {fmt_currency(scen["delta_profit"])}</li>
                            <li><span class="small-muted">{scen["note"]}</span></li>
                          </ul>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        else:
            st.info("Not enough Revenue/Media observations to segment actions.")

        # Ranked action list
        if df_actions is not None and not df_actions.empty:
            st.markdown("**Ranked action list (where to focus first)**")
            ranked = rank_action_list(df_actions).head(25)

            show = ranked.copy()
            show["Revenue"] = show["Revenue"].astype(float)
            show["MediaCost"] = show["MediaCost"].astype(float)
            show["Profit"] = show["Profit"].astype(float)
            show["Margin"] = show["Margin"].astype(float)

            st.dataframe(
                show.style.format(
                    {
                        "Revenue": "${:,.0f}",
                        "MediaCost": "${:,.0f}",
                        "Profit": "${:,.0f}",
                        "ROAS": "{:.2f}x",
                        "Margin": "{:.1%}",
                        "PriorityScore": "{:,.0f}",
                    }
                ),
                hide_index=True,
                width=STRETCH,
                height=520,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 04 — BUILDER DETAIL
    # ──────────────────────────────────────────────────────────────────────────
    with tab_detail:
        st.markdown('<div class="section-h">Builder detail</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Drill into a single builder’s unit economics and (if available) time series.</div>', unsafe_allow_html=True)

        builders = sorted(pnl_snapshot["BuilderRegionKey"].dropna().unique().tolist())
        if not builders:
            st.info("No builders available.")
        else:
            left, right = st.columns([1.2, 1.0], vertical_alignment="top")
            with left:
                selected_builder = st.selectbox("Select builder", builders, index=0)
            with right:
                sort_metric = st.selectbox("Compare peers by", ["Profit", "Revenue", "MediaCost", "ROAS"], index=0)

            b = pnl_snapshot[pnl_snapshot["BuilderRegionKey"] == selected_builder].copy()
            if not b.empty:
                row = b.iloc[0]
                b_rev = float(row["Revenue"])
                b_cost = float(row["MediaCost"])
                b_profit = float(row["Profit"])
                b_roas = float(row.get("ROAS", 0.0))
                b_margin = float(row.get("Margin_pct", 0.0))
                b_paid = float(row.get("PaidShare", np.nan))

                st.markdown(
                    f"""
<div class="story-card">
  <div class="story-title">Builder snapshot</div>
  <div class="story-headline">{selected_builder}</div>
  <ul class="story-bullets">
    <li>Revenue {fmt_currency(b_rev)} • Media {fmt_currency(b_cost)} • Profit {fmt_currency(b_profit)}</li>
    <li>Margin {b_margin:.1%} • ROAS {b_roas:.2f}x • Paid share {("" if np.isnan(b_paid) else f"{b_paid:.0%}")}</li>
  </ul>
</div>
""",
                    unsafe_allow_html=True,
                )

            # Peer compare
            peer = pnl_snapshot.sort_values(sort_metric, ascending=False).head(15).copy()
            peer["Rank"] = np.arange(1, len(peer) + 1)
            peer_view = peer[["Rank", "BuilderRegionKey", "Revenue", "MediaCost", "Profit", "ROAS", "Margin_pct"]].copy()
            peer_view.rename(columns={"BuilderRegionKey": "Builder", "MediaCost": "Media Cost", "Margin_pct": "Margin"}, inplace=True)

            st.markdown("**Top peers (snapshot)**")
            st.dataframe(
                peer_view.style.format(
                    {"Revenue": "${:,.0f}", "Media Cost": "${:,.0f}", "Profit": "${:,.0f}", "ROAS": "{:.2f}x", "Margin": "{:.1%}"}
                ),
                hide_index=True,
                width=STRETCH,
                height=460,
            )

            # Builder trajectory if available
            if pnl_ts is not None and not pnl_ts.empty and "period_start" in pnl_ts.columns:
                tsb = pnl_ts[pnl_ts["BuilderRegionKey"] == selected_builder].copy()
                if not tsb.empty:
                    tsb["period_start"] = pd.to_datetime(tsb["period_start"])
                    tsb = tsb.sort_values("period_start")
                    tsb_agg = tsb.groupby("period_start").agg(
                        Revenue=("Revenue", "sum"),
                        MediaCost=("MediaCost", "sum"),
                        Profit=("Profit", "sum"),
                    ).reset_index()
                    tsb_agg["ROAS"] = np.where(tsb_agg["MediaCost"] > 0, tsb_agg["Revenue"] / tsb_agg["MediaCost"], 0.0)
                    tsb_agg["Margin"] = np.where(tsb_agg["Revenue"] > 0, tsb_agg["Profit"] / tsb_agg["Revenue"], 0.0)

                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.12)
                    fig.add_trace(go.Bar(x=tsb_agg["period_start"], y=tsb_agg["Revenue"], name="Revenue", marker_color="#e2e8f0"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=tsb_agg["period_start"], y=tsb_agg["Profit"], name="Profit", mode="lines+markers",
                                             line=dict(color=POS, width=3), marker=dict(size=7)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=tsb_agg["period_start"], y=tsb_agg["Margin"] * 100, name="Margin %",
                                             mode="lines", line=dict(color="#2563eb", width=2), fill="tozeroy",
                                             fillcolor="rgba(37, 99, 235, 0.08)"), row=2, col=1)
                    fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=40), paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG, hovermode="x unified")
                    fig.update_yaxes(gridcolor=GRID, row=1, col=1, tickformat="$,.0f")
                    fig.update_yaxes(gridcolor=GRID, row=2, col=1, ticksuffix="%")
                    fig.update_xaxes(gridcolor=GRID)

                    st.markdown("**Builder trajectory**")
                    st.plotly_chart(fig, width=STRETCH)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 05 — DATA (search + export)
    # ──────────────────────────────────────────────────────────────────────────
    with tab_data:
        st.markdown('<div class="section-h">Data</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Search, sort, export. Use this as the “appendix” behind the story.</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns([2.0, 1.0, 1.0], vertical_alignment="top")
        with c1:
            q = st.text_input("Search builders", placeholder="Type to filter…", label_visibility="collapsed")
        with c2:
            sort_col = st.selectbox("Sort by", ["Profit", "Revenue", "MediaCost", "ROAS"], index=0)
        with c3:
            asc = st.checkbox("Ascending", value=False)

        display = pnl_snapshot.copy()
        if q:
            display = display[display["BuilderRegionKey"].str.lower().str.contains(q.lower(), na=False)]

        display = display.sort_values(sort_col, ascending=asc)

        cols = ["BuilderRegionKey", "Revenue", "MediaCost", "Profit", "ROAS", "Margin_pct"]
        if "PaidShare" in display.columns:
            cols.append("PaidShare")
        if "Status" in display.columns:
            cols.append("Status")

        table = display[cols].copy()
        table.rename(
            columns={
                "BuilderRegionKey": "Builder",
                "MediaCost": "Media Cost",
                "Margin_pct": "Margin",
                "PaidShare": "Paid Share",
            },
            inplace=True,
        )

        fmt = {
            "Revenue": "${:,.0f}",
            "Media Cost": "${:,.0f}",
            "Profit": "${:,.0f}",
            "ROAS": "{:.2f}x",
            "Margin": "{:.1%}",
        }
        if "Paid Share" in table.columns:
            fmt["Paid Share"] = "{:.0%}"

        st.dataframe(
            table.style.format(fmt),
            hide_index=True,
            width=STRETCH,
            height=560,
        )

        d1, d2 = st.columns([1, 1], vertical_alignment="top")
        with d1:
            st.download_button("Export Excel", export_to_excel(display, "builder_pnl.xlsx"), "builder_pnl.xlsx")
        with d2:
            st.download_button("Export CSV", display.to_csv(index=False), "builder_pnl.csv", "text/csv")

    # ──────────────────────────────────────────────────────────────────────────
    # DEFINITIONS (optional)
    # ──────────────────────────────────────────────────────────────────────────
    if show_definitions:
        with st.expander("Metric definitions & caveats", expanded=True):
            st.markdown(
                """
- **Revenue**: attributed revenue under the selected lens (recipient / payer / origin).
- **Media Cost**: attributed media spend under the same lens.
- **Gross Profit**: Revenue − Media Cost (this page treats media as the primary variable cost).
- **Margin**: Profit / Revenue.
- **ROAS**: Revenue / Media Cost.
- **Concentration**: share of profit driven by top builders (exposure risk).
- **Reallocation scenario**: **directional** prioritisation signal; assumes linear scaling of revenue with spend at observed ROAS.
"""
            )

    # Footer
    st.caption(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} • Snapshot: {period_label} • Lens: {lens} • Date basis: {date_basis}")


if __name__ == "__main__":
    main()
