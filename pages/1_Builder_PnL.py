"""
Builder P&L Dashboard - Streamlit Page
Filename: pages/1_Builder_PnL.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
from pathlib import Path

# Add parent directory to path for imports
root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data_loader import load_events, export_to_excel
from src.normalization import normalize_events
from src.builder_pnl import build_builder_pnl, apply_status_bands, compute_paid_share
from src.utils import fmt_currency, fmt_percent, fmt_roas, get_status_color

st.set_page_config(page_title="Builder P&L", page_icon="📊", layout="wide")

st.title("📊 Builder P&L Dashboard")
st.markdown("Analyze builder economics across multiple lenses and time periods.")


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
    
    # Store in session
    st.session_state['events_df'] = events
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        lens = st.selectbox(
            "Attribution Lens",
            options=["recipient", "payer", "origin"],
            format_func=lambda x: {
                "recipient": "📥 Recipient (who gets the lead)",
                "payer": "💰 Payer (who funds media)",
                "origin": "🎯 Origin (original lead source)"
            }[x]
        )
        
        date_basis = st.radio(
            "Date Basis",
            options=["lead_date", "RefDate"],
            format_func=lambda x: "Lead Date" if x == "lead_date" else "Referral Date"
        )
        
        freq = st.radio(
            "Time Grain",
            options=["ALL", "M", "W"],
            format_func=lambda x: {"ALL": "All Time", "M": "Monthly", "W": "Weekly"}[x]
        )
        
        # Month filter (if monthly/weekly)
        month_filter = None
        if freq in ("M", "W"):
            month_col = "lead_month_start" if date_basis == "lead_date" else "ref_month_start"
            if month_col in events.columns:
                months = sorted(events[month_col].dropna().unique())
                if months:
                    month_options = ["All"] + [m.strftime("%Y-%m") for m in months]
                    selected_month = st.selectbox("Filter Month", month_options)
                    if selected_month != "All":
                        month_filter = pd.Timestamp(selected_month + "-01")
        
        st.divider()
        
        # Filters
        st.subheader("Filters")
        min_revenue = st.number_input("Min Revenue ($)", value=0, step=1000)
        min_media = st.number_input("Min Media Spend ($)", value=0, step=1000)
        include_zero = st.checkbox("Include zero spend/revenue", value=False)
        
        # Builder search
        builder_search = st.text_input("Search Builder", placeholder="Type to filter...")
        
        top_n = st.slider("Top N Builders", min_value=5, max_value=100, value=25, step=5)
        
        sort_by = st.selectbox(
            "Sort By",
            options=["Profit", "Revenue", "MediaCost", "ROAS"],
            format_func=lambda x: {
                "Profit": "Gross Profit ↓",
                "Revenue": "Revenue ↓",
                "MediaCost": "Media Cost ↓",
                "ROAS": "ROAS ↓"
            }[x]
        )
    
    # Apply month filter
    filtered_events = events.copy()
    if month_filter is not None:
        month_col = "lead_month_start" if date_basis == "lead_date" else "ref_month_start"
        filtered_events = filtered_events[filtered_events[month_col] == month_filter]
    
    if filtered_events.empty:
        st.warning("No data for selected filters.")
        return
    
    # Build P&L
    try:
        pnl = build_builder_pnl(filtered_events, lens=lens, date_basis=date_basis, freq=freq)
    except Exception as e:
        st.error(f"Error building P&L: {e}")
        return
    
    if pnl.empty:
        st.warning("No P&L data for selected parameters.")
        return
    
    # Apply status bands and paid share
    pnl = apply_status_bands(pnl)
    pnl = compute_paid_share(filtered_events, pnl, lens)
    
    # Apply filters
    if not include_zero:
        pnl = pnl[~((pnl["Revenue"] == 0) & (pnl["MediaCost"] == 0))]
    if min_revenue > 0:
        pnl = pnl[pnl["Revenue"] >= min_revenue]
    if min_media > 0:
        pnl = pnl[pnl["MediaCost"] >= min_media]
    if builder_search:
        pnl = pnl[pnl["BuilderRegionKey"].str.lower().str.contains(builder_search.lower(), na=False)]
    
    if pnl.empty:
        st.warning("No builders match the current filters.")
        return
    
    # Overview metrics
    st.header("📈 Portfolio Overview")
    
    total_rev = pnl["Revenue"].sum()
    total_cost = pnl["MediaCost"].sum()
    total_profit = pnl["Profit"].sum()
    overall_roas = total_rev / total_cost if total_cost > 0 else np.nan
    overall_margin = total_profit / total_rev if total_rev > 0 else np.nan
    n_builders = pnl["BuilderRegionKey"].nunique()
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Builders", f"{n_builders:,}")
    col2.metric("Revenue", fmt_currency(total_rev))
    col3.metric("Media Cost", fmt_currency(total_cost))
    col4.metric("Gross Profit", fmt_currency(total_profit))
    col5.metric("ROAS", fmt_roas(overall_roas))
    col6.metric("Margin", fmt_percent(overall_margin))
    
    # Status mix
    st.subheader("Performance Mix")
    
    status_summary = (
        pnl.groupby("Status", dropna=False)
        .agg(
            Builders=("BuilderRegionKey", "nunique"),
            MediaCost=("MediaCost", "sum"),
            Profit=("Profit", "sum")
        )
        .reset_index()
    )
    
    status_summary["MediaShare"] = status_summary["MediaCost"] / status_summary["MediaCost"].sum()
    status_summary["ProfitShare"] = status_summary["Profit"] / status_summary["Profit"].sum()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_status = go.Figure(data=[
            go.Pie(
                labels=status_summary["Status"],
                values=status_summary["MediaCost"],
                hole=0.4,
                marker_colors=[get_status_color(s) for s in status_summary["Status"]]
            )
        ])
        fig_status.update_layout(title="Media Spend by Status", height=350)
        st.plotly_chart(fig_status, use_container_width=True)
    
    with col2:
        st.dataframe(
            status_summary.style.format({
                "MediaCost": "${:,.0f}",
                "Profit": "${:,.0f}",
                "MediaShare": "{:.1%}",
                "ProfitShare": "{:.1%}"
            }),
            hide_index=True,
            use_container_width=True
        )
    
    # Trend chart (if time-series)
    if freq in ("M", "W") and "period_start" in pnl.columns:
        st.subheader("📊 Trend Analysis")
        
        ts = (
            pnl.groupby("period_start")
            .agg(Revenue=("Revenue", "sum"), MediaCost=("MediaCost", "sum"), Profit=("Profit", "sum"))
            .reset_index()
        )
        ts["period_start"] = pd.to_datetime(ts["period_start"])
        ts["Margin_pct"] = np.where(ts["Revenue"] > 0, ts["Profit"] / ts["Revenue"] * 100, 0)
        
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        fig_trend.add_trace(
            go.Bar(x=ts["period_start"], y=ts["Revenue"], name="Revenue", opacity=0.8),
            secondary_y=False
        )
        fig_trend.add_trace(
            go.Scatter(x=ts["period_start"], y=ts["Margin_pct"], mode="lines+markers", name="Margin %"),
            secondary_y=True
        )
        fig_trend.update_layout(title="Revenue & Margin Over Time", height=400)
        fig_trend.update_yaxes(title_text="Revenue", secondary_y=False)
        fig_trend.update_yaxes(title_text="Margin %", secondary_y=True)
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Builder ranking table
    st.header("🏆 Builder Ranking")
    
    pnl_sorted = pnl.sort_values(sort_by, ascending=False).head(top_n)
    
    display_df = pnl_sorted[[
        "BuilderRegionKey", "Status", "Revenue", "MediaCost", "Profit", "ROAS", "Margin_pct"
    ]].copy()
    
    if "PaidShare_any" in pnl_sorted.columns:
        display_df["PaidShare"] = pnl_sorted["PaidShare_any"]
    
    st.dataframe(
        display_df.style.format({
            "Revenue": "${:,.0f}",
            "MediaCost": "${:,.0f}",
            "Profit": "${:,.0f}",
            "ROAS": "{:.2f}",
            "Margin_pct": "{:.1%}",
            "PaidShare": "{:.1%}"
        }).background_gradient(subset=["Profit"], cmap="RdYlGn"),
        hide_index=True,
        use_container_width=True,
        height=600
    )
    
    # Download button
    st.download_button(
        label="📥 Download Builder P&L",
        data=export_to_excel(pnl_sorted, "builder_pnl.xlsx"),
        file_name=f"builder_pnl_{lens}_{freq}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # Diagnostics
    with st.expander("🔍 Diagnostics"):
        st.subheader("ROAS vs Margin Scatter")
        
        diag_df = pnl[pnl["MediaCost"] > 0].copy()
        if not diag_df.empty:
            diag_df["Margin_pct_100"] = diag_df["Margin_pct"] * 100
            
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=diag_df["ROAS"].clip(lower=0),
                y=diag_df["Margin_pct_100"],
                mode="markers",
                marker=dict(
                    size=10 + diag_df["MediaCost"] / diag_df["MediaCost"].max() * 30,
                    color=[get_status_color(s) for s in diag_df["Status"]],
                    opacity=0.7
                ),
                text=diag_df["BuilderRegionKey"],
                hovertemplate="<b>%{text}</b><br>ROAS: %{x:.2f}<br>Margin: %{y:.1f}%<extra></extra>"
            ))
            
            roas_med = diag_df["ROAS"].median()
            margin_med = diag_df["Margin_pct_100"].median()
            
            fig_scatter.add_hline(y=margin_med, line_dash="dash", line_color="gray", opacity=0.5)
            fig_scatter.add_vline(x=roas_med, line_dash="dash", line_color="gray", opacity=0.5)
            
            fig_scatter.update_layout(
                title="ROAS vs Margin (bubble size = media spend)",
                xaxis_title="ROAS",
                yaxis_title="Margin %",
                height=500
            )
            st.plotly_chart(fig_scatter, use_container_width=True)


if __name__ == "__main__":
    main()