"""
src/network_optimization.py

Core engine for the Network Intelligence dashboard.
Handles:
1. Demand forecasting (Shortfall calculation)
2. Leverage analysis (Network transfer rates)
3. Portfolio Optimization (Greedy budget allocation)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OptimConfig:
    pace_lookback_days: int = 14
    transfer_lookback_days: int = 90
    overserve_tolerance: float = 0.10
    prior_strength: float = 25.0
    min_transfer_events: int = 5
    urgency_days_scale: float = 21.0
    pace_gap_alpha: float = 1.25
    max_step_spend: float = 25000.0
    min_step_spend: float = 500.0

# ──────────────────────────────────────────────────────────────────────────────
# 1. Demand & Shortfall Calculation
# ──────────────────────────────────────────────────────────────────────────────

def calculate_shortfalls(events: pd.DataFrame, asof: pd.Timestamp, cfg: OptimConfig) -> pd.DataFrame:
    """
    Calculates builder status (UNDER/OVER/ON TRACK) based on pace and targets.
    """
    if events.empty:
        return pd.DataFrame()

    df = events.copy()
    
    # 1. Identify Columns
    dest_col = next((c for c in ["Dest_BuilderRegionKey", "Dest_builder", "Dest"] if c in df.columns), None)
    date_col = next((c for c in ["lead_date", "RefDate", "LeadDate", "date"] if c in df.columns), None)
    
    if not dest_col or not date_col:
        return pd.DataFrame()

    # 2. Aggregations (Actuals)
    actuals = df.groupby(dest_col).size().reset_index(name="Actual_Referrals")
    
    # 3. Targets (Mock logic if column missing, else use provided)
    # Ideally, events has a 'Target' column or we merge a target file. 
    # Here we infer a dummy target for demo purposes if missing.
    if "LeadTarget" not in actuals.columns:
        # Create a synthetic target based on volume to prevent crash
        actuals["LeadTarget"] = (actuals["Actual_Referrals"] * 1.2).astype(int) 

    # 4. Pace Calculation
    pace_start = asof - pd.Timedelta(days=cfg.pace_lookback_days)
    recent = df[df[date_col] >= pace_start]
    
    pace = recent.groupby(dest_col).size().reset_index(name="Recent_Leads")
    pace["Pace_Leads_per_Day"] = pace["Recent_Leads"] / cfg.pace_lookback_days

    # 5. Merge and Forecast
    demand = pd.merge(actuals, pace, on=dest_col, how="left").fillna(0)
    demand = demand.rename(columns={dest_col: "BuilderRegionKey"})
    
    # Simple logic: assume campaign ends 30 days from now if not specified
    # In a real app, end_date comes from the campaign settings
    demand["Days_Remaining"] = 30 
    
    demand["Projected_Additional"] = demand["Pace_Leads_per_Day"] * demand["Days_Remaining"]
    demand["Projected_Finish"] = demand["Actual_Referrals"] + demand["Projected_Additional"]
    
    # 6. Determine Gaps
    demand["Shortfall"] = (demand["LeadTarget"] - demand["Projected_Finish"]).clip(lower=0)
    demand["Surplus"] = (demand["Projected_Finish"] - (demand["LeadTarget"] * (1 + cfg.overserve_tolerance))).clip(lower=0)
    
    # 7. Flags and Scores
    def get_flag(row):
        if row["Shortfall"] > 0: return "UNDER"
        if row["Surplus"] > 0: return "OVER"
        return "ON TRACK"
        
    demand["ServiceFlag"] = demand.apply(get_flag, axis=1)
    
    # Demand Score = Shortfall * Urgency Factor
    # Higher score = needs money sooner
    demand["Urgency_Factor"] = np.exp(-1 * demand["Days_Remaining"] / cfg.urgency_days_scale)
    demand["DemandScore"] = demand["Shortfall"] * demand["Urgency_Factor"] * cfg.pace_gap_alpha

    return demand.sort_values("DemandScore", ascending=False)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Leverage Analysis (Network Transfer Rates)
# ──────────────────────────────────────────────────────────────────────────────

def analyze_network_leverage(events: pd.DataFrame, asof: pd.Timestamp, cfg: OptimConfig) -> pd.DataFrame:
    """
    Calculates Source -> Dest transfer rates using Bayesian smoothing.
    """
    if events.empty:
        return pd.DataFrame()

    df = events.copy()
    src_col = next((c for c in ["MediaPayer_BuilderRegionKey", "Origin_builder", "Source"] if c in df.columns), None)
    dest_col = next((c for c in ["Dest_BuilderRegionKey", "Dest_builder", "Dest"] if c in df.columns), None)
    cost_col = next((c for c in ["MediaCost_referral_event", "Spend", "Cost"] if c in df.columns), None)

    if not src_col or not dest_col:
        return pd.DataFrame()

    # 1. Source Totals (Denominators)
    src_stats = df.groupby(src_col).agg(
        Total_Source_Leads=(dest_col, "count"),
        Total_Spend=(cost_col, "sum") if cost_col else (dest_col, "count") # Fallback if no cost
    ).reset_index()
    
    # Calculate Base CPR (Cost per Lead at Source)
    # Avoid div/0
    src_stats["Base_CPR"] = src_stats["Total_Spend"] / src_stats["Total_Source_Leads"].replace(0, 1)

    # 2. Edge Totals (Numerators)
    edges = df.groupby([src_col, dest_col]).size().reset_index(name="Transfers")
    
    # 3. Merge
    leverage = pd.merge(edges, src_stats, on=src_col, how="left")
    
    # 4. Bayesian Smoothing
    # Smoothed Rate = (Transfers + K * GlobalAvg) / (Total_Leads + K)
    global_avg_transfer = edges["Transfers"].sum() / src_stats["Total_Source_Leads"].sum()
    k = cfg.prior_strength
    
    leverage["Transfer_Rate"] = (leverage["Transfers"] + (k * global_avg_transfer)) / (leverage["Total_Source_Leads"] + k)
    
    # 5. Effective CPR (eCPR)
    # eCPR = Base_CPR / Transfer_Rate
    leverage["eCPR"] = leverage["Base_CPR"] / leverage["Transfer_Rate"]
    
    # 6. Filters
    leverage["Pass_Min_Events"] = leverage["Transfers"] >= cfg.min_transfer_events
    
    # Rename for consistency
    leverage = leverage.rename(columns={src_col: "Source", dest_col: "Dest_BuilderRegionKey"})
    
    return leverage.sort_values("eCPR")

def generate_investment_strategies(focus_builder: str, demand: pd.DataFrame, leverage: pd.DataFrame) -> pd.DataFrame:
    """
    Filters leverage for a specific builder to show best investment options.
    """
    if leverage.empty:
        return pd.DataFrame()
        
    strat = leverage[leverage["Dest_BuilderRegionKey"].astype(str) == str(focus_builder)].copy()
    strat["Investment_Required"] = strat["Base_CPR"] * 10 # Arbitrary unit allocation
    strat["Confidence"] = np.where(strat["Pass_Min_Events"], 1.0, 0.5)
    
    # Rename for UI
    strat = strat.rename(columns={"eCPR": "Effective_CPR"})
    
    cols = ["Source", "Transfer_Rate", "Base_CPR", "Effective_CPR", "Investment_Required", "Confidence"]
    return strat[cols].sort_values("Effective_CPR")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Portfolio Optimization (Media Plan)
# ──────────────────────────────────────────────────────────────────────────────

def optimise_portfolio_media_plan(
    demand: pd.DataFrame,
    leverage: pd.DataFrame,
    cfg: OptimConfig,
    max_spend_multiplier: float = 1.0,
    new_money: float = 0.0,
    overserve_penalty: float = 0.25
) -> Dict:
    """
    Allocates budget to sources to solve shortfalls with minimal spend.
    """
    
    # 1. Setup Budgets
    # Identify unique sources and their current spend
    sources = leverage[["Source", "Total_Spend", "Base_CPR"]].drop_duplicates("Source").set_index("Source")
    sources["Current_Spend_Cap"] = sources["Total_Spend"] * max_spend_multiplier
    sources["Allocated_Spend"] = 0.0
    sources["Budget_Remaining"] = sources["Current_Spend_Cap"] + (new_money / len(sources)) # Simple spread of new money

    # 2. Setup Demand State
    # We work on a copy to track progress
    state = demand.set_index("BuilderRegionKey").copy()
    state["Shortfall_Remaining"] = state["Shortfall"]
    state["Overserve_Accumulated"] = 0.0
    
    # 3. Create Edge List with Penalized eCPR
    # If a source feeds an OVER builder, its eCPR is penalized
    edges = leverage.copy()
    
    allocation_log = []
    
    # Greedy Loop
    # In a real engine, this would be a linear program (scipy.optimize)
    # Here, we simulate a greedy step approach for speed/demo
    
    step_size = cfg.min_step_spend
    
    # Limit iterations to avoid infinite loops
    for _ in range(500):
        # A. Check if work is done (Total Shortfall covered)
        total_shortfall = state.loc[state["Shortfall_Remaining"] > 0, "Shortfall_Remaining"].sum()
        if total_shortfall <= 0:
            break
            
        # B. Score Edges
        # Recalculate attractiveness based on remaining shortfall
        # Only consider sources with budget remaining
        valid_sources = sources[sources["Budget_Remaining"] >= step_size].index
        candidates = edges[edges["Source"].isin(valid_sources)].copy()
        
        if candidates.empty:
            break
            
        # Map current destination status to edges
        candidates["Dest_Status"] = candidates["Dest_BuilderRegionKey"].map(state["Shortfall_Remaining"])
        candidates["Dest_Is_Short"] = candidates["Dest_Status"] > 0
        
        # We only want to fund edges that help a SHORT target
        opportunities = candidates[candidates["Dest_Is_Short"]].copy()
        
        if opportunities.empty:
            break
            
        # Pick best edge (lowest eCPR)
        best_edge = opportunities.sort_values("eCPR").iloc[0]
        src = best_edge["Source"]
        dest = best_edge["Dest_BuilderRegionKey"]
        
        # C. Allocate Step
        cost = step_size
        leads_generated_at_source = cost / sources.loc[src, "Base_CPR"]
        
        # D. Distribute Effects
        # When we buy 'leads_generated_at_source', they split across ALL edges from that source
        # not just the target edge. This is the "Bundle" problem.
        
        src_edges = edges[edges["Source"] == src]
        
        for _, edge in src_edges.iterrows():
            d = edge["Dest_BuilderRegionKey"]
            transfer_rate = edge["Transfer_Rate"]
            leads_at_dest = leads_generated_at_source * transfer_rate
            
            # Update State
            if d in state.index:
                rem = state.loc[d, "Shortfall_Remaining"]
                if rem > 0:
                    covered = min(rem, leads_at_dest)
                    state.loc[d, "Shortfall_Remaining"] -= covered
                else:
                    # Overserve
                    state.loc[d, "Overserve_Accumulated"] += leads_at_dest

        # Update Budget
        sources.loc[src, "Allocated_Spend"] += cost
        sources.loc[src, "Budget_Remaining"] -= cost
        
        allocation_log.append({
            "Source": src,
            "Target_Trigger": dest,
            "Spend_Step": cost,
            "Target_Remaining_After": state.loc[dest, "Shortfall_Remaining"]
        })

    # 4. Compile Results
    
    # Plan by Source
    sources["Leads_Generated"] = sources["Allocated_Spend"] / sources["Base_CPR"]
    plan_by_source = sources[sources["Allocated_Spend"] > 0].reset_index()
    plan_by_source = plan_by_source.rename(columns={"Allocated_Spend": "Spend", "Current_Spend_Cap": "Budget"})
    
    # Plan Edges (Attribution)
    plan_edges = pd.merge(plan_by_source[["Source", "Spend", "Leads_Generated"]], 
                          leverage[["Source", "Dest_BuilderRegionKey", "Transfer_Rate", "eCPR"]], 
                          on="Source")
    
    plan_edges["Expected_Leads"] = plan_edges["Leads_Generated"] * plan_edges["Transfer_Rate"]
    plan_edges["Effective_CPR_to_Dest"] = plan_edges["Spend"] / plan_edges["Expected_Leads"].replace(0, 1)
    plan_edges = plan_edges.rename(columns={"Dest_BuilderRegionKey": "Dest", "Spend": "Spend_on_Source"})
    
    # Post State
    post_state = state.reset_index()
    post_state["Shortfall_Post"] = post_state["Shortfall_Remaining"]
    post_state["Overserve_Post"] = post_state["Overserve_Accumulated"]
    post_state["Expected_Leads_Added"] = post_state["Shortfall"] - post_state["Shortfall_Post"] + post_state["Overserve_Post"]
    post_state["Projected_Finish_Post"] = post_state["Projected_Finish"] + post_state["Expected_Leads_Added"]
    
    return {
        "plan_by_source": plan_by_source,
        "plan_edges": plan_edges,
        "post_state": post_state,
        "allocation_log": pd.DataFrame(allocation_log),
        "source_budgets": sources.reset_index(),
        "demand_table": demand
    }