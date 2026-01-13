"""
Network Optimization Engine v2.1
Enhanced with velocity tracking, budget-constrained optimization, and full traceability.
Fixed: Velocity calculation now handles sparse data properly.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class RiskLevel(Enum):
    CRITICAL = "🔴 Critical"
    HIGH = "🟠 High"
    MEDIUM = "🟡 Medium"
    LOW = "🟢 Low"
    ON_TRACK = "✅ On Track"


@dataclass
class VelocityMetrics:
    """Velocity analysis for a single builder."""
    builder_key: str
    pace_2d: float = 0.0
    pace_7d: float = 0.0
    pace_14d: float = 0.0
    decel_2d_vs_14d: float = 0.0
    decel_7d_vs_14d: float = 0.0
    is_decelerating: bool = False
    decel_severity: str = "None"


@dataclass
class AllocationRecord:
    """Single allocation in the media plan."""
    target_builder: str
    source_builder: str
    transfer_rate: float
    source_cpr: float
    effective_cpr: float
    leads_needed: float
    leads_allocated: float
    investment_amount: float
    source_surplus: float
    reasoning: str
    priority_rank: int = 0


@dataclass
class OptimizationResult:
    """Complete optimization output."""
    allocations: list = field(default_factory=list)
    total_budget_used: float = 0.0
    total_leads_recovered: float = 0.0
    unmet_demand: float = 0.0
    efficiency_score: float = 0.0
    warnings: list = field(default_factory=list)


def calculate_velocity_metrics(
    events_df: pd.DataFrame,
    builder_col: str = "Dest_BuilderRegionKey",
    date_col: str = "lead_date",
    all_builders: list = None
) -> pd.DataFrame:
    """
    Calculate rolling velocity metrics for each builder.
    FIXED: Handles sparse data, ensures all builders get metrics.
    
    Args:
        events_df: Events dataframe
        builder_col: Column with builder keys
        date_col: Column with dates
        all_builders: Optional list of all builders to include (even with no events)
    
    Returns DataFrame with columns:
        BuilderRegionKey, pace_2d, pace_7d, pace_14d, 
        decel_2d_vs_14d, decel_7d_vs_14d, is_decelerating, decel_severity
    """
    df = events_df.copy()
    
    # Find date column
    if date_col not in df.columns:
        date_col = "RefDate" if "RefDate" in df.columns else None
    if date_col is None or df.empty:
        # Return empty dataframe with correct columns for all builders
        if all_builders:
            return pd.DataFrame({
                "BuilderRegionKey": all_builders,
                "pace_2d": 0.0, "pace_7d": 0.0, "pace_14d": 0.0,
                "decel_2d_vs_14d": 0.0, "decel_7d_vs_14d": 0.0,
                "is_decelerating": False, "decel_severity": "No Data"
            })
        return pd.DataFrame()
    
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    
    # Filter to referrals if column exists
    if "is_referral" in df.columns:
        df_refs = df[df["is_referral"] == True].copy()
    else:
        df_refs = df.copy()
    
    # Get date range
    if df_refs.empty or df_refs[date_col].isna().all():
        max_date = pd.Timestamp.now()
    else:
        max_date = df_refs[date_col].max()
    
    # Get all builders (from parameter or from data)
    if all_builders:
        builders = list(set(all_builders))
    else:
        builders = df[builder_col].dropna().unique().tolist()
    
    if not builders:
        return pd.DataFrame()
    
    # Calculate daily counts per builder
    if not df_refs.empty:
        daily = (
            df_refs.groupby([builder_col, pd.Grouper(key=date_col, freq="D")])
            .size()
            .reset_index(name="daily_leads")
        )
        daily.columns = ["BuilderRegionKey", "date", "daily_leads"]
        daily_dict = daily.groupby("BuilderRegionKey").apply(
            lambda x: x.set_index("date")["daily_leads"].to_dict()
        ).to_dict()
    else:
        daily_dict = {}
    
    # Calculate metrics for each builder
    results = []
    for builder in builders:
        builder_daily = daily_dict.get(builder, {})
        
        # Sum leads in each window
        leads_2d = sum(
            builder_daily.get(max_date - pd.Timedelta(days=i), 0)
            for i in range(2)
        )
        leads_7d = sum(
            builder_daily.get(max_date - pd.Timedelta(days=i), 0)
            for i in range(7)
        )
        leads_14d = sum(
            builder_daily.get(max_date - pd.Timedelta(days=i), 0)
            for i in range(14)
        )
        
        # Calculate pace (leads per day)
        pace_2d = leads_2d / 2 if leads_2d > 0 else 0
        pace_7d = leads_7d / 7 if leads_7d > 0 else 0
        pace_14d = leads_14d / 14 if leads_14d > 0 else 0
        
        # Deceleration calculation
        # Compare recent pace to baseline (14-day)
        if pace_14d > 0.1:  # Only calculate decel if baseline is meaningful
            decel_2d = ((pace_2d - pace_14d) / pace_14d * 100)
            decel_7d = ((pace_7d - pace_14d) / pace_14d * 100)
        else:
            # If baseline is near zero, check if recent is also near zero
            decel_2d = 0.0
            decel_7d = 0.0
        
        # Flag deceleration (only if there was meaningful baseline activity)
        is_decel = False
        severity = "None"
        
        if pace_14d > 0.1:  # Had baseline activity
            if decel_2d < -50 or decel_7d < -40:
                is_decel = True
                severity = "Severe"
            elif decel_2d < -30 or decel_7d < -25:
                is_decel = True
                severity = "Moderate"
            elif decel_2d < -20 or decel_7d < -15:
                is_decel = True
                severity = "Mild"
        elif pace_14d == 0 and pace_7d == 0 and pace_2d == 0:
            severity = "No Activity"
        
        results.append({
            "BuilderRegionKey": builder,
            "pace_2d": round(pace_2d, 3),
            "pace_7d": round(pace_7d, 3),
            "pace_14d": round(pace_14d, 3),
            "decel_2d_vs_14d": round(decel_2d, 1),
            "decel_7d_vs_14d": round(decel_7d, 1),
            "is_decelerating": is_decel,
            "decel_severity": severity
        })
    
    return pd.DataFrame(results)


def calculate_shortfalls_v2(
    events_df: pd.DataFrame,
    total_events_df: pd.DataFrame = None,
    targets_df: pd.DataFrame = None,
    period_days: int = None
) -> pd.DataFrame:
    """
    Enhanced shortfall calculation with velocity metrics and risk scoring.
    """
    # Use existing logic as base
    if "is_referral" in events_df.columns:
        period_refs = events_df[events_df["is_referral"] == True]
    else:
        period_refs = events_df
    
    period_actuals = (
        period_refs.groupby("Dest_BuilderRegionKey")
        .size()
        .reset_index(name="Period_Referrals")
    )
    
    # Cumulative actuals
    source_df = total_events_df if total_events_df is not None else events_df
    if "is_referral" in source_df.columns:
        cum_refs = source_df[source_df["is_referral"] == True]
    else:
        cum_refs = source_df
    
    cum_actuals = (
        cum_refs.groupby("Dest_BuilderRegionKey")
        .size()
        .reset_index(name="Actual_Referrals")
    )
    
    # Targets
    if targets_df is None:
        builders = source_df["Dest_BuilderRegionKey"].dropna().unique()
        if "LeadTarget_from_job" in source_df.columns:
            targets_df = (
                source_df[["Dest_BuilderRegionKey", "LeadTarget_from_job", "WIP_JOB_LIVE_END"]]
                .drop_duplicates("Dest_BuilderRegionKey")
                .rename(columns={
                    "Dest_BuilderRegionKey": "BuilderRegionKey",
                    "LeadTarget_from_job": "LeadTarget"
                })
            )
        else:
            targets_df = pd.DataFrame({
                "BuilderRegionKey": builders,
                "LeadTarget": 50,
                "WIP_JOB_LIVE_END": pd.Timestamp.now() + pd.Timedelta(days=90)
            })
    else:
        targets_df = targets_df.rename(columns={
            "LeadTarget_from_job": "LeadTarget",
            "Builder": "BuilderRegionKey",
            "Dest_BuilderRegionKey": "BuilderRegionKey"
        })
    
    # Merge
    df = targets_df.merge(
        cum_actuals, 
        left_on="BuilderRegionKey", 
        right_on="Dest_BuilderRegionKey", 
        how="left"
    )
    df["Actual_Referrals"] = df["Actual_Referrals"].fillna(0)
    
    df = df.merge(
        period_actuals,
        left_on="BuilderRegionKey",
        right_on="Dest_BuilderRegionKey",
        how="left",
        suffixes=("", "_period")
    )
    df["Period_Referrals"] = df["Period_Referrals"].fillna(0)
    
    # Time calculations
    now = pd.Timestamp.now()
    if "WIP_JOB_LIVE_END" in df.columns:
        df["WIP_JOB_LIVE_END"] = pd.to_datetime(df["WIP_JOB_LIVE_END"], errors="coerce")
        df["Days_Remaining"] = (df["WIP_JOB_LIVE_END"] - now).dt.days.fillna(30).astype(int)
    else:
        df["Days_Remaining"] = 30
    df["Days_Remaining"] = df["Days_Remaining"].clip(lower=0)
    
    # Velocity
    velocity_days = max(period_days or 90, 1)
    df["Velocity_LeadsPerDay"] = df["Period_Referrals"] / velocity_days
    
    # Projections
    df["Projected_Additional"] = df["Velocity_LeadsPerDay"] * df["Days_Remaining"]
    df["Projected_Total"] = df["Actual_Referrals"] + df["Projected_Additional"]
    df["Pct_to_Target"] = np.where(
        df["LeadTarget"] > 0,
        df["Projected_Total"] / df["LeadTarget"] * 100,
        100
    )
    
    # Gap analysis
    df["Net_Gap"] = df["Projected_Total"] - df["LeadTarget"]
    df["Projected_Shortfall"] = np.where(df["Net_Gap"] < 0, abs(df["Net_Gap"]), 0)
    df["Projected_Surplus"] = np.where(df["Net_Gap"] > 0, df["Net_Gap"], 0)
    
    # Enhanced Risk Scoring
    df["Urgency_Factor"] = np.where(
        df["Days_Remaining"] > 0,
        1 / np.sqrt(df["Days_Remaining"] + 1),
        1.0
    )
    
    df["Risk_Score"] = (
        df["Projected_Shortfall"] * 10 * df["Urgency_Factor"]
    ).round(1)
    
    # Risk Level Classification
    def classify_risk(row):
        if row["Projected_Shortfall"] == 0:
            return RiskLevel.ON_TRACK.value
        if row["Risk_Score"] > 100 or (row["Days_Remaining"] < 14 and row["Projected_Shortfall"] > 10):
            return RiskLevel.CRITICAL.value
        if row["Risk_Score"] > 50 or row["Days_Remaining"] < 30:
            return RiskLevel.HIGH.value
        if row["Risk_Score"] > 20:
            return RiskLevel.MEDIUM.value
        return RiskLevel.LOW.value
    
    df["Risk_Level"] = df.apply(classify_risk, axis=1)
    
    # Catch-up pace required
    df["CatchUp_Pace_Req"] = np.where(
        (df["Projected_Shortfall"] > 0) & (df["Days_Remaining"] > 0),
        df["Projected_Shortfall"] / df["Days_Remaining"],
        0
    )
    
    return df


def analyze_network_leverage_v2(
    events_df: pd.DataFrame,
    excluded_sources: list = None
) -> pd.DataFrame:
    """
    Enhanced network leverage analysis with exclusion support.
    """
    excluded_sources = excluded_sources or []
    
    if "is_referral" in events_df.columns:
        refs = events_df[events_df["is_referral"] == True].copy()
    else:
        refs = events_df.copy()
    
    if refs.empty:
        return pd.DataFrame()
    
    # Filter out excluded sources
    if excluded_sources:
        refs = refs[~refs["MediaPayer_BuilderRegionKey"].isin(excluded_sources)]
    
    if refs.empty:
        return pd.DataFrame()
    
    # Source-level metrics
    agg_dict = {"MediaPayer_BuilderRegionKey": "size"}
    if "LeadId" in refs.columns:
        agg_dict = {"LeadId": "count"}
    if "MediaCost_referral_event" in refs.columns:
        agg_dict["MediaCost_referral_event"] = "sum"
    
    source_stats = refs.groupby("MediaPayer_BuilderRegionKey").agg(
        **{k: (k if k != "MediaPayer_BuilderRegionKey" else "LeadId", v if k != "MediaPayer_BuilderRegionKey" else "size") 
           for k, v in agg_dict.items()}
    ).reset_index()
    
    # Simpler approach
    source_stats = refs.groupby("MediaPayer_BuilderRegionKey").agg(
        Total_Referrals_Sent=("Dest_BuilderRegionKey", "size"),
    ).reset_index()
    
    if "MediaCost_referral_event" in refs.columns:
        spend = refs.groupby("MediaPayer_BuilderRegionKey")["MediaCost_referral_event"].sum().reset_index()
        spend.columns = ["MediaPayer_BuilderRegionKey", "Total_Media_Spend"]
        source_stats = source_stats.merge(spend, on="MediaPayer_BuilderRegionKey", how="left")
    else:
        source_stats["Total_Media_Spend"] = source_stats["Total_Referrals_Sent"] * 50
    
    source_stats["CPR_base"] = np.where(
        source_stats["Total_Referrals_Sent"] > 0,
        source_stats["Total_Media_Spend"] / source_stats["Total_Referrals_Sent"],
        np.nan
    )
    
    # Flow metrics
    flows = (
        refs.groupby(["MediaPayer_BuilderRegionKey", "Dest_BuilderRegionKey"])
        .size()
        .reset_index(name="Referrals_to_Target")
    )
    
    # Merge and calculate
    leverage = flows.merge(source_stats, on="MediaPayer_BuilderRegionKey", how="left")
    
    leverage["Transfer_Rate"] = np.where(
        leverage["Total_Referrals_Sent"] > 0,
        leverage["Referrals_to_Target"] / leverage["Total_Referrals_Sent"],
        0
    )
    
    leverage["eCPR"] = np.where(
        leverage["Transfer_Rate"] > 0,
        leverage["CPR_base"] / leverage["Transfer_Rate"],
        np.inf
    )
    leverage["eCPR"] = leverage["eCPR"].replace([np.inf], 99999)
    
    return leverage


def run_budget_constrained_optimization(
    shortfall_df: pd.DataFrame,
    leverage_df: pd.DataFrame,
    budget: float = 50000.0,
    excluded_sources: list = None,
    min_transfer_rate: float = 0.01
) -> OptimizationResult:
    """
    Budget-constrained greedy optimization.
    """
    excluded_sources = excluded_sources or []
    result = OptimizationResult()
    
    if leverage_df.empty:
        result.warnings.append("No network leverage data available")
        return result
    
    leverage = leverage_df[
        (~leverage_df["MediaPayer_BuilderRegionKey"].isin(excluded_sources)) &
        (leverage_df["Transfer_Rate"] >= min_transfer_rate) &
        (leverage_df["eCPR"] < 99999)
    ].copy()
    
    if leverage.empty:
        result.warnings.append("No valid source paths after filtering")
        return result
    
    deficits = shortfall_df[shortfall_df["Projected_Shortfall"] > 0].copy()
    if deficits.empty:
        result.warnings.append("No builders with shortfall")
        return result
    
    surplus_map = shortfall_df.set_index("BuilderRegionKey")["Projected_Surplus"].to_dict()
    
    remaining_budget = budget
    demand_remaining = deficits.set_index("BuilderRegionKey")["Projected_Shortfall"].to_dict()
    
    candidates = []
    for _, row in leverage.iterrows():
        source = row["MediaPayer_BuilderRegionKey"]
        target = row["Dest_BuilderRegionKey"]
        
        if target not in demand_remaining or demand_remaining.get(target, 0) <= 0:
            continue
        
        source_surplus = surplus_map.get(source, 0)
        
        candidates.append({
            "source": source,
            "target": target,
            "transfer_rate": row["Transfer_Rate"],
            "cpr_base": row["CPR_base"],
            "ecpr": row["eCPR"],
            "source_surplus": source_surplus,
            "has_surplus": source_surplus > 0
        })
    
    candidates.sort(key=lambda x: (-x["has_surplus"], x["ecpr"]))
    
    priority_rank = 0
    for cand in candidates:
        if remaining_budget <= 0:
            break
        
        target = cand["target"]
        source = cand["source"]
        
        leads_needed = demand_remaining.get(target, 0)
        if leads_needed <= 0:
            continue
        
        cost_per_lead = cand["ecpr"]
        max_leads_by_budget = remaining_budget / cost_per_lead if cost_per_lead > 0 else 0
        
        leads_to_allocate = min(leads_needed, max_leads_by_budget)
        if leads_to_allocate <= 0:
            continue
        
        investment = leads_to_allocate * cost_per_lead
        
        if cand["has_surplus"]:
            reasoning = f"Source has {cand['source_surplus']:.0f} surplus; TR={cand['transfer_rate']:.1%}"
        else:
            reasoning = f"Best eCPR (${cand['ecpr']:.0f}); TR={cand['transfer_rate']:.1%}"
        
        priority_rank += 1
        allocation = AllocationRecord(
            target_builder=target,
            source_builder=source,
            transfer_rate=cand["transfer_rate"],
            source_cpr=cand["cpr_base"],
            effective_cpr=cand["ecpr"],
            leads_needed=leads_needed,
            leads_allocated=leads_to_allocate,
            investment_amount=investment,
            source_surplus=cand["source_surplus"],
            reasoning=reasoning,
            priority_rank=priority_rank
        )
        
        result.allocations.append(allocation)
        
        remaining_budget -= investment
        demand_remaining[target] -= leads_to_allocate
        result.total_budget_used += investment
        result.total_leads_recovered += leads_to_allocate
    
    result.unmet_demand = sum(max(0, v) for v in demand_remaining.values())
    
    if result.total_budget_used > 0:
        result.efficiency_score = result.total_leads_recovered / result.total_budget_used
    
    if result.unmet_demand > 0:
        result.warnings.append(f"Budget exhausted with {result.unmet_demand:.0f} leads still needed")
    
    if remaining_budget > budget * 0.5:
        result.warnings.append(f"Only {((budget - remaining_budget) / budget * 100):.0f}% of budget allocated")
    
    return result


def optimization_result_to_dataframe(result: OptimizationResult) -> pd.DataFrame:
    """Convert OptimizationResult to DataFrame."""
    if not result.allocations:
        return pd.DataFrame()
    
    records = []
    for alloc in result.allocations:
        records.append({
            "Priority": alloc.priority_rank,
            "Target_Builder": alloc.target_builder,
            "Source_Builder": alloc.source_builder,
            "Transfer_Rate": alloc.transfer_rate,
            "Source_CPR": alloc.source_cpr,
            "Effective_CPR": alloc.effective_cpr,
            "Leads_Needed": alloc.leads_needed,
            "Leads_Allocated": alloc.leads_allocated,
            "Investment": alloc.investment_amount,
            "Source_Surplus": alloc.source_surplus,
            "Reasoning": alloc.reasoning
        })
    
    return pd.DataFrame(records)


# Legacy wrappers
def calculate_shortfalls(events_df, targets_df=None, period_days=None, total_events_df=None):
    return calculate_shortfalls_v2(events_df, total_events_df, targets_df, period_days)

def analyze_network_leverage(events_df):
    return analyze_network_leverage_v2(events_df)

def generate_global_media_plan(shortfall_df, leverage_df):
    result = run_budget_constrained_optimization(shortfall_df, leverage_df)
    return optimization_result_to_dataframe(result)

def generate_investment_strategies(focus_builder, shortfall_data, leverage_data, events_df):
    strategies = leverage_data[leverage_data["Dest_BuilderRegionKey"] == focus_builder].copy()
    if strategies.empty:
        return pd.DataFrame()
    
    target_row = shortfall_data[shortfall_data["BuilderRegionKey"] == focus_builder]
    if target_row.empty:
        return strategies
    
    col = "Projected_Shortfall" if "Projected_Shortfall" in target_row.columns else "Shortfall"
    shortfall = target_row[col].iloc[0] if col in target_row.columns else 0
    
    results = []
    for _, strat in strategies.iterrows():
        tr = strat["Transfer_Rate"]
        ecpr = strat["eCPR"]
        investment = shortfall * ecpr if shortfall > 0 and ecpr < 99999 else 0
        
        results.append({
            "Source_Builder": strat["MediaPayer_BuilderRegionKey"],
            "Transfer_Rate": tr,
            "Base_CPR": strat["CPR_base"],
            "Effective_CPR": ecpr,
            "Investment_Required": investment
        })
    
    return pd.DataFrame(results).sort_values("Effective_CPR")

def compute_effective_network_cpr(events_df, shortfall_df):
    leverage = analyze_network_leverage_v2(events_df)
    if leverage.empty:
        return pd.DataFrame()
    
    grouped = leverage.groupby("MediaPayer_BuilderRegionKey").agg({
        "Total_Referrals_Sent": "first",
        "CPR_base": "first",
        "Dest_BuilderRegionKey": "nunique"
    }).rename(columns={"Dest_BuilderRegionKey": "Beneficiaries_Count", "CPR_base": "CPR"})
    
    return grouped.reset_index().rename(columns={"MediaPayer_BuilderRegionKey": "BuilderRegionKey"})