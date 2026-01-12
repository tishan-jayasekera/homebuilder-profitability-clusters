from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
import pandas as pd


# =============================================================================
# Config
# =============================================================================

@dataclass
class OptimConfig:
    # Demand
    pace_lookback_days: int = 14
    overserve_tolerance: float = 0.10

    # Urgency scoring
    urgency_days_scale: float = 21.0   # smaller => more urgency as deadline approaches
    pace_gap_alpha: float = 1.25       # weight on required pace vs current pace

    # Transfers / leverage
    transfer_lookback_days: int = 90
    min_transfer_events: int = 5
    prior_strength: float = 25.0       # Bayesian smoothing toward global distribution

    # Portfolio optimiser
    max_step_spend: float = 25_000.0
    min_step_spend: float = 500.0


# =============================================================================
# Helpers
# =============================================================================

def _first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _clip_pos(x):
    return np.maximum(0.0, x)


def _safe_div(a, b, fill=np.nan):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    if isinstance(out, np.ndarray):
        out[~np.isfinite(out)] = fill
    else:
        if not np.isfinite(out):
            out = fill
    return out


# =============================================================================
# 1) Demand: forecast-aware shortfalls
# =============================================================================

def calculate_shortfalls(
    events_df: pd.DataFrame,
    targets_df: pd.DataFrame = None,
    *,
    asof: Optional[pd.Timestamp] = None,
    cfg: Optional[OptimConfig] = None,
    # Optional overrides for column mapping
    dest_col: Optional[str] = None,
    is_ref_col: Optional[str] = None,
    date_col: Optional[str] = None,
    target_col: Optional[str] = None,
    end_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Forecast-aware demand table:
      - Actual to date
      - Recent pace (lookback)
      - Days remaining until campaign end
      - Projected finish = Actual + Pace * DaysRemaining
      - Shortfall = max(0, Target - ProjectedFinish)
      - DemandScore weights shortfall by urgency + pace gap

    Returns builder-level table with:
      BuilderRegionKey, LeadTarget, Actual_Referrals, Pace_Leads_per_Day,
      Days_Remaining, Projected_Finish, Shortfall, Surplus, DemandScore, ServiceFlag
    """
    if events_df is None or events_df.empty:
        return pd.DataFrame()

    cfg = cfg or OptimConfig()
    asof = pd.Timestamp.today().normalize() if asof is None else pd.to_datetime(asof).normalize()

    # Detect columns if not provided
    dest_col = dest_col or _first_col(events_df, ["Dest_BuilderRegionKey", "Dest_builder", "DestBuilderRegionKey", "Dest"])
    is_ref_col = is_ref_col or _first_col(events_df, ["is_referral", "IsReferral", "isReferral"])
    date_col = date_col or _first_col(events_df, ["lead_date", "RefDate", "ref_date", "LeadDate", "date"])

    # Targets / end-date columns (best effort)
    target_col = target_col or _first_col(events_df, ["LeadTarget_from_job", "LeadTarget", "lead_target", "Target"])
    end_col = end_col or _first_col(events_df, ["WIP_JOB_LIVE_END", "JobLiveEnd", "campaign_end", "EndDate"])

    if dest_col is None:
        raise ValueError("calculate_shortfalls: could not find destination builder column")

    df = events_df.copy()

    # Referral filter (inbound)
    if is_ref_col is not None:
        # handle stringy booleans
        if df[is_ref_col].dtype == object:
            df[is_ref_col] = df[is_ref_col].astype(str).str.lower().isin(["true", "1", "yes", "y"])
        df = df[df[is_ref_col] == True].copy()

    # Dates
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df_actual = df[df[date_col].notna() & (df[date_col] <= asof)].copy()
    else:
        df_actual = df

    # Actual inbound to date
    actuals = (
        df_actual.groupby(dest_col)
        .size()
        .reset_index(name="Actual_Referrals")
        .rename(columns={dest_col: "BuilderRegionKey"})
    )

    # Pace over lookback window
    if date_col is not None:
        start_pace = asof - pd.Timedelta(days=int(cfg.pace_lookback_days))
        df_pace = df_actual[df_actual[date_col].notna() & (df_actual[date_col] >= start_pace)].copy()
        pace = (
            df_pace.groupby(dest_col)
            .size()
            .reset_index(name="Pace_Leads")
            .rename(columns={dest_col: "BuilderRegionKey"})
        )
        pace["Pace_Leads_per_Day"] = pace["Pace_Leads"] / float(max(1, int(cfg.pace_lookback_days)))
    else:
        pace = pd.DataFrame(columns=["BuilderRegionKey", "Pace_Leads", "Pace_Leads_per_Day"])

    # Build targets_df if not provided
    if targets_df is None:
        builders = pd.Series(events_df[dest_col].dropna().unique(), name="BuilderRegionKey")
        targets_df = pd.DataFrame({"BuilderRegionKey": builders})

        if target_col is not None:
            tmp = events_df[[dest_col, target_col]].dropna().drop_duplicates(dest_col)
            tmp = tmp.rename(columns={dest_col: "BuilderRegionKey", target_col: "LeadTarget"})
            targets_df = targets_df.merge(tmp, on="BuilderRegionKey", how="left")
        else:
            targets_df["LeadTarget"] = 0.0

        if end_col is not None:
            tmp = events_df[[dest_col, end_col]].dropna().drop_duplicates(dest_col)
            tmp = tmp.rename(columns={dest_col: "BuilderRegionKey", end_col: "WIP_JOB_LIVE_END"})
            targets_df = targets_df.merge(tmp, on="BuilderRegionKey", how="left")
        else:
            targets_df["WIP_JOB_LIVE_END"] = pd.NaT
    else:
        targets_df = targets_df.copy()
        if "BuilderRegionKey" not in targets_df.columns:
            raise ValueError("targets_df must contain BuilderRegionKey")
        if "LeadTarget" not in targets_df.columns:
            # try rename
            if "LeadTarget_from_job" in targets_df.columns:
                targets_df = targets_df.rename(columns={"LeadTarget_from_job": "LeadTarget"})
            else:
                targets_df["LeadTarget"] = 0.0
        if "WIP_JOB_LIVE_END" not in targets_df.columns:
            targets_df["WIP_JOB_LIVE_END"] = pd.NaT

    out = targets_df.merge(actuals, on="BuilderRegionKey", how="left")
    out = out.merge(pace[["BuilderRegionKey", "Pace_Leads_per_Day"]], on="BuilderRegionKey", how="left")

    out["Actual_Referrals"] = out["Actual_Referrals"].fillna(0.0).astype(float)
    out["LeadTarget"] = out["LeadTarget"].fillna(0.0).astype(float)
    out["Pace_Leads_per_Day"] = out["Pace_Leads_per_Day"].fillna(0.0).astype(float)

    out["WIP_JOB_LIVE_END"] = pd.to_datetime(out.get("WIP_JOB_LIVE_END", pd.NaT), errors="coerce")
    out["Days_Remaining"] = (out["WIP_JOB_LIVE_END"] - asof).dt.days
    out["Days_Remaining_Fill"] = out["Days_Remaining"].fillna(9999).clip(lower=0)

    out["Forecast_Remaining"] = out["Pace_Leads_per_Day"] * out["Days_Remaining_Fill"]
    out["Projected_Finish"] = out["Actual_Referrals"] + out["Forecast_Remaining"]

    out["Shortfall"] = _clip_pos(out["LeadTarget"] - out["Projected_Finish"])
    out["Surplus"] = _clip_pos(out["Projected_Finish"] - out["LeadTarget"])

    out["Required_Pace"] = np.where(
        out["Days_Remaining_Fill"] > 0,
        out["Shortfall"] / out["Days_Remaining_Fill"],
        out["Shortfall"]
    )
    out["Pace_Ratio"] = out["Required_Pace"] / np.maximum(out["Pace_Leads_per_Day"], 1e-6)

    urgency_factor = 1.0 + (cfg.urgency_days_scale / (out["Days_Remaining_Fill"] + cfg.urgency_days_scale))
    pace_gap = np.maximum(0.0, out["Pace_Ratio"] - 1.0)
    pace_factor = 1.0 + cfg.pace_gap_alpha * pace_gap

    out["DemandScore"] = out["Shortfall"] * urgency_factor * pace_factor

    out["ServiceFlag"] = np.select(
        [
            out["Shortfall"] > 0,
            out["Projected_Finish"] > out["LeadTarget"] * (1.0 + cfg.overserve_tolerance),
        ],
        ["UNDER", "OVER"],
        default="ON_TRACK"
    )

    out = out.sort_values(["DemandScore", "Shortfall"], ascending=[False, False]).reset_index(drop=True)
    return out


# =============================================================================
# 2) Leverage: smoothed transfers + base CPR
# =============================================================================

def analyze_network_leverage(
    events_df: pd.DataFrame,
    *,
    asof: Optional[pd.Timestamp] = None,
    cfg: Optional[OptimConfig] = None,
    # Optional overrides for column mapping
    source_col: Optional[str] = None,
    dest_col: Optional[str] = None,
    is_ref_col: Optional[str] = None,
    date_col: Optional[str] = None,
    spend_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build (Source, Dest) leverage table:
      - Referrals_to_Dest (count)
      - Total_Referrals_Sent (per source)
      - Total_Media_Spend (per source)
      - CPR_base (per source)
      - Transfer_Rate_Smoothed (Bayesian)
      - eCPR = CPR_base / Transfer_Rate_Smoothed
      - Confidence
    """
    if events_df is None or events_df.empty:
        return pd.DataFrame()

    cfg = cfg or OptimConfig()
    asof = pd.Timestamp.today().normalize() if asof is None else pd.to_datetime(asof).normalize()

    source_col = source_col or _first_col(events_df, ["MediaPayer_BuilderRegionKey", "Origin_builder", "Source_BuilderRegionKey", "Source"])
    dest_col = dest_col or _first_col(events_df, ["Dest_BuilderRegionKey", "Dest_builder", "DestBuilderRegionKey", "Dest"])
    is_ref_col = is_ref_col or _first_col(events_df, ["is_referral", "IsReferral", "isReferral"])
    date_col = date_col or _first_col(events_df, ["lead_date", "RefDate", "ref_date", "LeadDate", "date"])
    spend_col = spend_col or _first_col(events_df, ["MediaCost_referral_event", "MediaCost", "media_cost", "Spend", "Cost"])

    if source_col is None or dest_col is None:
        raise ValueError("analyze_network_leverage: could not find source/dest columns")

    df = events_df.copy()

    if is_ref_col is not None:
        if df[is_ref_col].dtype == object:
            df[is_ref_col] = df[is_ref_col].astype(str).str.lower().isin(["true", "1", "yes", "y"])
        df = df[df[is_ref_col] == True].copy()

    # Lookback window
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        start = asof - pd.Timedelta(days=int(cfg.transfer_lookback_days))
        df = df[df[date_col].notna() & (df[date_col] >= start) & (df[date_col] <= asof)].copy()

    if df.empty:
        return pd.DataFrame()

    # counts by flow
    flows = df.groupby([source_col, dest_col]).size().reset_index(name="Referrals_to_Dest")

    # source totals
    src_totals = df.groupby(source_col).size().reset_index(name="Total_Referrals_Sent")

    # spend per source
    if spend_col is not None:
        spend = df.groupby(source_col)[spend_col].sum().reset_index(name="Total_Media_Spend")
    else:
        spend = src_totals[[source_col]].copy()
        spend["Total_Media_Spend"] = np.nan

    src_stats = src_totals.merge(spend, on=source_col, how="left")
    src_stats["Total_Media_Spend"] = src_stats["Total_Media_Spend"].fillna(0.0).astype(float)
    src_stats["CPR_base"] = np.where(
        src_stats["Total_Referrals_Sent"] > 0,
        src_stats["Total_Media_Spend"] / src_stats["Total_Referrals_Sent"],
        np.nan
    )

    # Global destination prior distribution
    dest_global = df.groupby(dest_col).size().reset_index(name="Global_Dest_Count")
    dest_global["Global_p"] = dest_global["Global_Dest_Count"] / dest_global["Global_Dest_Count"].sum()

    lev = flows.merge(src_stats, on=source_col, how="left")
    lev = lev.merge(dest_global[[dest_col, "Global_p"]], on=dest_col, how="left")
    lev["Global_p"] = lev["Global_p"].fillna(0.0)

    lev["Transfer_Rate"] = lev["Referrals_to_Dest"] / lev["Total_Referrals_Sent"].replace(0, np.nan)

    lev["Transfer_Rate_Smoothed"] = (
        lev["Referrals_to_Dest"] + cfg.prior_strength * lev["Global_p"]
    ) / (
        lev["Total_Referrals_Sent"] + cfg.prior_strength
    )

    lev["Confidence"] = lev["Total_Referrals_Sent"] / (lev["Total_Referrals_Sent"] + cfg.prior_strength)

    lev["eCPR"] = np.where(
        lev["Transfer_Rate_Smoothed"] > 0,
        lev["CPR_base"] / lev["Transfer_Rate_Smoothed"],
        np.inf
    )

    lev["Pass_Min_Events"] = lev["Referrals_to_Dest"] >= int(cfg.min_transfer_events)

    lev = lev.rename(columns={
        source_col: "MediaPayer_BuilderRegionKey",
        dest_col: "Dest_BuilderRegionKey",
    })

    lev = lev.sort_values(["eCPR", "Transfer_Rate_Smoothed"], ascending=[True, False]).reset_index(drop=True)
    return lev


# =============================================================================
# 3) Focus builder strategies (rank best sources to close one gap)
# =============================================================================

def generate_investment_strategies(
    focus_builder: str,
    shortfall_data: pd.DataFrame,
    leverage_data: pd.DataFrame,
) -> pd.DataFrame:
    if shortfall_data is None or shortfall_data.empty:
        return pd.DataFrame()
    if leverage_data is None or leverage_data.empty:
        return pd.DataFrame()

    tgt = shortfall_data[shortfall_data["BuilderRegionKey"] == focus_builder]
    if tgt.empty:
        return pd.DataFrame()

    shortfall = float(tgt["Shortfall"].iloc[0])
    if shortfall <= 0:
        return pd.DataFrame()

    cand = leverage_data[leverage_data["Dest_BuilderRegionKey"] == focus_builder].copy()
    if cand.empty:
        return pd.DataFrame()

    out = []
    for _, r in cand.iterrows():
        source = r["MediaPayer_BuilderRegionKey"]
        tr = float(r.get("Transfer_Rate_Smoothed", np.nan))
        cpr_base = float(r.get("CPR_base", np.nan))
        ecpr = float(r.get("eCPR", np.nan))

        if (not np.isfinite(tr)) or tr <= 0:
            continue
        if (not np.isfinite(ecpr)) or ecpr <= 0:
            continue

        inv = shortfall * ecpr
        total_leads = shortfall / tr
        excess = max(0.0, total_leads - shortfall)

        out.append({
            "Source_Builder": source,
            "Transfer_Rate": tr,
            "Base_CPR": cpr_base,
            "Effective_CPR": ecpr,
            "Investment_Required": inv,
            "Total_Leads_Generated": total_leads,
            "Excess_Leads": excess,
            "Evidence_Referrals_to_Target": int(r.get("Referrals_to_Dest", 0)),
            "Confidence": float(r.get("Confidence", np.nan)),
            "Pass_Min_Events": bool(r.get("Pass_Min_Events", False)),
        })

    res = pd.DataFrame(out)
    if res.empty:
        return res
    return res.sort_values(["Effective_CPR", "Transfer_Rate"], ascending=[True, False]).reset_index(drop=True)


# =============================================================================
# 4) Portfolio optimiser (greedy): closes all forecast shortfalls at min cost
# =============================================================================

def optimise_portfolio_media_plan(
    demand_df: pd.DataFrame,
    leverage_df: pd.DataFrame,
    *,
    cfg: Optional[OptimConfig] = None,
    max_spend_multiplier: float = 1.0,
    new_money: float = 0.0,
    overserve_penalty: float = 0.25,
) -> Dict[str, pd.DataFrame]:
    """
    Greedy portfolio allocation:
      - Spend on source s generates leads = spend / CPR_base
      - Leads distribute by Transfer_Rate_Smoothed to destinations
      - Select next best source by (weighted deficit coverage per $) - penalty(spillover into overserved)

    Outputs:
      demand_table, source_budgets, plan_by_source, plan_edges, post_state, allocation_log
    """
    cfg = cfg or OptimConfig()

    if demand_df is None or demand_df.empty:
        return {"error": pd.DataFrame({"error": ["empty demand_df"]})}
    if leverage_df is None or leverage_df.empty:
        return {"error": pd.DataFrame({"error": ["empty leverage_df"]})}

    dem = demand_df.copy()
    dem["Shortfall"] = dem["Shortfall"].fillna(0.0)

    deficits = dem[dem["Shortfall"] > 0].copy()
    if deficits.empty:
        return {
            "demand_table": dem,
            "source_budgets": pd.DataFrame(),
            "plan_by_source": pd.DataFrame(),
            "plan_edges": pd.DataFrame(),
            "post_state": dem,
            "allocation_log": pd.DataFrame(),
        }

    # weights
    w = deficits.set_index("BuilderRegionKey")["DemandScore"].fillna(deficits["Shortfall"]).to_dict()
    rem = deficits.set_index("BuilderRegionKey")["Shortfall"].to_dict()

    # overserved penalty weights
    over_base = dem.set_index("BuilderRegionKey")
    over_amt = (over_base["Projected_Finish"] - over_base["LeadTarget"] * (1.0 + cfg.overserve_tolerance)).fillna(0.0).clip(lower=0.0)
    over_w = _safe_div(over_amt, np.maximum(over_base["LeadTarget"].fillna(0.0), 1.0), fill=0.0)
    over_w = over_w.to_dict()

    # Source CPRs + baseline budgets
    src = (
        leverage_df[["MediaPayer_BuilderRegionKey", "CPR_base", "Total_Media_Spend", "Total_Referrals_Sent"]]
        .drop_duplicates("MediaPayer_BuilderRegionKey")
        .copy()
    )
    src = src[np.isfinite(src["CPR_base"]) & (src["CPR_base"] > 0)].copy()
    if src.empty:
        return {"error": pd.DataFrame({"error": ["no valid sources with CPR_base"]})}

    budgets = src[["MediaPayer_BuilderRegionKey", "Total_Media_Spend", "CPR_base"]].copy()
    budgets["Baseline_Spend"] = budgets["Total_Media_Spend"].fillna(0.0).astype(float)
    budgets["Budget"] = budgets["Baseline_Spend"] * float(max_spend_multiplier)

    if new_money and new_money > 0:
        base = budgets["Budget"].copy()
        if base.sum() <= 0:
            budgets["Budget"] += new_money / len(budgets)
        else:
            budgets["Budget"] += new_money * (base / base.sum())

    budgets["Budget_Remaining"] = budgets["Budget"]

    # Dist per source (normalised)
    T = leverage_df[["MediaPayer_BuilderRegionKey", "Dest_BuilderRegionKey", "Transfer_Rate_Smoothed"]].copy()

    dist: Dict[str, Dict[str, float]] = {}
    for s, grp in T.groupby("MediaPayer_BuilderRegionKey"):
        d = grp.set_index("Dest_BuilderRegionKey")["Transfer_Rate_Smoothed"].to_dict()
        tot = sum(v for v in d.values() if np.isfinite(v) and v > 0)
        if tot > 0:
            d = {k: float(v) / tot for k, v in d.items() if np.isfinite(v) and v > 0}
        else:
            d = {}
        dist[s] = d

    cpr_map = budgets.set_index("MediaPayer_BuilderRegionKey")["CPR_base"].to_dict()
    budget_rem = budgets.set_index("MediaPayer_BuilderRegionKey")["Budget_Remaining"].to_dict()

    spend_alloc = {s: 0.0 for s in budget_rem.keys()}
    leads_added = {b: 0.0 for b in rem.keys()}
    allocation_log = []

    def score_source(s: str) -> float:
        cpr = cpr_map.get(s, np.nan)
        if not np.isfinite(cpr) or cpr <= 0:
            return -np.inf
        if budget_rem.get(s, 0.0) <= 0:
            return -np.inf

        d = dist.get(s, {})
        if not d:
            return -np.inf

        benefit = 0.0
        for b, need in rem.items():
            if need <= 0:
                continue
            p = d.get(b, 0.0)
            if p <= 0:
                continue
            benefit += w.get(b, need) * (p / cpr)

        penalty = 0.0
        for b, ow in over_w.items():
            if ow <= 0:
                continue
            p = d.get(b, 0.0)
            if p <= 0:
                continue
            penalty += ow * (p / cpr)

        return benefit - float(overserve_penalty) * penalty

    it = 0
    while True:
        it += 1
        if all(v <= 0 for v in rem.values()):
            break

        scored = [(s, score_source(s)) for s in budget_rem.keys()]
        scored.sort(key=lambda x: x[1], reverse=True)
        s_best, best_score = scored[0]

        if not np.isfinite(best_score) or best_score <= 0:
            break

        cpr = cpr_map[s_best]
        d = dist.get(s_best, {})

        # spend required to close at least one remaining deficit this source touches
        spend_to_close = []
        for b, need in rem.items():
            if need <= 0:
                continue
            p = d.get(b, 0.0)
            if p <= 0:
                continue
            spend_to_close.append(need * cpr / p)

        if not spend_to_close:
            budget_rem[s_best] = 0.0
            continue

        spend = min(
            budget_rem[s_best],
            max(cfg.min_step_spend, min(spend_to_close)),
            cfg.max_step_spend,
        )

        spend_alloc[s_best] += spend
        budget_rem[s_best] -= spend

        total_leads = spend / cpr

        for b, p in d.items():
            add = total_leads * p
            if b in rem:
                rem[b] = max(0.0, rem[b] - add)
                leads_added[b] += add

        allocation_log.append({
            "iter": it,
            "source": s_best,
            "score": best_score,
            "spend": spend,
            "cpr_base": cpr,
            "leads_generated": total_leads,
            "budget_remaining": budget_rem[s_best],
        })

        if it > 2000:
            break

    # plan by source
    rows = []
    for s, spend in spend_alloc.items():
        if spend <= 0:
            continue
        cpr = cpr_map.get(s, np.nan)
        leads = spend / cpr if (np.isfinite(cpr) and cpr > 0) else np.nan
        rows.append({
            "MediaPayer_BuilderRegionKey": s,
            "Spend": spend,
            "CPR_base": cpr,
            "Leads_Generated": leads,
            "Budget": float(budgets.set_index("MediaPayer_BuilderRegionKey").loc[s, "Budget"]) if s in budgets["MediaPayer_BuilderRegionKey"].values else np.nan,
        })
    plan_by_source = pd.DataFrame(rows).sort_values("Spend", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()

    # plan edges (source->dest contributions)
    edges = []
    for s, spend in spend_alloc.items():
        if spend <= 0:
            continue
        cpr = cpr_map.get(s, np.nan)
        if not np.isfinite(cpr) or cpr <= 0:
            continue
        total_leads = spend / cpr
        d = dist.get(s, {})
        for b, p in d.items():
            exp_leads = total_leads * p
            if exp_leads <= 0:
                continue
            edges.append({
                "Source": s,
                "Dest": b,
                "Transfer_Rate_Smoothed": p,
                "Base_CPR": cpr,
                "Effective_CPR_to_Dest": cpr / max(p, 1e-9),
                "Spend_on_Source": spend,
                "Expected_Leads": exp_leads,
            })
    plan_edges = pd.DataFrame(edges).sort_values("Expected_Leads", ascending=False).reset_index(drop=True) if edges else pd.DataFrame()

    # post-state reconciliation
    post = dem.copy()
    add_map = {k: float(v) for k, v in leads_added.items()}
    post["Expected_Leads_Added"] = post["BuilderRegionKey"].map(add_map).fillna(0.0)
    post["Projected_Finish_Post"] = post["Projected_Finish"] + post["Expected_Leads_Added"]
    post["Shortfall_Post"] = _clip_pos(post["LeadTarget"] - post["Projected_Finish_Post"])
    post["Overserve_Post"] = _clip_pos(post["Projected_Finish_Post"] - post["LeadTarget"] * (1.0 + cfg.overserve_tolerance))

    return {
        "demand_table": dem,
        "source_budgets": budgets,
        "plan_by_source": plan_by_source,
        "plan_edges": plan_edges,
        "post_state": post,
        "allocation_log": pd.DataFrame(allocation_log),
    }
