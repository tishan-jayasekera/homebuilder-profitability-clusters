"""
Network Optimization Engine - Enhanced for Campaign Planning
Supports targeted optimization for user-selected builder groups.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple

def calculate_shortfalls(
    events_df: pd.DataFrame, 
    targets_df: pd.DataFrame = None, 
    period_days: int = None, 
    total_events_df: pd.DataFrame = None,
    scenario_params: dict = None
) -> pd.DataFrame:
    """Calculate Demand (Shortfalls) with Pace & Projection."""
    if scenario_params is None:
        scenario_params = {}
    
    velocity_mult = scenario_params.get('velocity_mult', 1.0)
    target_mult = scenario_params.get('target_mult', 1.0)

    period_actuals = events_df[events_df['is_referral'] == True].groupby('Dest_BuilderRegionKey').size().reset_index(name='Period_Referrals')
    
    if total_events_df is not None:
        cum_actuals = total_events_df[total_events_df['is_referral'] == True].groupby('Dest_BuilderRegionKey').size().reset_index(name='Actual_Referrals')
        target_source_df = total_events_df 
    else:
        cum_actuals = period_actuals.rename(columns={'Period_Referrals': 'Actual_Referrals'})
        target_source_df = events_df

    if targets_df is None:
        if 'Dest_BuilderRegionKey' in target_source_df.columns:
            builders = target_source_df['Dest_BuilderRegionKey'].dropna().unique()
        else:
            builders = []

        cols_to_check = ['LeadTarget_from_job', 'WIP_JOB_LIVE_END']
        
        if all(c in target_source_df.columns for c in cols_to_check):
            targets_df = target_source_df[['Dest_BuilderRegionKey', 'LeadTarget_from_job', 'WIP_JOB_LIVE_END']].drop_duplicates('Dest_BuilderRegionKey').copy()
            targets_df = targets_df.rename(columns={'Dest_BuilderRegionKey': 'BuilderRegionKey', 'LeadTarget_from_job': 'LeadTarget'})
        else:
            targets_df = pd.DataFrame({
                'BuilderRegionKey': builders,
                'LeadTarget': 50, 
                'WIP_JOB_LIVE_END': pd.Timestamp.now() + pd.Timedelta(days=90)
            })
    else:
        rename_map = {'LeadTarget_from_job': 'LeadTarget', 'Builder': 'BuilderRegionKey'}
        targets_df = targets_df.rename(columns=rename_map)
    
    targets_df['LeadTarget'] = targets_df['LeadTarget'] * target_mult

    df = targets_df.merge(cum_actuals, left_on='BuilderRegionKey', right_on='Dest_BuilderRegionKey', how='left')
    df['Actual_Referrals'] = df['Actual_Referrals'].fillna(0)
    
    df = df.merge(period_actuals, left_on='BuilderRegionKey', right_on='Dest_BuilderRegionKey', how='left', suffixes=('', '_period'))
    df['Period_Referrals'] = df['Period_Referrals'].fillna(0)
    
    now = pd.Timestamp.now()
    if 'WIP_JOB_LIVE_END' in df.columns:
        df['WIP_JOB_LIVE_END'] = pd.to_datetime(df['WIP_JOB_LIVE_END'], errors='coerce')
        df['Days_Remaining'] = (df['WIP_JOB_LIVE_END'] - now).dt.days.fillna(0).astype(int)
    else:
        df['Days_Remaining'] = 30

    df['Days_Remaining'] = df['Days_Remaining'].clip(lower=0)
    
    if period_days:
        velocity_days = max(period_days, 1)
    else:
        date_col = 'lead_date' if 'lead_date' in events_df.columns else 'RefDate'
        if date_col in events_df.columns and not events_df.empty:
            dates = pd.to_datetime(events_df[date_col], errors='coerce')
            velocity_days = (dates.max() - dates.min()).days
            velocity_days = max(velocity_days, 1)
        else:
            velocity_days = 90
            
    base_velocity = df['Period_Referrals'] / velocity_days
    df['Velocity_LeadsPerDay'] = base_velocity * velocity_mult
    
    df['Projected_Additional'] = df['Velocity_LeadsPerDay'] * df['Days_Remaining']
    df['Projected_Total'] = df['Actual_Referrals'] + df['Projected_Additional']
    
    df['Net_Gap'] = df['Projected_Total'] - df['LeadTarget']
    df['Projected_Shortfall'] = np.where(df['Net_Gap'] < 0, abs(df['Net_Gap']), 0)
    df['Projected_Surplus'] = np.where(df['Net_Gap'] > 0, df['Net_Gap'], 0)
    
    df['CatchUp_Pace_Req'] = np.where(
        (df['Projected_Shortfall'] > 0) & (df['Days_Remaining'] > 0),
        df['Projected_Shortfall'] / df['Days_Remaining'], 0
    )
    
    df['Risk_Score'] = ((df['Projected_Shortfall'] * 5) + (df['CatchUp_Pace_Req'] * 20)).fillna(0)
    
    return df


def analyze_network_leverage(events_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze Supply Sources - Transfer Rate, CPR, eCPR."""
    refs = events_df[events_df['is_referral'] == True].copy()
    if refs.empty:
        return pd.DataFrame()

    source_stats = refs.groupby('MediaPayer_BuilderRegionKey').agg(
        Total_Referrals_Sent=('LeadId', 'count'),
        Total_Media_Spend=('MediaCost_referral_event', 'sum')
    ).reset_index()
    
    source_stats['CPR_base'] = np.where(
        source_stats['Total_Referrals_Sent'] > 0,
        source_stats['Total_Media_Spend'] / source_stats['Total_Referrals_Sent'],
        np.nan
    )
    
    flows = refs.groupby(['MediaPayer_BuilderRegionKey', 'Dest_BuilderRegionKey']).size().reset_index(name='Referrals_to_Target')
    
    leverage = flows.merge(source_stats, on='MediaPayer_BuilderRegionKey', how='left')
    leverage['Transfer_Rate'] = leverage['Referrals_to_Target'] / leverage['Total_Referrals_Sent']
    leverage['eCPR'] = np.where(leverage['Transfer_Rate'] > 0, leverage['CPR_base'] / leverage['Transfer_Rate'], np.inf)
    
    return leverage


def build_flow_matrix(events_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Build directed flow matrix with reciprocity detection."""
    refs = events_df[events_df['is_referral'] == True].copy()
    if refs.empty:
        return pd.DataFrame(), {}
    
    flows = refs.groupby(['MediaPayer_BuilderRegionKey', 'Dest_BuilderRegionKey']).agg(
        flow_count=('LeadId', 'count'),
        flow_value=('MediaCost_referral_event', 'sum')
    ).reset_index()
    
    flows.columns = ['source', 'target', 'flow_count', 'flow_value']
    
    # Detect reciprocal flows
    reciprocal_pairs = {}
    for _, row in flows.iterrows():
        s, t = row['source'], row['target']
        reverse = flows[(flows['source'] == t) & (flows['target'] == s)]
        if not reverse.empty:
            pair_key = tuple(sorted([s, t]))
            if pair_key not in reciprocal_pairs:
                reciprocal_pairs[pair_key] = {
                    'nodes': (s, t),
                    'forward': row['flow_count'],
                    'reverse': reverse['flow_count'].iloc[0],
                    'total': row['flow_count'] + reverse['flow_count'].iloc[0]
                }
    
    flows['is_reciprocal'] = flows.apply(
        lambda r: tuple(sorted([r['source'], r['target']])) in reciprocal_pairs, axis=1
    )
    
    return flows, reciprocal_pairs


def get_builder_ego_network(
    events_df: pd.DataFrame, 
    builder_key: str, 
    depth: int = 1
) -> Dict:
    """Extract ego network for a specific builder."""
    refs = events_df[events_df['is_referral'] == True].copy()
    if refs.empty:
        return {'nodes': [], 'edges': [], 'center': builder_key}
    
    # Direct connections (depth 1)
    inbound = refs[refs['Dest_BuilderRegionKey'] == builder_key]['MediaPayer_BuilderRegionKey'].unique().tolist()
    outbound = refs[refs['MediaPayer_BuilderRegionKey'] == builder_key]['Dest_BuilderRegionKey'].unique().tolist()
    
    connected = set(inbound + outbound + [builder_key])
    
    if depth > 1:
        # Second degree connections
        for node in list(connected):
            if node == builder_key:
                continue
            second_in = refs[refs['Dest_BuilderRegionKey'] == node]['MediaPayer_BuilderRegionKey'].unique().tolist()
            second_out = refs[refs['MediaPayer_BuilderRegionKey'] == node]['Dest_BuilderRegionKey'].unique().tolist()
            connected.update(second_in[:5])  # Limit to prevent explosion
            connected.update(second_out[:5])
    
    # Build edges within ego network
    mask = (refs['MediaPayer_BuilderRegionKey'].isin(connected)) & (refs['Dest_BuilderRegionKey'].isin(connected))
    ego_refs = refs[mask]
    
    edges = ego_refs.groupby(['MediaPayer_BuilderRegionKey', 'Dest_BuilderRegionKey']).size().reset_index(name='weight')
    edges.columns = ['source', 'target', 'weight']
    
    # Classify node roles
    nodes = []
    for n in connected:
        role = 'center' if n == builder_key else ('inbound' if n in inbound else ('outbound' if n in outbound else 'secondary'))
        nodes.append({'id': n, 'role': role})
    
    return {
        'nodes': nodes,
        'edges': edges.to_dict('records'),
        'center': builder_key,
        'inbound_count': len(inbound),
        'outbound_count': len(outbound)
    }


def generate_targeted_media_plan(
    target_builders: List[str],
    shortfall_df: pd.DataFrame,
    leverage_df: pd.DataFrame,
    budget_cap: float = None
) -> pd.DataFrame:
    """Generate optimized media plan for specific target builders."""
    if not target_builders:
        return pd.DataFrame()
    
    plan_rows = []
    remaining_budget = budget_cap if budget_cap else float('inf')
    
    # Filter to requested targets with shortfalls
    targets = shortfall_df[
        (shortfall_df['BuilderRegionKey'].isin(target_builders)) &
        (shortfall_df['Projected_Shortfall'] > 0)
    ].copy()
    
    if targets.empty:
        # Return info even for targets with no shortfall
        for t in target_builders:
            row = shortfall_df[shortfall_df['BuilderRegionKey'] == t]
            if not row.empty:
                plan_rows.append({
                    'Target_Builder': t,
                    'Status': '✅ On Track',
                    'Projected_Surplus': row['Projected_Surplus'].iloc[0],
                    'Recommended_Source': '-',
                    'Budget_Allocation': 0,
                    'Projected_Leads': 0,
                    'Effective_CPR': 0,
                    'Strategy': 'No intervention needed'
                })
        return pd.DataFrame(plan_rows)
    
    # Sort by risk (highest first)
    targets = targets.sort_values('Risk_Score', ascending=False)
    
    for _, builder_row in targets.iterrows():
        target = builder_row['BuilderRegionKey']
        gap = builder_row['Projected_Shortfall']
        risk = builder_row['Risk_Score']
        
        # Find leverage paths to this target
        candidates = leverage_df[leverage_df['Dest_BuilderRegionKey'] == target].copy()
        
        if candidates.empty:
            plan_rows.append({
                'Target_Builder': target,
                'Status': '⚠️ No Path',
                'Gap_Leads': gap,
                'Risk_Score': risk,
                'Recommended_Source': 'COLD START REQUIRED',
                'Budget_Allocation': np.nan,
                'Projected_Leads': 0,
                'Effective_CPR': np.nan,
                'Strategy': 'Establish new referral partnership'
            })
            continue
        
        # Sort by efficiency (lowest eCPR first)
        candidates = candidates[candidates['eCPR'] < np.inf].sort_values('eCPR')
        
        if candidates.empty:
            plan_rows.append({
                'Target_Builder': target,
                'Status': '⚠️ Inefficient Paths',
                'Gap_Leads': gap,
                'Risk_Score': risk,
                'Recommended_Source': 'REVIEW SOURCES',
                'Budget_Allocation': np.nan,
                'Projected_Leads': 0,
                'Effective_CPR': np.nan,
                'Strategy': 'Current paths have infinite eCPR'
            })
            continue
        
        # Allocate budget across sources to fill gap
        leads_needed = gap
        builder_allocations = []
        
        for _, source_row in candidates.iterrows():
            if leads_needed <= 0 or remaining_budget <= 0:
                break
            
            source = source_row['MediaPayer_BuilderRegionKey']
            ecpr = source_row['eCPR']
            tr = source_row['Transfer_Rate']
            
            # Calculate how much to allocate to this source
            cost_for_gap = leads_needed * ecpr
            allocation = min(cost_for_gap, remaining_budget)
            leads_from_source = allocation / ecpr if ecpr > 0 else 0
            
            builder_allocations.append({
                'source': source,
                'allocation': allocation,
                'leads': leads_from_source,
                'ecpr': ecpr,
                'tr': tr
            })
            
            leads_needed -= leads_from_source
            remaining_budget -= allocation
        
        # Add to plan
        for alloc in builder_allocations:
            status = '🔴 Critical' if risk > 50 else ('🟠 High' if risk > 25 else '🟡 Medium')
            plan_rows.append({
                'Target_Builder': target,
                'Status': status,
                'Gap_Leads': gap,
                'Risk_Score': risk,
                'Recommended_Source': alloc['source'],
                'Budget_Allocation': alloc['allocation'],
                'Projected_Leads': alloc['leads'],
                'Effective_CPR': alloc['ecpr'],
                'Transfer_Rate': alloc['tr'],
                'Strategy': f"Scale media via {alloc['source'][:20]}..."
            })
    
    return pd.DataFrame(plan_rows)


def analyze_network_health(events_df: pd.DataFrame) -> pd.DataFrame:
    """Network Health Analysis - Zombie Nodes, Bridge Nodes."""
    if events_df.empty:
        return pd.DataFrame()
    
    receivers = events_df[events_df['is_referral'] == True]['Dest_BuilderRegionKey'].value_counts().reset_index()
    receivers.columns = ['Builder', 'Leads_Received']
    
    senders = events_df[events_df['is_referral'] == True]['MediaPayer_BuilderRegionKey'].value_counts().reset_index()
    senders.columns = ['Builder', 'Leads_Sent']
    
    health = pd.merge(receivers, senders, on='Builder', how='outer').fillna(0)
    health['Ratio_Give_Take'] = np.where(health['Leads_Received'] > 0, health['Leads_Sent'] / health['Leads_Received'], 0)
    
    def diagnose(row):
        if row['Leads_Received'] > 10 and row['Leads_Sent'] == 0:
            return "🧟 Zombie (Dead End)"
        if row['Leads_Sent'] > 10 and row['Leads_Received'] == 0:
            return "📡 Feeder Only"
        if row['Leads_Sent'] > 5 and row['Leads_Received'] > 5:
            return "🔄 Healthy Hub"
        return "⚪ Low Volume"
        
    health['Role'] = health.apply(diagnose, axis=1)
    health['Health_Score'] = np.clip(health['Ratio_Give_Take'] * 20 + np.minimum(health['Leads_Sent'], 50), 0, 100)
    
    return health


def generate_investment_strategies(
    focus_builder: str, 
    shortfall_data: pd.DataFrame, 
    leverage_data: pd.DataFrame, 
    events_df: pd.DataFrame
) -> pd.DataFrame:
    """Generate detailed investment strategies for a single builder."""
    target_row = shortfall_data[shortfall_data['BuilderRegionKey'] == focus_builder]
    if target_row.empty:
        return pd.DataFrame()
    
    col = 'Projected_Shortfall' if 'Projected_Shortfall' in target_row.columns else 'Shortfall'
    shortfall = target_row[col].iloc[0]
    
    strategies = leverage_data[leverage_data['Dest_BuilderRegionKey'] == focus_builder].copy()
    if strategies.empty:
        return pd.DataFrame()
    
    results = []
    
    for _, strat in strategies.iterrows():
        source = strat['MediaPayer_BuilderRegionKey']
        tr = strat['Transfer_Rate']
        ecpr = strat['eCPR']
        
        if shortfall > 0 and tr > 0:
            investment = shortfall * ecpr
            leads_gen = shortfall / tr
            excess = leads_gen - shortfall
        else:
            investment, leads_gen, excess = 0, 0, 0
            
        results.append({
            'Source_Builder': source,
            'Transfer_Rate': tr,
            'Base_CPR': strat['CPR_base'],
            'Effective_CPR': ecpr,
            'Investment_Required': investment,
            'Total_Leads_Generated': leads_gen,
            'Excess_Leads': excess
        })
        
    res = pd.DataFrame(results)
    if not res.empty:
        res = res.sort_values('Effective_CPR')
    return res


def get_cluster_members(events_df: pd.DataFrame, builder_key: str, partition: Dict) -> List[str]:
    """Get all builders in the same cluster as the target."""
    if builder_key not in partition:
        return [builder_key]
    
    target_cluster = partition[builder_key]
    return [b for b, c in partition.items() if c == target_cluster]


def calculate_campaign_summary(plan_df: pd.DataFrame) -> Dict:
    """Calculate summary statistics for a campaign plan."""
    if plan_df.empty:
        return {'total_budget': 0, 'total_leads': 0, 'avg_cpr': 0, 'targets_covered': 0}
    
    total_budget = plan_df['Budget_Allocation'].sum()
    total_leads = plan_df['Projected_Leads'].sum()
    avg_cpr = total_budget / total_leads if total_leads > 0 else 0
    targets_covered = plan_df['Target_Builder'].nunique()
    
    return {
        'total_budget': total_budget,
        'total_leads': total_leads,
        'avg_cpr': avg_cpr,
        'targets_covered': targets_covered,
        'sources_used': plan_df['Recommended_Source'].nunique()
    }