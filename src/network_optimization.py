import pandas as pd
import numpy as np

def calculate_shortfalls(
    events_df: pd.DataFrame, 
    targets_df: pd.DataFrame = None, 
    period_days: int = None, 
    total_events_df: pd.DataFrame = None,
    scenario_params: dict = None
) -> pd.DataFrame:
    """
    Step 1: Calculate Demand (Shortfalls) with Pace & Projection.
    Now supports Scenario Planning modifiers.
    """
    if scenario_params is None:
        scenario_params = {}
    
    velocity_mult = scenario_params.get('velocity_mult', 1.0)
    target_mult = scenario_params.get('target_mult', 1.0)

    # 1. Period Actuals (For Velocity)
    period_actuals = events_df[events_df['is_referral'] == True].groupby('Dest_BuilderRegionKey').size().reset_index(name='Period_Referrals')
    
    # 2. Cumulative Actuals (For Progress)
    if total_events_df is not None:
        cum_actuals = total_events_df[total_events_df['is_referral'] == True].groupby('Dest_BuilderRegionKey').size().reset_index(name='Actual_Referrals')
        target_source_df = total_events_df 
    else:
        cum_actuals = period_actuals.rename(columns={'Period_Referrals': 'Actual_Referrals'})
        target_source_df = events_df

    # 3. Targets: Retrieve or Mock
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
    
    # Apply Scenario: Target Multiplier
    targets_df['LeadTarget'] = targets_df['LeadTarget'] * target_mult

    # 4. Merge Targets with Cumulative Actuals
    df = targets_df.merge(cum_actuals, left_on='BuilderRegionKey', right_on='Dest_BuilderRegionKey', how='left')
    df['Actual_Referrals'] = df['Actual_Referrals'].fillna(0)
    
    # 5. Merge Period Actuals for Pace calculation
    df = df.merge(period_actuals, left_on='BuilderRegionKey', right_on='Dest_BuilderRegionKey', how='left', suffixes=('', '_period'))
    df['Period_Referrals'] = df['Period_Referrals'].fillna(0)
    
    # 6. Time Components & Pace
    now = pd.Timestamp.now()
    if 'WIP_JOB_LIVE_END' in df.columns:
        df['WIP_JOB_LIVE_END'] = pd.to_datetime(df['WIP_JOB_LIVE_END'], errors='coerce')
        df['Days_Remaining'] = (df['WIP_JOB_LIVE_END'] - now).dt.days.fillna(0).astype(int)
    else:
        df['Days_Remaining'] = 30 # Default assumption

    df['Days_Remaining'] = df['Days_Remaining'].clip(lower=0)
    
    # Determine Velocity
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
    
    # Apply Scenario: Velocity Multiplier
    df['Velocity_LeadsPerDay'] = base_velocity * velocity_mult
    
    # 7. Projection
    df['Projected_Additional'] = df['Velocity_LeadsPerDay'] * df['Days_Remaining']
    df['Projected_Total'] = df['Actual_Referrals'] + df['Projected_Additional']
    
    # 8. Gap Analysis
    df['Net_Gap'] = df['Projected_Total'] - df['LeadTarget']
    df['Projected_Shortfall'] = np.where(df['Net_Gap'] < 0, abs(df['Net_Gap']), 0)
    df['Projected_Surplus'] = np.where(df['Net_Gap'] > 0, df['Net_Gap'], 0)
    
    # 9. Risk Scoring
    df['CatchUp_Pace_Req'] = np.where(
        (df['Projected_Shortfall'] > 0) & (df['Days_Remaining'] > 0),
        df['Projected_Shortfall'] / df['Days_Remaining'],
        0
    )
    
    df['Risk_Score'] = (
        (df['Projected_Shortfall'] * 5) + 
        (df['CatchUp_Pace_Req'] * 20)
    ).fillna(0)
    
    return df


def analyze_network_leverage(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Analyze Supply Sources (The Leverage).
    Calculates Transfer Rate (TR), Base CPR (CPR_base), and Effective CPR (eCPR).
    """
    refs = events_df[events_df['is_referral'] == True].copy()
    if refs.empty: return pd.DataFrame()

    # A. Source Metrics
    source_stats = refs.groupby('MediaPayer_BuilderRegionKey').agg(
        Total_Referrals_Sent=('LeadId', 'count'),
        Total_Media_Spend=('MediaCost_referral_event', 'sum')
    ).reset_index()
    
    source_stats['CPR_base'] = np.where(
        source_stats['Total_Referrals_Sent'] > 0,
        source_stats['Total_Media_Spend'] / source_stats['Total_Referrals_Sent'],
        np.nan
    )
    
    # B. Flow Metrics
    flows = refs.groupby(['MediaPayer_BuilderRegionKey', 'Dest_BuilderRegionKey']).size().reset_index(name='Referrals_to_Target')
    
    # C. Merge & Calculate TR / eCPR
    leverage = flows.merge(source_stats, on='MediaPayer_BuilderRegionKey', how='left')
    
    leverage['Transfer_Rate'] = leverage['Referrals_to_Target'] / leverage['Total_Referrals_Sent']
    
    leverage['eCPR'] = np.where(
        leverage['Transfer_Rate'] > 0,
        leverage['CPR_base'] / leverage['Transfer_Rate'],
        np.inf
    )
    
    return leverage


def generate_global_media_plan(
    shortfall_df: pd.DataFrame, 
    leverage_df: pd.DataFrame,
    strict_capacity: bool = False
) -> pd.DataFrame:
    """
    Step 3: Global Media Planning.
    Matches EVERY shortfall builder to their most efficient available leverage points.
    """
    # Simply reuse targeted logic for all deficit builders
    deficits = shortfall_df[shortfall_df['Projected_Shortfall'] > 0]['BuilderRegionKey'].unique().tolist()
    return optimize_campaign_spend(deficits, shortfall_df, leverage_df, strict_capacity)


def optimize_campaign_spend(
    target_builders: list,
    shortfall_df: pd.DataFrame,
    leverage_df: pd.DataFrame,
    strict_capacity: bool = False
) -> pd.DataFrame:
    """
    Optimizes media spend specifically for a list of target builders.
    
    Args:
        target_builders: List of builder IDs to optimize for.
        shortfall_df: Dataframe with gap and surplus info.
        leverage_df: Dataframe with eCPR and source info.
        strict_capacity: If True, only uses sources with surplus.
        
    Returns:
        DataFrame representing the media plan.
    """
    plan_rows = []
    
    # Identify Surplus Builders (The Supply Registry)
    surplus_builders = set(shortfall_df[shortfall_df['Projected_Surplus'] > 0]['BuilderRegionKey'].tolist())
    
    # Get shortfall data for targets
    target_data = shortfall_df[shortfall_df['BuilderRegionKey'].isin(target_builders)].copy()
    
    for _, builder_row in target_data.iterrows():
        target = builder_row['BuilderRegionKey']
        gap = builder_row['Projected_Shortfall']
        risk = builder_row['Risk_Score']
        
        # If no gap, skip or note
        if gap <= 0:
            plan_rows.append({
                'Target_Builder': target,
                'Status': 'On Track',
                'Gap_Leads': 0,
                'Recommended_Source': '-',
                'Est_Investment': 0,
                'Strategy_Note': 'Builder is hitting targets.'
            })
            continue

        # 1. Find candidates
        candidates = leverage_df[leverage_df['Dest_BuilderRegionKey'] == target].copy()
        
        if candidates.empty:
            plan_rows.append({
                'Target_Builder': target,
                'Status': 'Cold Start',
                'Gap_Leads': gap,
                'Recommended_Source': 'NO HISTORICAL DATA',
                'Est_Investment': np.nan,
                'Effective_CPR': np.nan,
                'Strategy_Note': 'No existing inbound paths found.'
            })
            continue

        # 2. Capacity Constraint
        if strict_capacity:
            valid_candidates = candidates[candidates['MediaPayer_BuilderRegionKey'].isin(surplus_builders)].copy()
            if valid_candidates.empty:
                # Fallback
                best_source = candidates.sort_values('eCPR', ascending=True).iloc[0]
                plan_rows.append({
                    'Target_Builder': target,
                    'Status': 'Constrained',
                    'Gap_Leads': gap,
                    'Recommended_Source': best_source['MediaPayer_BuilderRegionKey'],
                    'Est_Investment': gap * best_source['eCPR'],
                    'Effective_CPR': best_source['eCPR'],
                    'Strategy_Note': 'WARNING: Best source has no surplus.'
                })
                continue
            else:
                candidates = valid_candidates
        
        # 3. Optimization (Lowest eCPR)
        best_source = candidates.sort_values('eCPR', ascending=True).iloc[0]
        invest_needed = gap * best_source['eCPR']
        
        plan_rows.append({
            'Target_Builder': target,
            'Status': 'Actionable',
            'Gap_Leads': gap,
            'Recommended_Source': best_source['MediaPayer_BuilderRegionKey'],
            'Est_Investment': invest_needed,
            'Effective_CPR': best_source['eCPR'],
            'Strategy_Note': f"Scale {best_source['MediaPayer_BuilderRegionKey']} (eCPR: ${best_source['eCPR']:.0f})"
        })
        
    return pd.DataFrame(plan_rows)


def analyze_network_health(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 4: Network Health Analysis.
    Identifies Zombie Nodes (Takes leads, never gives) and Bridge Nodes.
    """
    if events_df.empty: return pd.DataFrame()
    
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
    
    return health


def generate_investment_strategies(focus_builder: str, shortfall_data: pd.DataFrame, leverage_data: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper for single-builder detailed view."""
    
    target_row = shortfall_data[shortfall_data['BuilderRegionKey'] == focus_builder]
    if target_row.empty: return pd.DataFrame()
    
    col = 'Projected_Shortfall' if 'Projected_Shortfall' in target_row.columns else 'Shortfall'
    shortfall = target_row[col].iloc[0]
    
    strategies = leverage_data[leverage_data['Dest_BuilderRegionKey'] == focus_builder].copy()
    if strategies.empty: return pd.DataFrame()
    
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
            investment = 0; leads_gen = 0; excess = 0
            
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