import pandas as pd
import numpy as np

def calculate_shortfalls(events_df: pd.DataFrame, targets_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Step 1: Calculate Demand (Shortfalls) with Pace & Projection.
    
    Logic:
    - Actuals = Current Leads
    - Pace = Recent Lead Velocity (leads/day over last 60 days)
    - Projected = Actuals + (Pace * Days_Remaining)
    - Shortfall = Target - Projected
    """
    # 1. Actuals: Current_Leads
    actuals = events_df[events_df['is_referral'] == True].groupby('Dest_BuilderRegionKey').size().reset_index(name='Actual_Referrals')
    
    # 2. Targets: Retrieve or Mock
    if targets_df is None:
        builders = events_df['Dest_BuilderRegionKey'].dropna().unique()
        cols_to_check = ['LeadTarget_from_job', 'WIP_JOB_LIVE_END']
        
        if all(c in events_df.columns for c in cols_to_check):
             targets_df = events_df[['Dest_BuilderRegionKey', 'LeadTarget_from_job', 'WIP_JOB_LIVE_END']].drop_duplicates('Dest_BuilderRegionKey').copy()
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
    
    # 3. Merge
    df = targets_df.merge(actuals, left_on='BuilderRegionKey', right_on='Dest_BuilderRegionKey', how='left')
    df['Actual_Referrals'] = df['Actual_Referrals'].fillna(0)
    
    # 4. Time Components & Pace
    now = pd.Timestamp.now()
    if 'WIP_JOB_LIVE_END' in df.columns:
        df['WIP_JOB_LIVE_END'] = pd.to_datetime(df['WIP_JOB_LIVE_END'], errors='coerce')
        df['Days_Remaining'] = (df['WIP_JOB_LIVE_END'] - now).dt.days.fillna(0).astype(int)
    else:
        df['Days_Remaining'] = 30 # Default assumption if missing

    # Ensure no negative days
    df['Days_Remaining'] = df['Days_Remaining'].clip(lower=0)
    
    # Calculate Velocity (Leads per day in last 60 days)
    # Simplified Velocity: Actuals / (Implied Elapsed Days). 
    # Assuming campaign started 90 days ago if no start date.
    elapsed_est = 90 
    df['Velocity_LeadsPerDay'] = df['Actual_Referrals'] / elapsed_est
    
    # 5. Projection
    df['Projected_Additional'] = df['Velocity_LeadsPerDay'] * df['Days_Remaining']
    df['Projected_Total'] = df['Actual_Referrals'] + df['Projected_Additional']
    
    # 6. Gap Analysis
    # Deficit: Expected to miss target
    # Surplus: Expected to exceed target
    df['Net_Gap'] = df['Projected_Total'] - df['LeadTarget']
    
    df['Projected_Shortfall'] = np.where(df['Net_Gap'] < 0, abs(df['Net_Gap']), 0)
    df['Projected_Surplus'] = np.where(df['Net_Gap'] > 0, df['Net_Gap'], 0)
    
    # 7. Risk Scoring
    # Urgency = Shortfall / Days_Remaining (Leads needed per day to catch up)
    df['CatchUp_Pace_Req'] = np.where(
        (df['Projected_Shortfall'] > 0) & (df['Days_Remaining'] > 0),
        df['Projected_Shortfall'] / df['Days_Remaining'],
        0
    )
    
    # Risk Index: High Shortfall + Low Time
    # Normalized score 0-100
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
    
    # eCPR: The actual cost to get a lead to the Specific Target
    leverage['eCPR'] = np.where(
        leverage['Transfer_Rate'] > 0,
        leverage['CPR_base'] / leverage['Transfer_Rate'],
        np.inf
    )
    
    return leverage

def generate_global_media_plan(shortfall_df: pd.DataFrame, leverage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3: Global Media Planning.
    Matches EVERY shortfall builder to their most efficient available leverage points.
    """
    plan_rows = []
    
    # Filter only those with projected shortfall
    deficits = shortfall_df[shortfall_df['Projected_Shortfall'] > 0].copy()
    
    # Sort by Risk (fix critical fires first)
    deficits = deficits.sort_values('Risk_Score', ascending=False)
    
    for _, builder_row in deficits.iterrows():
        target = builder_row['BuilderRegionKey']
        gap = builder_row['Projected_Shortfall']
        
        # Find sources connected to this target
        sources = leverage_df[leverage_df['Dest_BuilderRegionKey'] == target].copy()
        
        if sources.empty:
            # No existing leverage - this is a "Cold Start" problem
            plan_rows.append({
                'Priority': 'Critical' if builder_row['Risk_Score'] > 50 else 'High',
                'Target_Builder': target,
                'Gap_Leads': gap,
                'Recommended_Source': 'NO HISTORICAL PATH',
                'Action': 'Establish New Partnership',
                'Est_Investment': np.nan,
                'Effective_CPR': np.nan,
                'Strategy_Note': 'No historical referral flow found. Direct media or new cluster intro required.'
            })
            continue
            
        # Pick the most efficient source (Lowest eCPR)
        # We could split volume across multiple, but for Executive Summary, pick the "Lead Source"
        best_source = sources.sort_values('eCPR', ascending=True).iloc[0]
        
        invest_needed = gap * best_source['eCPR']
        
        plan_rows.append({
            'Priority': 'Critical' if builder_row['Risk_Score'] > 50 else 'High',
            'Target_Builder': target,
            'Gap_Leads': gap,
            'Recommended_Source': best_source['MediaPayer_BuilderRegionKey'],
            'Action': f"Scale Media on {best_source['MediaPayer_BuilderRegionKey']}",
            'Est_Investment': invest_needed,
            'Effective_CPR': best_source['eCPR'],
            'Strategy_Note': f"Levg: {best_source['Transfer_Rate']:.1%} TR via {best_source['MediaPayer_BuilderRegionKey']}"
        })
        
    return pd.DataFrame(plan_rows)

def compute_effective_network_cpr(events_df, shortfall_df):
    """Legacy helper for compatibility."""
    leverage = analyze_network_leverage(events_df)
    if leverage.empty: return pd.DataFrame()
    
    grouped = leverage.groupby('MediaPayer_BuilderRegionKey').agg({
        'Total_Referrals_Sent': 'first',
        'CPR_base': 'first',
        'Dest_BuilderRegionKey': 'nunique'
    }).rename(columns={'Dest_BuilderRegionKey': 'Beneficiaries_Count', 'CPR_base': 'Raw_CPR'})
    
    grouped['CPR'] = grouped['Raw_CPR']
    return grouped.reset_index().rename(columns={'MediaPayer_BuilderRegionKey': 'BuilderRegionKey'})

def generate_investment_strategies(focus_builder: str, shortfall_data: pd.DataFrame, leverage_data: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper for single-builder detailed view (compatibility)."""
    
    target_row = shortfall_data[shortfall_data['BuilderRegionKey'] == focus_builder]
    if target_row.empty: return pd.DataFrame()
    
    # Use Projected Shortfall if available, else Shortfall (legacy)
    col = 'Projected_Shortfall' if 'Projected_Shortfall' in target_row.columns else 'Shortfall'
    shortfall = target_row[col].iloc[0]
    
    strategies = leverage_data[leverage_data['Dest_BuilderRegionKey'] == focus_builder].copy()
    if strategies.empty: return pd.DataFrame()
    
    results = []
    # FIX: Use the 'col' variable determined above to avoid KeyError when 'Shortfall' is missing
    shortfall_map = shortfall_data.set_index('BuilderRegionKey')[col].to_dict()
    
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
            
        # Spillover
        source_flows = leverage_data[
            (leverage_data['MediaPayer_BuilderRegionKey'] == source) & 
            (leverage_data['Dest_BuilderRegionKey'] != focus_builder)
        ].copy()
        
        spillover_txt = "None"; score = 0
        if not source_flows.empty and excess > 0:
            source_flows['Proj'] = leads_gen * source_flows['Transfer_Rate']
            impacts = []
            for _, r in source_flows.iterrows():
                ben = r['Dest_BuilderRegionKey']
                need = shortfall_map.get(ben, 0)
                if need > 0:
                    score += min(r['Proj'], need)
                    impacts.append(f"{ben}")
            if impacts: spillover_txt = ", ".join(impacts[:2])
            
        results.append({
            'Source_Builder': source,
            'Transfer_Rate': tr,
            'Base_CPR': strat['CPR_base'],
            'Effective_CPR': ecpr,
            'Investment_Required': investment,
            'Total_Leads_Generated': leads_gen,
            'Excess_Leads': excess,
            'Spillover_Impact': spillover_txt,
            'Optimization_Score': score
        })
        
    res = pd.DataFrame(results)
    if not res.empty:
        res = res.sort_values('Effective_CPR')
    return res