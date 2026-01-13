import pandas as pd
import numpy as np

def calculate_shortfalls(events_df: pd.DataFrame, targets_df: pd.DataFrame = None, period_days: int = None, total_events_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Step 1: Calculate Demand (Shortfalls) with Pace & Projection.
    
    Args:
        events_df: Events data for the selected period (used for Velocity/Pace).
        targets_df: Optional dataframe with builder targets.
        period_days: Duration of the selected period in days. Used for velocity calc.
        total_events_df: Optional cumulative events data. Used for Actuals vs Target.
                        If None, events_df is assumed to be cumulative.
    
    Logic:
    - Actuals = Cumulative Leads (from total_events_df if provided, else events_df)
    - Pace = Leads/Day based on events_df (selected period)
    - Projected = Actuals + (Pace * Days_Remaining)
    - Shortfall = Target - Projected
    - Granularity: Automatically drills down to Project (WIP_JOB_MATCHED) and Suburb if available.
    """
    # Determine the source for cumulative data
    if total_events_df is not None:
        cumulative_source = total_events_df
    else:
        cumulative_source = events_df
        
    # --- 0. Determine Granularity & Keys ---
    # Base key
    group_keys = ['Dest_BuilderRegionKey']
    
    # Check for Project Level Data
    has_project = 'WIP_JOB_MATCHED' in cumulative_source.columns
    if has_project:
        group_keys.append('WIP_JOB_MATCHED')
        
    # Check for Suburb Level Data
    has_suburb = 'Suburb' in cumulative_source.columns
    if has_suburb:
        group_keys.append('Suburb')

    # --- 1. Period Actuals (For Velocity) ---
    # Ensure grouping keys exist in period df
    period_keys = [k for k in group_keys if k in events_df.columns]
    
    period_actuals = events_df[events_df['is_referral'] == True].groupby(period_keys).size().reset_index(name='Period_Referrals')
    
    # --- 2. Cumulative Actuals (For Progress) ---
    cum_actuals = cumulative_source[cumulative_source['is_referral'] == True].groupby(group_keys).size().reset_index(name='Actual_Referrals')

    # --- 3. Targets: Retrieve or Mock ---
    if targets_df is None:
        # Extract unique targets from source data
        # We attempt to pull targets at the most granular level available
        cols_needed = ['LeadTarget_from_job', 'WIP_JOB_LIVE_END'] + group_keys
        cols_available = [c for c in cols_needed if c in cumulative_source.columns]
        
        if 'LeadTarget_from_job' in cols_available and 'WIP_JOB_LIVE_END' in cols_available:
             targets_df = cumulative_source[cols_available].drop_duplicates(group_keys).copy()
             targets_df = targets_df.rename(columns={'Dest_BuilderRegionKey': 'BuilderRegionKey', 'LeadTarget_from_job': 'LeadTarget'})
        else:
            # Fallback Mocking (Builder Level Only)
            builders = cumulative_source['Dest_BuilderRegionKey'].dropna().unique()
            targets_df = pd.DataFrame({
                'BuilderRegionKey': builders,
                'LeadTarget': 50, 
                'WIP_JOB_LIVE_END': pd.Timestamp.now() + pd.Timedelta(days=90)
            })
            # If we are mocking, we can't support granular project targets easily, 
            # so granular actuals will roll up to builder via merge unless we broadcast.
            # For now, simplistic merge handles it.
    else:
        rename_map = {'LeadTarget_from_job': 'LeadTarget', 'Builder': 'BuilderRegionKey'}
        targets_df = targets_df.rename(columns=rename_map)
    
    # --- 4. Merge Targets with Cumulative Actuals ---
    # Renaming grouping keys in targets_df to match standard if needed
    if 'BuilderRegionKey' in targets_df.columns and 'Dest_BuilderRegionKey' not in targets_df.columns:
        # If targets_df is builder-level only but we have granular actuals
        if has_project or has_suburb:
            # We merge on BuilderRegionKey
            df = cum_actuals.merge(targets_df, left_on='Dest_BuilderRegionKey', right_on='BuilderRegionKey', how='left')
        else:
            df = targets_df.merge(cum_actuals, left_on='BuilderRegionKey', right_on='Dest_BuilderRegionKey', how='left')
    else:
        # If targets_df has granular keys (e.g. WIP_JOB_MATCHED), merge on full key set
        # Ensure column names align
        merge_keys = [k for k in group_keys if k in targets_df.columns]
        if not merge_keys: 
            merge_keys = ['Dest_BuilderRegionKey'] # Fallback
            
        df = targets_df.merge(cum_actuals, on=merge_keys, how='left')
        
    df['Actual_Referrals'] = df['Actual_Referrals'].fillna(0)
    
    # Ensure standard BuilderRegionKey exists
    if 'BuilderRegionKey' not in df.columns and 'Dest_BuilderRegionKey' in df.columns:
        df['BuilderRegionKey'] = df['Dest_BuilderRegionKey']
    
    # --- 5. Merge Period Actuals for Pace calculation ---
    # Merge on available keys
    common_keys = [k for k in group_keys if k in df.columns and k in period_actuals.columns]
    df = df.merge(period_actuals, on=common_keys, how='left')
    df['Period_Referrals'] = df['Period_Referrals'].fillna(0)
    
    # --- 6. Time Components & Pace ---
    now = pd.Timestamp.now()
    if 'WIP_JOB_LIVE_END' in df.columns:
        df['WIP_JOB_LIVE_END'] = pd.to_datetime(df['WIP_JOB_LIVE_END'], errors='coerce')
        df['Days_Remaining'] = (df['WIP_JOB_LIVE_END'] - now).dt.days.fillna(0).astype(int)
    else:
        df['Days_Remaining'] = 30 

    df['Days_Remaining'] = df['Days_Remaining'].clip(lower=0)
    
    # Velocity
    if period_days:
        velocity_days = max(period_days, 1)
    else:
        # Estimate from data or default
        velocity_days = 90 
            
    df['Velocity_LeadsPerDay'] = df['Period_Referrals'] / velocity_days
    
    # --- 7. Projection ---
    df['Projected_Additional'] = df['Velocity_LeadsPerDay'] * df['Days_Remaining']
    df['Projected_Total'] = df['Actual_Referrals'] + df['Projected_Additional']
    
    # --- 8. Gap Analysis ---
    df['Net_Gap'] = df['Projected_Total'] - df['LeadTarget']
    df['Projected_Shortfall'] = np.where(df['Net_Gap'] < 0, abs(df['Net_Gap']), 0)
    df['Projected_Surplus'] = np.where(df['Net_Gap'] > 0, df['Net_Gap'], 0)
    
    # --- 9. Risk Scoring ---
    df['CatchUp_Pace_Req'] = np.where(
        (df['Projected_Shortfall'] > 0) & (df['Days_Remaining'] > 0),
        df['Projected_Shortfall'] / df['Days_Remaining'],
        0
    )
    
    df['Risk_Score'] = (
        (df['Projected_Shortfall'] * 5) + 
        (df['CatchUp_Pace_Req'] * 20)
    ).fillna(0)

    # --- 10. Extract Region (Optimization Layer) ---
    if 'BuilderRegionKey' in df.columns:
        df['Region'] = df['BuilderRegionKey'].astype(str).apply(
            lambda x: x.split('|')[1].strip() if '|' in x else 'Unknown'
        )
    
    return df

def analyze_network_leverage(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Analyze Supply Sources (The Leverage).
    Calculates Transfer Rate (TR), Base CPR (CPR_base), and Effective CPR (eCPR).
    """
    refs = events_df[events_df['is_referral'] == True].copy()
    if refs.empty: return pd.DataFrame()

    # Determine Grouping for Flows
    # We want to know Source -> Dest. 
    # If Suburb is available, we track Source -> (Dest, Suburb) to optimize geo-targeting.
    flow_keys = ['MediaPayer_BuilderRegionKey', 'Dest_BuilderRegionKey']
    if 'Suburb' in refs.columns:
        flow_keys.append('Suburb')

    # A. Source Metrics (Global per Source)
    source_stats = refs.groupby('MediaPayer_BuilderRegionKey').agg(
        Total_Referrals_Sent=('LeadId', 'count'),
        Total_Media_Spend=('MediaCost_referral_event', 'sum')
    ).reset_index()
    
    source_stats['CPR_base'] = np.where(
        source_stats['Total_Referrals_Sent'] > 0,
        source_stats['Total_Media_Spend'] / source_stats['Total_Referrals_Sent'],
        np.nan
    )
    
    # B. Flow Metrics (Granular)
    flows = refs.groupby(flow_keys).size().reset_index(name='Referrals_to_Target')
    
    # C. Merge & Calculate TR / eCPR
    leverage = flows.merge(source_stats, on='MediaPayer_BuilderRegionKey', how='left')
    
    leverage['Transfer_Rate'] = leverage['Referrals_to_Target'] / leverage['Total_Referrals_Sent']
    
    leverage['eCPR'] = np.where(
        leverage['Transfer_Rate'] > 0,
        leverage['CPR_base'] / leverage['Transfer_Rate'],
        np.inf
    )
    
    return leverage

def generate_global_media_plan(shortfall_df: pd.DataFrame, leverage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3: Global Media Planning.
    Matches EVERY shortfall builder (or project) to their most efficient available leverage points.
    """
    plan_rows = []
    
    # Filter only those with projected shortfall
    deficits = shortfall_df[shortfall_df['Projected_Shortfall'] > 0].copy()
    
    # Sort by Risk
    deficits = deficits.sort_values('Risk_Score', ascending=False)
    
    for _, row in deficits.iterrows():
        target_builder = row['BuilderRegionKey']
        gap = row['Projected_Shortfall']
        target_suburb = row.get('Suburb', None)
        target_project = row.get('WIP_JOB_MATCHED', None)
        
        # --- Intelligent Leverage Matching ---
        # 1. Filter leverage for this builder
        builder_leverage = leverage_df[leverage_df['Dest_BuilderRegionKey'] == target_builder].copy()
        
        # 2. Try to match specific Suburb if available (Geo-Optimization)
        matched_leverage = pd.DataFrame()
        match_type = "Generic"
        
        if target_suburb and 'Suburb' in builder_leverage.columns:
            suburb_match = builder_leverage[builder_leverage['Suburb'] == target_suburb]
            if not suburb_match.empty:
                matched_leverage = suburb_match
                match_type = f"Suburb ({target_suburb})"
        
        # 3. Fallback to Builder-wide leverage
        if matched_leverage.empty:
            matched_leverage = builder_leverage
            match_type = "Builder-Wide"

        if matched_leverage.empty:
            # Cold Start
            plan_rows.append({
                'Priority': 'Critical' if row['Risk_Score'] > 50 else 'High',
                'Target_Builder': target_builder,
                'Project': target_project,
                'Suburb': target_suburb,
                'Gap_Leads': gap,
                'Recommended_Source': 'NO HISTORICAL PATH',
                'Action': 'Establish New Partnership',
                'Est_Investment': np.nan,
                'Effective_CPR': np.nan,
                'Strategy_Note': f'No history found ({match_type}). Direct media required.'
            })
            continue
            
        # Pick Best Source
        best_source = matched_leverage.sort_values('eCPR', ascending=True).iloc[0]
        
        invest_needed = gap * best_source['eCPR']
        
        plan_rows.append({
            'Priority': 'Critical' if row['Risk_Score'] > 50 else 'High',
            'Target_Builder': target_builder,
            'Project': target_project,
            'Suburb': target_suburb,
            'Gap_Leads': gap,
            'Recommended_Source': best_source['MediaPayer_BuilderRegionKey'],
            'Action': f"Scale Media on {best_source['MediaPayer_BuilderRegionKey']}",
            'Est_Investment': invest_needed,
            'Effective_CPR': best_source['eCPR'],
            'Strategy_Note': f"{match_type} Match. TR: {best_source['Transfer_Rate']:.1%}."
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
    
    # Filter for the focus builder
    target_rows = shortfall_data[shortfall_data['BuilderRegionKey'] == focus_builder]
    if target_rows.empty: return pd.DataFrame()
    
    # Sum shortfalls if we have multiple projects/suburbs for this builder
    col = 'Projected_Shortfall' if 'Projected_Shortfall' in target_rows.columns else 'Shortfall'
    total_shortfall = target_rows[col].sum()
    
    strategies = leverage_data[leverage_data['Dest_BuilderRegionKey'] == focus_builder].copy()
    if strategies.empty: return pd.DataFrame()
    
    # If strategies are granular (Suburb), roll them up to Source level for the simple view
    # Weighted average eCPR? Or just best? Let's take best efficiency per source for now.
    strategies = strategies.sort_values('eCPR').drop_duplicates('MediaPayer_BuilderRegionKey')
    
    results = []
    # Map builder needs (simplified for spillover calculation)
    shortfall_map = shortfall_data.groupby('BuilderRegionKey')[col].sum().to_dict()
    
    for _, strat in strategies.iterrows():
        source = strat['MediaPayer_BuilderRegionKey']
        tr = strat['Transfer_Rate']
        ecpr = strat['eCPR']
        
        if total_shortfall > 0 and tr > 0:
            investment = total_shortfall * ecpr
            leads_gen = total_shortfall / tr
            excess = leads_gen - total_shortfall
        else:
            investment = 0; leads_gen = 0; excess = 0
            
        # Spillover logic (simplified)
        source_flows = leverage_data[
            (leverage_data['MediaPayer_BuilderRegionKey'] == source) & 
            (leverage_data['Dest_BuilderRegionKey'] != focus_builder)
        ].copy()
        
        spillover_txt = "None"; score = 0
        if not source_flows.empty and excess > 0:
            # Approximate proj based on rolled up flow? 
            # This is complex with granular data. Simplified approach:
            source_flows['Proj'] = leads_gen * source_flows['Transfer_Rate']
            impacts = []
            seen_ben = set()
            for _, r in source_flows.iterrows():
                ben = r['Dest_BuilderRegionKey']
                if ben in seen_ben: continue
                seen_ben.add(ben)
                
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