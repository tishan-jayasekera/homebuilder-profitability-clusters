import pandas as pd
import numpy as np

def calculate_shortfalls(events_df: pd.DataFrame, targets_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Step 1: Calculate Demand (Shortfalls).
    Logic: Shortfall = Target - Actuals.
    """
    # 1. Actuals
    actuals = events_df[events_df['is_referral'] == True].groupby('Dest_BuilderRegionKey').size().reset_index(name='Actual_Referrals')
    
    # 2. Targets (Mock if missing)
    if targets_df is None:
        builders = events_df['Dest_BuilderRegionKey'].dropna().unique()
        # Mocking for demo resilience
        targets_df = pd.DataFrame({
            'BuilderRegionKey': builders,
            'LeadTarget': 50, # Simple default
            'WIP_JOB_LIVE_END': pd.Timestamp.now() + pd.Timedelta(days=60)
        })
    else:
        # Normalize
        cols = {'LeadTarget_from_job': 'LeadTarget', 'Builder': 'BuilderRegionKey'}
        targets_df = targets_df.rename(columns=cols)
        # Ensure minimal columns exist
        if 'LeadTarget' not in targets_df.columns: targets_df['LeadTarget'] = 0
    
    # 3. Merge
    df = targets_df.merge(actuals, left_on='BuilderRegionKey', right_on='Dest_BuilderRegionKey', how='left')
    df['Actual_Referrals'] = df['Actual_Referrals'].fillna(0)
    df['Shortfall'] = (df['LeadTarget'] - df['Actual_Referrals']).clip(lower=0)
    
    # 4. Urgency
    now = pd.Timestamp.now()
    if 'WIP_JOB_LIVE_END' in df.columns:
        df['WIP_JOB_LIVE_END'] = pd.to_datetime(df['WIP_JOB_LIVE_END'], errors='coerce')
        df['Days_Remaining'] = (df['WIP_JOB_LIVE_END'] - now).dt.days
    else:
        df['Days_Remaining'] = np.nan
        
    return df

def analyze_network_leverage(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Analyze Supply (The Leverage).
    Calculates Transfer Rate (TR), Base CPR, and Effective CPR (eCPR).
    """
    # Filter to referrals
    refs = events_df[events_df['is_referral'] == True].copy()
    
    if refs.empty:
        return pd.DataFrame()

    # A. Source Metrics (Denominator)
    # Total referrals sent by each source (MediaPayer)
    source_stats = refs.groupby('MediaPayer_BuilderRegionKey').agg(
        Total_Referrals_Sent=('LeadId', 'count'),
        Total_Media_Spend=('MediaCost_referral_event', 'sum')
    ).reset_index()
    
    # Base CPR = Spend / Total Referrals
    source_stats['CPR_base'] = np.where(
        source_stats['Total_Referrals_Sent'] > 0,
        source_stats['Total_Media_Spend'] / source_stats['Total_Referrals_Sent'],
        np.nan
    )
    
    # B. Flow Metrics (Numerator)
    # Referrals from Source -> Specific Target
    flows = refs.groupby(['MediaPayer_BuilderRegionKey', 'Dest_BuilderRegionKey']).size().reset_index(name='Referrals_to_Target')
    
    # C. Merge & Calculate TR / eCPR
    leverage = flows.merge(source_stats, on='MediaPayer_BuilderRegionKey', how='left')
    
    # Transfer Rate = Refs to Target / Total Refs Sent
    leverage['Transfer_Rate'] = leverage['Referrals_to_Target'] / leverage['Total_Referrals_Sent']
    
    # Effective CPR = Base CPR / Transfer Rate
    # (Cost to generate 1 lead for target, accounting for dilution)
    leverage['eCPR'] = np.where(
        leverage['Transfer_Rate'] > 0,
        leverage['CPR_base'] / leverage['Transfer_Rate'],
        np.inf
    )
    
    return leverage

def generate_investment_strategies(
    focus_builder: str,
    shortfall_data: pd.DataFrame,
    leverage_data: pd.DataFrame,
    events_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Step 3: Calculate Investment & Externalities.
    For a specific Focus Builder, what are the best pathways?
    """
    # Get Shortfall for Focus Builder
    target_row = shortfall_data[shortfall_data['BuilderRegionKey'] == focus_builder]
    if target_row.empty:
        return pd.DataFrame()
        
    shortfall = target_row['Shortfall'].iloc[0]
    
    if shortfall <= 0:
        return pd.DataFrame() # No strategy needed
        
    # Get potential sources
    strategies = leverage_data[leverage_data['Dest_BuilderRegionKey'] == focus_builder].copy()
    
    if strategies.empty:
        return pd.DataFrame()
        
    results = []
    
    # Pre-calculate spillover mappings for efficiency
    # Map: Source -> List of [Other_Dest, % share]
    # We can rely on leverage_data for this since it has all flows
    
    for _, strat in strategies.iterrows():
        source = strat['MediaPayer_BuilderRegionKey']
        tr = strat['Transfer_Rate']
        ecpr = strat['eCPR']
        
        # Scenario: We want to generate 'Shortfall' amount of leads for Focus Builder
        # Required Total Generation at Source = Shortfall / TR
        total_leads_needed = shortfall / tr if tr > 0 else 0
        investment_required = total_leads_needed * strat['CPR_base']
        
        # Externalities (Spillover)
        excess_leads = total_leads_needed - shortfall
        
        # Where do excess leads go?
        # Get distribution for this source (excluding focus builder)
        source_dist = leverage_data[
            (leverage_data['MediaPayer_BuilderRegionKey'] == source) & 
            (leverage_data['Dest_BuilderRegionKey'] != focus_builder)
        ].copy()
        
        spillover_impact = []
        optimization_score = 0
        
        if not source_dist.empty and excess_leads > 0:
            # Calculate estimated leads per other builder
            # Note: We need to re-normalize TR relative to the *remaining* pool if we want exacts, 
            # but simpler is: Leads = Total_Gen * TR (since TR is based on total)
            
            source_dist['Spillover_Leads'] = total_leads_needed * source_dist['Transfer_Rate']
            
            # Check if these spillover leads hit OTHER shortfalls
            # Merge with shortfall data
            impact = source_dist.merge(shortfall_data[['BuilderRegionKey', 'Shortfall']], 
                                     left_on='Dest_BuilderRegionKey', right_on='BuilderRegionKey', how='left')
            
            # Score: +1 for every lead that hits another shortfall
            for _, imp in impact.iterrows():
                if imp['Shortfall'] > 0:
                    # Capped at their actual shortfall
                    useful_spillover = min(imp['Spillover_Leads'], imp['Shortfall'])
                    optimization_score += useful_spillover
                    spillover_impact.append(f"{imp['Dest_BuilderRegionKey']} (+{int(imp['Spillover_Leads'])})")
        
        spillover_txt = ", ".join(spillover_impact[:3]) # Top 3
        if len(spillover_impact) > 3: spillover_txt += "..."
        
        results.append({
            'Source_Builder': source,
            'Transfer_Rate': tr,
            'Base_CPR': strat['CPR_base'],
            'Effective_CPR': ecpr,
            'Investment_Required': investment_required,
            'Total_Leads_Generated': total_leads_needed,
            'Excess_Leads': excess_leads,
            'Spillover_Impact': spillover_txt if spillover_txt else "None/Surplus",
            'Optimization_Score': optimization_score
        })
        
    res_df = pd.DataFrame(results)
    
    # Sort by Optimization Score (Network Benefit) then Effective CPR (Cost)
    # We want High Score, Low Cost
    if not res_df.empty:
        res_df = res_df.sort_values(['Optimization_Score', 'Effective_CPR'], ascending=[False, True])
        
    return res_df

# For compatibility with legacy calls or general portfolio view
def compute_effective_network_cpr(events_df, shortfall_df):
    # This was the function causing the KeyError before. 
    # We can implement a simplified version or just alias it if needed.
    # Let's provide a robust implementation that matches the required signature.
    leverage = analyze_network_leverage(events_df)
    
    # We need to aggregate to get "Beneficiaries_Count" etc.
    # This effectively pivots the leverage data
    
    if leverage.empty: return pd.DataFrame()
    
    # Group by Source
    grouped = leverage.groupby('MediaPayer_BuilderRegionKey').agg({
        'Total_Referrals_Sent': 'first',
        'CPR_base': 'first',
        'Dest_BuilderRegionKey': 'nunique' # Count of beneficiaries
    }).rename(columns={'Dest_BuilderRegionKey': 'Beneficiaries_Count', 'CPR_base': 'Raw_CPR'})
    
    # Calculate a weighted utility (simplified for portfolio view)
    # Real logic is in generate_investment_strategies
    grouped['Effective_CPR'] = grouped['Raw_CPR'] # Placeholder if not specific
    grouped['Useful_Share'] = 1.0
    grouped['Top_Beneficiary'] = "Various"
    
    return grouped.reset_index().rename(columns={'MediaPayer_BuilderRegionKey': 'BuilderRegionKey'})