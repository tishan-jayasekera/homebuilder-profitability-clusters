import pandas as pd
import numpy as np

def calculate_shortfalls(events_df: pd.DataFrame, targets_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Step 1: Calculate Demand (Shortfalls).
    Logic: Shortfall = Target - Actuals.
    """
    # 1. Actuals: Current_Leads (sum of inbound referrals to date)
    # We use events where is_referral is True as 'inbound referrals'
    actuals = events_df[events_df['is_referral'] == True].groupby('Dest_BuilderRegionKey').size().reset_index(name='Actual_Referrals')
    
    # 2. Targets: Retrieve LeadTarget_from_job and WIP_JOB_LIVE_END
    if targets_df is None:
        # Fallback: Extract from events_df if available, else mock
        # Assuming targets might be repeated on rows for each builder in a real dataset
        # For now, we'll try to aggregate if columns exist, otherwise mock
        builders = events_df['Dest_BuilderRegionKey'].dropna().unique()
        
        cols_to_check = ['LeadTarget_from_job', 'WIP_JOB_LIVE_END']
        if all(c in events_df.columns for c in cols_to_check):
             # Extract unique targets per builder
             targets_df = events_df[['Dest_BuilderRegionKey', 'LeadTarget_from_job', 'WIP_JOB_LIVE_END']].drop_duplicates('Dest_BuilderRegionKey').copy()
             targets_df = targets_df.rename(columns={'Dest_BuilderRegionKey': 'BuilderRegionKey', 'LeadTarget_from_job': 'LeadTarget'})
        else:
            # Mocking for resilience if columns strictly don't exist
            targets_df = pd.DataFrame({
                'BuilderRegionKey': builders,
                'LeadTarget': 50, # Default target
                'WIP_JOB_LIVE_END': pd.Timestamp.now() + pd.Timedelta(days=60)
            })
    else:
        # Normalize external targets_df
        # Mapping variations to internal standard
        rename_map = {
            'LeadTarget_from_job': 'LeadTarget', 
            'Builder': 'BuilderRegionKey',
            'WIP_JOB_LIVE_END': 'WIP_JOB_LIVE_END' # explicit keep
        }
        targets_df = targets_df.rename(columns=rename_map)
        
        # Ensure minimal columns exist
        if 'LeadTarget' not in targets_df.columns: targets_df['LeadTarget'] = 0
    
    # 3. Merge Targets with Actuals
    df = targets_df.merge(actuals, left_on='BuilderRegionKey', right_on='Dest_BuilderRegionKey', how='left')
    df['Actual_Referrals'] = df['Actual_Referrals'].fillna(0)
    
    # 4. Calculate Shortfall
    # Shortfall = LeadTarget_from_job - Current_Leads
    df['Shortfall'] = (df['LeadTarget'] - df['Actual_Referrals']).clip(lower=0)
    
    # 5. Urgency: Days Remaining
    now = pd.Timestamp.now()
    if 'WIP_JOB_LIVE_END' in df.columns:
        df['WIP_JOB_LIVE_END'] = pd.to_datetime(df['WIP_JOB_LIVE_END'], errors='coerce')
        df['Days_Remaining'] = (df['WIP_JOB_LIVE_END'] - now).dt.days
    else:
        df['Days_Remaining'] = np.nan
        
    return df

def analyze_network_leverage(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Analyze Supply Sources (The Leverage).
    Calculates Transfer Rate (TR), Base CPR (CPR_base), and Effective CPR (eCPR).
    """
    # Filter to referrals only (outbound flow)
    refs = events_df[events_df['is_referral'] == True].copy()
    
    if refs.empty:
        return pd.DataFrame()

    # A. Source Metrics (Denominator for TR & CPR_base)
    # Total Referrals sent by S
    # Media Spend of S (using MediaCost_referral_event as proxy for allocation)
    source_stats = refs.groupby('MediaPayer_BuilderRegionKey').agg(
        Total_Referrals_Sent=('LeadId', 'count'),
        Total_Media_Spend=('MediaCost_referral_event', 'sum')
    ).reset_index()
    
    # Base CPR = Media Spend of S / Total Referrals sent by S
    source_stats['CPR_base'] = np.where(
        source_stats['Total_Referrals_Sent'] > 0,
        source_stats['Total_Media_Spend'] / source_stats['Total_Referrals_Sent'],
        np.nan
    )
    
    # B. Flow Metrics (Numerator for TR)
    # Referrals from S to B_target (specific flows)
    flows = refs.groupby(['MediaPayer_BuilderRegionKey', 'Dest_BuilderRegionKey']).size().reset_index(name='Referrals_to_Target')
    
    # C. Merge & Calculate TR / eCPR
    leverage = flows.merge(source_stats, on='MediaPayer_BuilderRegionKey', how='left')
    
    # Transfer Rate (TR) = Referrals from S to B_target / Total Referrals sent by S
    leverage['Transfer_Rate'] = leverage['Referrals_to_Target'] / leverage['Total_Referrals_Sent']
    
    # Effective CPR (eCPR) = CPR_base / TR
    # Logic: Cost to generate 1 lead specifically for B_target
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
    Step 3: Calculate Investment & Externalities (The "Spillover").
    For a specific Focus Builder, calculates investment scenarios for each Source.
    """
    # Get Shortfall for Focus Builder
    target_row = shortfall_data[shortfall_data['BuilderRegionKey'] == focus_builder]
    if target_row.empty:
        return pd.DataFrame()
        
    shortfall = target_row['Shortfall'].iloc[0]
    
    # Identify all builders (S) who have historically sent referrals to focus_builder
    strategies = leverage_data[leverage_data['Dest_BuilderRegionKey'] == focus_builder].copy()
    
    if strategies.empty:
        return pd.DataFrame()
        
    results = []
    
    # Pre-calculate shortfall map for spillover checking
    shortfall_map = shortfall_data.set_index('BuilderRegionKey')['Shortfall'].to_dict()
    
    for _, strat in strategies.iterrows():
        source = strat['MediaPayer_BuilderRegionKey']
        tr = strat['Transfer_Rate']
        ecpr = strat['eCPR']
        cpr_base = strat['CPR_base']
        
        # Scenario: Plug the Shortfall using Source S
        if shortfall > 0 and tr > 0:
            # Investment Required = Shortfall * eCPR
            investment_required = shortfall * ecpr
            
            # Total Leads Generated = Shortfall / TR
            total_leads_generated = shortfall / tr
            
            # Excess Leads = Total Leads Generated - Shortfall
            excess_leads = total_leads_generated - shortfall
        else:
            investment_required = 0
            total_leads_generated = 0
            excess_leads = 0
            
        # Externalities: Where do these Excess Leads go?
        # Use S's historical distribution to calculate which other builders get leads.
        
        # Get distribution for this source (excluding focus builder)
        source_flows = leverage_data[
            (leverage_data['MediaPayer_BuilderRegionKey'] == source) & 
            (leverage_data['Dest_BuilderRegionKey'] != focus_builder)
        ].copy()
        
        spillover_impact = []
        optimization_score = 0
        
        if not source_flows.empty and excess_leads > 0:
            # How many leads go to other builders?
            # We assume the distribution of excess leads follows the Source's general Transfer Rates
            # Note: TR is % of Total. Total_Leads_Generated is the new total.
            # So Leads to Other B = Total_Leads_Generated * TR_other
            
            source_flows['Projected_Leads'] = total_leads_generated * source_flows['Transfer_Rate']
            
            for _, row_flow in source_flows.iterrows():
                other_builder = row_flow['Dest_BuilderRegionKey']
                proj_leads = row_flow['Projected_Leads']
                
                # Check if this helps another builder's shortfall
                other_shortfall = shortfall_map.get(other_builder, 0)
                
                if other_shortfall > 0:
                    # It's a "useful" spillover (Optimization Score)
                    # Capped at their actual need for scoring purposes
                    useful_amount = min(proj_leads, other_shortfall)
                    optimization_score += useful_amount
                    spillover_impact.append(f"{other_builder} (+{int(proj_leads)})")
        
        spillover_txt = ", ".join(spillover_impact[:3]) # Show top 3
        if len(spillover_impact) > 3: spillover_txt += f", +{len(spillover_impact)-3} others"
        if not spillover_txt: spillover_txt = "Non-critical Surplus"
        
        results.append({
            'Source_Builder': source,
            'Transfer_Rate': tr,
            'Base_CPR': cpr_base,
            'Effective_CPR': ecpr,
            'Investment_Required': investment_required,
            'Total_Leads_Generated': total_leads_generated,
            'Excess_Leads': excess_leads,
            'Spillover_Impact': spillover_txt,
            'Optimization_Score': optimization_score
        })
        
    res_df = pd.DataFrame(results)
    
    # Sort by Optimization Score (High spillover benefit) then Effective CPR (Low cost)
    if not res_df.empty:
        res_df = res_df.sort_values(['Optimization_Score', 'Effective_CPR'], ascending=[False, True])
        
    return res_df

# Helper for compatibility
def compute_effective_network_cpr(events_df, shortfall_df):
    """
    Computes a network-wide effective CPR metric for ranking.
    Used for the general 'Investment Recommendations' table.
    """
    leverage = analyze_network_leverage(events_df)
    if leverage.empty: return pd.DataFrame()
    
    # Aggregate to Source level for ranking
    # We want to know: "If I invest in S, how good is it generally?"
    grouped = leverage.groupby('MediaPayer_BuilderRegionKey').agg({
        'Total_Referrals_Sent': 'first',
        'CPR_base': 'first',
        'Dest_BuilderRegionKey': 'nunique'
    }).rename(columns={'Dest_BuilderRegionKey': 'Beneficiaries_Count', 'CPR_base': 'Raw_CPR'})
    
    # Simplified Effective CPR for the network view:
    # Just pass through Base CPR as Raw CPR. Real eCPR is target-specific.
    grouped['CPR'] = grouped['Raw_CPR']
    
    return grouped.reset_index().rename(columns={'MediaPayer_BuilderRegionKey': 'BuilderRegionKey'})