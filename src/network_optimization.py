import pandas as pd
import numpy as np

def calculate_shortfalls(
    events_df: pd.DataFrame,
    targets_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Step 1: Calculate Lead Shortfalls and Urgency.
    
    If targets_df is missing (common in demo), we simulate it based on 
    active builders found in the events data.
    """
    # 1. Calculate Actual Referrals Received
    # We look at who received referrals in the events data
    actuals = events_df[events_df['is_referral'] == True].groupby('Dest_BuilderRegionKey').size().reset_index(name='Actual_Referrals')
    
    # 2. Mock Targets if not provided (In production, load from 'LeadTarget_from_job')
    if targets_df is None:
        # Create dummy targets for all builders found in events
        builders = events_df['Dest_BuilderRegionKey'].dropna().unique()
        
        # Create randomized targets and dates for demonstration
        # In production, this would come from the ERP/CRM integration
        np.random.seed(42) # For consistent demo results
        targets_df = pd.DataFrame({
            'BuilderRegionKey': builders,
            'LeadTarget': np.random.randint(5, 50, size=len(builders)), 
            'WIP_JOB_LIVE_END': pd.date_range(start=pd.Timestamp.now(), periods=len(builders), freq='2W') 
        })
    else:
        # Standardize column names if real file provided
        # Mapping common variations to internal standard
        targets_df = targets_df.rename(columns={
            'LeadTarget_from_job': 'LeadTarget', 
            'Builder': 'BuilderRegionKey'
        })

    # 3. Merge Targets with Actuals
    status = targets_df.merge(actuals, left_on='BuilderRegionKey', right_on='Dest_BuilderRegionKey', how='left')
    status['Actual_Referrals'] = status['Actual_Referrals'].fillna(0)
    
    # 4. Calculate Shortfall
    # Shortfall = Target - Actuals. If Actuals > Target, Shortfall is 0 (we don't penalize surplus here, just ignore it)
    status['Shortfall'] = status['LeadTarget'] - status['Actual_Referrals']
    status['Shortfall'] = status['Shortfall'].clip(lower=0) 
    
    # 5. Calculate Urgency Weight based on WIP Date
    now = pd.Timestamp.now()
    status['WIP_JOB_LIVE_END'] = pd.to_datetime(status['WIP_JOB_LIVE_END'], errors='coerce')
    status['Days_Remaining'] = (status['WIP_JOB_LIVE_END'] - now).dt.days
    
    def get_urgency(days):
        if pd.isna(days): return 1.0
        if days < 30: return 2.0  # Critical: Needs leads NOW
        if days < 60: return 1.5  # Urgent
        if days < 90: return 1.2  # Warning
        return 1.0                # Standard pace
        
    status['Urgency_Weight'] = status['Days_Remaining'].apply(get_urgency)
    
    # 6. Weighted Demand = Shortfall * Urgency
    # A shortfall of 10 leads due in 2 weeks is worth more than a shortfall of 10 leads due in 6 months
    status['Weighted_Demand'] = status['Shortfall'] * status['Urgency_Weight']
    
    return status

def compute_effective_network_cpr(
    events_df: pd.DataFrame,
    shortfall_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Step 3: Calculate Effective Utility (Externalities).
    
    Standard CPR = Total Cost / Total Referrals
    Effective CPR = Total Cost / (Referrals that hit a Shortfall)
    
    This penalizes builders who send leads to people who don't need them (wasted surplus),
    and rewards builders who send leads to those in critical need.
    """
    
    # 1. Map the Demand (Shortfall)
    # Create a lookup: Builder -> Weighted Demand
    demand_map = shortfall_df.set_index('BuilderRegionKey')['Weighted_Demand'].to_dict()
    
    # 2. Analyze Sender Flows
    # We need to know: If Payer P sends 100 leads, where do they go?
    
    # Filter to referrals only
    refs = events_df[events_df['is_referral'] == True].copy()
    
    if refs.empty:
        return pd.DataFrame()

    # Group by Payer -> Destination
    # We use MediaPayer as the Source of the investment
    flows = refs.groupby(['MediaPayer_BuilderRegionKey', 'Dest_BuilderRegionKey']).size().reset_index(name='Flow_Count')
    
    # Calculate total referrals generated per payer
    payer_totals = flows.groupby('MediaPayer_BuilderRegionKey')['Flow_Count'].sum().reset_index(name='Total_Referrals_Out')
    
    # Merge totals back to calculate distribution % (Propensity to send to specific targets)
    flows = flows.merge(payer_totals, on='MediaPayer_BuilderRegionKey')
    flows['Flow_Pct'] = flows['Flow_Count'] / flows['Total_Referrals_Out']
    
    # 3. Calculate "Effective Yield" per Payer
    results = []
    
    # Get Media Cost per Payer
    media_costs = events_df.groupby('MediaPayer_BuilderRegionKey')['MediaCost_referral_event'].sum()
    
    for payer in flows['MediaPayer_BuilderRegionKey'].unique():
        payer_flows = flows[flows['MediaPayer_BuilderRegionKey'] == payer]
        
        # Base Stats
        total_refs = payer_flows['Total_Referrals_Out'].iloc[0]
        total_cost = media_costs.get(payer, 0)
        
        if total_refs == 0 or total_cost == 0:
            continue
            
        raw_cpr = total_cost / total_refs
        
        # Calculate Effectiveness / Utility
        # Sum of (Flow Count * Destination Demand)
        # We define "Useful Referrals" as referrals that landed at a builder with >0 Demand
        
        useful_referrals_count = 0
        utility_score = 0
        beneficiaries = []
        
        for _, row in payer_flows.iterrows():
            dest = row['Dest_BuilderRegionKey']
            count = row['Flow_Count']
            
            # Check if destination has demand
            demand = demand_map.get(dest, 0)
            
            if demand > 0:
                useful_referrals_count += count
                # Utility score is weighted by urgency (not just count)
                utility_score += (count * demand) 
                beneficiaries.append(dest)
        
        # Metrics
        # "Useful Share": % of referrals that went to builders with active shortfalls
        useful_share = useful_referrals_count / total_refs
        
        # "Effective CPR": The cost to get one *useful* referral
        # If Useful Share is 50%, Effective CPR is double the Raw CPR because half the money is "wasted" on surplus
        effective_cpr = raw_cpr / useful_share if useful_share > 0 else np.inf
        
        results.append({
            'BuilderRegionKey': payer,
            'Total_Referrals_Sent': total_refs,
            'Useful_Referrals': useful_referrals_count,
            'Useful_Share': useful_share,
            'Beneficiaries_Count': len(set(beneficiaries)),
            'Top_Beneficiary': beneficiaries[0] if beneficiaries else None,
            'Media_Cost': total_cost,
            'Raw_CPR': raw_cpr,
            'Effective_CPR': effective_cpr,
            'Urgency_Score': utility_score
        })
        
    results_df = pd.DataFrame(results)
    
    # Rank by Effective CPR (Lower is better)
    if not results_df.empty:
        results_df = results_df.sort_values('Effective_CPR', ascending=True)
        
    return results_df