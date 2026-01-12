import pandas as pd
import numpy as np

def calculate_shortfalls(
    events_df: pd.DataFrame,
    targets_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Step 1: Calculate Lead Shortfalls and Urgency.
    """
    # 1. Calculate Actual Referrals Received
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
        targets_df = targets_df.rename(columns={
            'LeadTarget_from_job': 'LeadTarget', 
            'Builder': 'BuilderRegionKey'
        })

    # 3. Merge Targets with Actuals
    status = targets_df.merge(actuals, left_on='BuilderRegionKey', right_on='Dest_BuilderRegionKey', how='left')
    status['Actual_Referrals'] = status['Actual_Referrals'].fillna(0)
    
    # 4. Calculate Shortfall
    status['Shortfall'] = status['LeadTarget'] - status['Actual_Referrals']
    status['Shortfall'] = status['Shortfall'].clip(lower=0) 
    
    # 5. Calculate Urgency Weight based on WIP Date
    now = pd.Timestamp.now()
    status['WIP_JOB_LIVE_END'] = pd.to_datetime(status['WIP_JOB_LIVE_END'], errors='coerce')
    status['Days_Remaining'] = (status['WIP_JOB_LIVE_END'] - now).dt.days
    
    def get_urgency(days):
        if pd.isna(days): return 1.0
        if days < 30: return 2.0  # Critical
        if days < 60: return 1.5  # Urgent
        if days < 90: return 1.2  # Warning
        return 1.0                # Standard pace
        
    status['Urgency_Weight'] = status['Days_Remaining'].apply(get_urgency)
    status['Weighted_Demand'] = status['Shortfall'] * status['Urgency_Weight']
    
    return status

def get_targeted_fulfillment_strategies(
    events_df: pd.DataFrame,
    shortfall_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Step 2: Develop Media Strategy to Fulfill Specific Shortfalls.
    """
    
    # 1. Calculate Payer Performance (Cost per generic lead)
    payer_stats = events_df.groupby('MediaPayer_BuilderRegionKey').agg(
        Total_Spend=('MediaCost_referral_event', 'sum'),
        Total_Leads_Generated=('LeadId', 'count')
    ).reset_index()
    
    # Filter out zero spend/leads
    payer_stats = payer_stats[payer_stats['Total_Leads_Generated'] > 0]
    payer_stats['Payer_Raw_CPL'] = payer_stats['Total_Spend'] / payer_stats['Total_Leads_Generated']
    
    # 2. Calculate Flow Rates (Payer -> Recipient)
    flows_df = events_df.copy()
    flows_df['Dest_Builder'] = np.where(
        flows_df['is_referral'] == True, 
        flows_df['Dest_BuilderRegionKey'], 
        flows_df['MediaPayer_BuilderRegionKey'] 
    )
    
    flows = flows_df.groupby(['MediaPayer_BuilderRegionKey', 'Dest_Builder']).size().reset_index(name='Flow_Count')
    
    # Calculate totals to get %
    payer_totals = flows.groupby('MediaPayer_BuilderRegionKey')['Flow_Count'].sum().reset_index(name='Total_Flow')
    flows = flows.merge(payer_totals, on='MediaPayer_BuilderRegionKey')
    flows['Flow_Rate'] = flows['Flow_Count'] / flows['Total_Flow']
    
    # 3. Merge Cost Data into Flows
    strategies = flows.merge(payer_stats[['MediaPayer_BuilderRegionKey', 'Payer_Raw_CPL']], on='MediaPayer_BuilderRegionKey')
    
    # 4. Calculate "Effective Cost to Serve Target"
    strategies['Cost_Per_Target_Lead'] = strategies['Payer_Raw_CPL'] / strategies['Flow_Rate']
    strategies['Cost_Per_Target_Lead'] = strategies['Cost_Per_Target_Lead'].replace([np.inf, -np.inf], 0)
    
    # 5. Connect to Shortfalls
    active_shortfalls = shortfall_df[shortfall_df['Shortfall'] > 0][['BuilderRegionKey', 'Shortfall', 'Urgency_Weight', 'Weighted_Demand']]
    
    opportunities = strategies.merge(
        active_shortfalls, 
        left_on='Dest_Builder', 
        right_on='BuilderRegionKey', 
        how='inner'
    )
    
    # 6. Sorting
    opportunities = opportunities.sort_values(
        ['Weighted_Demand', 'Cost_Per_Target_Lead'], 
        ascending=[False, True]
    )
    
    # Select useful columns
    cols = [
        'Dest_Builder', 'Shortfall', 'Urgency_Weight', 
        'MediaPayer_BuilderRegionKey', 'Payer_Raw_CPL', 
        'Flow_Rate', 'Cost_Per_Target_Lead'
    ]
    
    return opportunities[cols]

def generate_network_fulfillment_plan(
    shortfall_df: pd.DataFrame, 
    strategies_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Step 3: Reconciliation (The Master Plan).
    Matches every Shortfall to its Best Available Strategy.
    """
    plan_rows = []
    
    # 1. Identify Demand
    demand_nodes = shortfall_df[shortfall_df['Shortfall'] > 0].copy()
    
    if demand_nodes.empty:
        return pd.DataFrame()
        
    for _, node in demand_nodes.iterrows():
        target = node['BuilderRegionKey']
        needed = node['Shortfall']
        urgency = node['Urgency_Weight']
        deadline = node.get('WIP_JOB_LIVE_END', pd.NaT)
        
        # 2. Find Best Strategy
        # Filter strategies serving this target
        options = strategies_df[strategies_df['Dest_Builder'] == target].copy()
        
        if options.empty:
            # Under-serviced
            plan_rows.append({
                'Priority': '🔴 Critical Gap',
                'Target_Builder': target,
                'Shortfall': needed,
                'Urgency': urgency,
                'WIP_Deadline': deadline,
                'Status': 'Under-Serviced',
                'Best_Payer': 'None',
                'Recommended_Action': 'Direct Media / New Partner',
                'Est_Budget_Required': 0, 
                'Strategy_Detail': 'No inbound flow detected'
            })
        else:
            # Pick best option (Lowest Cost Per Target Lead)
            best = options.sort_values('Cost_Per_Target_Lead').iloc[0]
            
            est_cost = needed * best['Cost_Per_Target_Lead']
            
            plan_rows.append({
                'Priority': '🟢 Actionable',
                'Target_Builder': target,
                'Shortfall': needed,
                'Urgency': urgency,
                'WIP_Deadline': deadline,
                'Status': 'Fulfillable',
                'Best_Payer': best['MediaPayer_BuilderRegionKey'],
                'Recommended_Action': f"Invest in {best['MediaPayer_BuilderRegionKey']}",
                'Est_Budget_Required': est_cost,
                'Strategy_Detail': f"Effective CPL: ${best['Cost_Per_Target_Lead']:,.0f}"
            })
            
    plan_df = pd.DataFrame(plan_rows)
    if not plan_df.empty:
        return plan_df.sort_values(['Urgency', 'Shortfall'], ascending=[False, False])
    return pd.DataFrame()

def compute_effective_network_cpr(
    events_df: pd.DataFrame,
    shortfall_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Step 4: Portfolio View.
    Calculates 'Effective CPR' for Payers.
    """
    demand_map = shortfall_df.set_index('BuilderRegionKey')['Weighted_Demand'].to_dict()
    
    refs = events_df[events_df['is_referral'] == True].copy()
    if refs.empty: return pd.DataFrame()

    flows = refs.groupby(['MediaPayer_BuilderRegionKey', 'Dest_BuilderRegionKey']).size().reset_index(name='Flow_Count')
    payer_totals = flows.groupby('MediaPayer_BuilderRegionKey')['Flow_Count'].sum().reset_index(name='Total_Referrals_Out')
    flows = flows.merge(payer_totals, on='MediaPayer_BuilderRegionKey')
    
    media_costs = events_df.groupby('MediaPayer_BuilderRegionKey')['MediaCost_referral_event'].sum()
    
    results = []
    for payer in flows['MediaPayer_BuilderRegionKey'].unique():
        payer_flows = flows[flows['MediaPayer_BuilderRegionKey'] == payer]
        total_refs = payer_flows['Total_Referrals_Out'].iloc[0]
        total_cost = media_costs.get(payer, 0)
        
        if total_refs == 0 or total_cost == 0: continue
            
        raw_cpr = total_cost / total_refs
        useful_referrals_count = 0
        utility_score = 0
        
        for _, row in payer_flows.iterrows():
            dest = row['Dest_BuilderRegionKey']
            count = row['Flow_Count']
            demand = demand_map.get(dest, 0)
            if demand > 0:
                useful_referrals_count += count
                utility_score += (count * demand) 
        
        useful_share = useful_referrals_count / total_refs
        effective_cpr = raw_cpr / useful_share if useful_share > 0 else np.inf
        
        results.append({
            'BuilderRegionKey': payer,
            'Total_Referrals_Sent': total_refs,
            'Useful_Share': useful_share,
            'Media_Cost': total_cost,
            'Raw_CPR': raw_cpr,
            'Effective_CPR': effective_cpr
        })
        
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('Effective_CPR', ascending=True)
        
    return results_df