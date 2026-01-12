"""
IBN HS Analytics - Main Entry Point
Builder Economics & Referral Network Analysis Platform
"""
import streamlit as st

st.set_page_config(
    page_title="IBN HS Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #F9FAFB;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #E5E7EB;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<p class="main-header">📊 IBN HS Analytics</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Builder Economics & Referral Network Analysis Platform</p>',
        unsafe_allow_html=True
    )
    
    # Data upload section in sidebar
    with st.sidebar:
        st.header("📁 Data Upload")
        st.markdown("Upload your data files to begin analysis.")
        
        events_file = st.file_uploader(
            "Events Master (ref_events_master_*.xlsx)",
            type=["xlsx"],
            key="events_upload"
        )
        
        origin_file = st.file_uploader(
            "Origin Performance (ad_month_origin_perf.xlsx)",
            type=["xlsx"],
            key="origin_upload"
        )
        
        media_file = st.file_uploader(
            "Media Raw (media_raw_base_*.xlsx)",
            type=["xlsx"],
            key="media_upload"
        )
        
        # Store in session state
        if events_file:
            st.session_state['events_file'] = events_file
        if origin_file:
            st.session_state['origin_file'] = origin_file
        if media_file:
            st.session_state['media_file'] = media_file
        
        # Status indicators
        st.divider()
        st.subheader("Data Status")
        
        col1, col2 = st.columns(2)
        with col1:
            if 'events_df' in st.session_state:
                st.success("Events ✓")
            else:
                st.warning("Events ✗")
        with col2:
            if 'origin_df' in st.session_state:
                st.success("Origin ✓")
            else:
                st.warning("Origin ✗")
    
    # Main content
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Builder P&L")
        st.markdown("""
        Analyze builder economics across multiple lenses:
        - **Recipient**: Who receives the leads
        - **Payer**: Who funds the media
        - **Origin**: Original lead source
        
        View by all-time, monthly, or weekly grain.
        """)
        if st.button("Open Builder P&L →", key="btn_pnl"):
            st.switch_page("pages/1_📊_Builder_PnL.py")
    
    with col2:
        st.markdown("### 🎯 Orphan Media")
        st.markdown("""
        Identify wasted ad spend:
        - Zero-lead campaigns (true orphans)
        - Leads with no referrals (zombie funnels)
        - Active vs Paused analysis
        
        Generate kill lists for optimization.
        """)
        if st.button("Open Orphan Media →", key="btn_orphan"):
            st.switch_page("pages/2_🎯_Orphan_Media.py")
    
    with col3:
        st.markdown("### 🔗 Referral Networks")
        st.markdown("""
        Explore referral ecosystems:
        - Community clustering (Louvain)
        - Network visualization
        - Media efficiency pathfinding
        - Downstream cascade analysis
        """)
        if st.button("Open Referral Networks →", key="btn_network"):
            st.switch_page("pages/3_🔗_Referral_Networks.py")
    
    # Quick stats if data is loaded
    if 'events_df' in st.session_state:
        st.markdown("---")
        st.subheader("📈 Quick Overview")
        
        df = st.session_state['events_df']
        
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            n_events = len(df)
            st.metric("Total Events", f"{n_events:,}")
        
        with m2:
            if 'Dest_BuilderRegionKey' in df.columns:
                n_builders = df['Dest_BuilderRegionKey'].nunique()
                st.metric("Unique Builders", f"{n_builders:,}")
        
        with m3:
            if 'MediaCost_referral_event' in df.columns:
                total_media = df['MediaCost_referral_event'].sum()
                st.metric("Total Media Spend", f"${total_media:,.0f}")
        
        with m4:
            if 'ReferralRevenue_event' in df.columns:
                total_rev = df['ReferralRevenue_event'].sum()
                st.metric("Total Revenue", f"${total_rev:,.0f}")


if __name__ == "__main__":
    main()
