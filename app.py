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
            if 'events_file' in st.session_state:
                st.success("Events ✓")
            else:
                st.warning("Events ✗")
        with col2:
            if 'origin_file' in st.session_state:
                st.success("Origin ✓")
            else:
                st.warning("Origin ✗")
    
    # Main content
    st.markdown("---")
    
    st.info("👈 **Upload your data files in the sidebar**, then use the navigation menu to access the dashboards.")
    
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
        st.markdown("➡️ Go to **Builder_PnL** in sidebar")
    
    with col2:
        st.markdown("### 🎯 Orphan Media")
        st.markdown("""
        Identify wasted ad spend:
        - Zero-lead campaigns (true orphans)
        - Leads with no referrals (zombie funnels)
        - Active vs Paused analysis
        
        Generate kill lists for optimization.
        """)
        st.markdown("➡️ Go to **Orphan_Media** in sidebar")
    
    with col3:
        st.markdown("### 🔗 Referral Networks")
        st.markdown("""
        Explore referral ecosystems:
        - Community clustering (Louvain)
        - Network visualization
        - Media efficiency pathfinding
        - Downstream cascade analysis
        """)
        st.markdown("➡️ Go to **Referral_Networks** in sidebar")
    
    # Quick stats if data is loaded
    if 'events_file' in st.session_state:
        st.markdown("---")
        st.subheader("📈 Data Preview")
        
        try:
            import pandas as pd
            df = pd.read_excel(st.session_state['events_file'])
            st.session_state['events_df'] = df
            
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
                elif 'MediaCost_builder_touch' in df.columns:
                    total_media = df['MediaCost_builder_touch'].sum()
                    st.metric("Total Media Spend", f"${total_media:,.0f}")
            
            with m4:
                if 'RPL_from_job' in df.columns:
                    total_rev = df['RPL_from_job'].sum()
                    st.metric("Total Revenue", f"${total_rev:,.0f}")
                elif 'ReferralRevenue_event' in df.columns:
                    total_rev = df['ReferralRevenue_event'].sum()
                    st.metric("Total Revenue", f"${total_rev:,.0f}")
            
            with st.expander("Preview Data Columns"):
                st.write(f"**Columns ({len(df.columns)}):** {', '.join(df.columns[:20].tolist())}{'...' if len(df.columns) > 20 else ''}")
                
        except Exception as e:
            st.error(f"Error reading file: {e}")


if __name__ == "__main__":
    main()