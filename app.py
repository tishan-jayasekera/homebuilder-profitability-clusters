"""
IBN HS Analytics - Main Entry Point
Builder Economics & Referral Network Analysis Platform
"""
import streamlit as st
import os
from pathlib import Path

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
    .nav-card {
        background: #F9FAFB;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }
    div[data-testid="stButton"] > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def find_page(keyword):
    """Find a page file containing the keyword."""
    pages_dir = Path(__file__).parent / "pages"
    if pages_dir.exists():
        for f in pages_dir.iterdir():
            if f.suffix == '.py' and keyword.lower() in f.name.lower():
                return f"pages/{f.name}"
    return None


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
        
        with st.columns(2)[0]:
            if 'media_file' in st.session_state:
                st.success("Media ✓")
            else:
                st.warning("Media ✗")
    
    # Main content
    st.markdown("---")
    
    # Check if data is loaded
    if 'events_file' not in st.session_state:
        st.warning("👈 **Please upload your Events data file in the sidebar to get started.**")
    else:
        st.success("✅ **Data loaded!** Choose a dashboard below to begin analysis.")
    
    st.markdown("### 🚀 Choose a Dashboard")
    
    # Find page paths
    pnl_page = find_page("builder") or find_page("pnl")
    orphan_page = find_page("orphan")
    network_page = find_page("referral") or find_page("network")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📊 Builder P&L")
        st.markdown("""
        Analyze builder economics across multiple lenses:
        - **Recipient**: Who receives the leads
        - **Payer**: Who funds the media
        - **Origin**: Original lead source
        
        View by all-time, monthly, or weekly grain.
        """)
        if pnl_page:
            if st.button("📊 Open Builder P&L", key="btn_pnl", use_container_width=True):
                st.switch_page(pnl_page)
        else:
            st.info("Page: 1_Builder_PnL")
    
    with col2:
        st.markdown("#### 🎯 Orphan Media")
        st.markdown("""
        Identify wasted ad spend:
        - Zero-lead campaigns (true orphans)
        - Leads with no referrals (zombie funnels)
        - Active vs Paused analysis
        
        Generate kill lists for optimization.
        """)
        if orphan_page:
            if st.button("🎯 Open Orphan Media", key="btn_orphan", use_container_width=True):
                st.switch_page(orphan_page)
        else:
            st.info("Page: 2_Orphan_Media")
    
    with col3:
        st.markdown("#### 🔗 Referral Networks")
        st.markdown("""
        Explore referral ecosystems:
        - Community clustering (Louvain)
        - Network visualization
        - Media efficiency pathfinding
        - Downstream cascade analysis
        """)
        if network_page:
            if st.button("🔗 Open Referral Networks", key="btn_network", use_container_width=True):
                st.switch_page(network_page)
        else:
            st.info("Page: 3_Referral_Networks")
    
    # Debug: show available pages
    pages_dir = Path(__file__).parent / "pages"
    if pages_dir.exists():
        pages_found = [f.name for f in pages_dir.iterdir() if f.suffix == '.py']
        if pages_found:
            with st.expander("🔧 Debug: Available Pages"):
                st.write("Pages found in /pages directory:")
                for p in sorted(pages_found):
                    st.code(p)
    
    # Quick stats if data is loaded
    if 'events_file' in st.session_state:
        st.markdown("---")
        st.subheader("📈 Data Preview")
        
        try:
            import pandas as pd
            
            # Only read if not already cached
            if 'events_df' not in st.session_state:
                df = pd.read_excel(st.session_state['events_file'])
                st.session_state['events_df'] = df
            else:
                df = st.session_state['events_df']
            
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                n_events = len(df)
                st.metric("Total Events", f"{n_events:,}")
            
            with m2:
                if 'Dest_BuilderRegionKey' in df.columns:
                    n_builders = df['Dest_BuilderRegionKey'].nunique()
                    st.metric("Unique Builders", f"{n_builders:,}")
                else:
                    st.metric("Unique Builders", "N/A")
            
            with m3:
                media_col = None
                for col in ['MediaCost_referral_event', 'MediaCost_builder_touch', 'MediaCost']:
                    if col in df.columns:
                        media_col = col
                        break
                if media_col:
                    total_media = df[media_col].sum()
                    st.metric("Total Media Spend", f"${total_media:,.0f}")
                else:
                    st.metric("Total Media Spend", "N/A")
            
            with m4:
                rev_col = None
                for col in ['RPL_from_job', 'ReferralRevenue_event', 'Revenue']:
                    if col in df.columns:
                        rev_col = col
                        break
                if rev_col:
                    total_rev = df[rev_col].sum()
                    st.metric("Total Revenue", f"${total_rev:,.0f}")
                else:
                    st.metric("Total Revenue", "N/A")
            
            with st.expander("📋 View Data Columns"):
                st.write(f"**Total columns:** {len(df.columns)}")
                st.write(df.columns.tolist())
                
        except Exception as e:
            st.error(f"Error reading file: {e}")


if __name__ == "__main__":
    main()