"""
Settings and Configuration Page
"""

import streamlit as st
import yaml
from pathlib import Path

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

st.title("⚙️ Settings & Configuration")
st.markdown("Configure arbitrage detection parameters and transaction costs")
st.markdown("---")

# Transaction Costs
st.subheader("Transaction Costs")

col1, col2, col3 = st.columns(3)

with col1:
    equity_cost = st.number_input(
        "Equity Trading (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.01,
        format="%.3f",
        help="Transaction cost for equity trades"
    )

with col2:
    options_cost = st.number_input(
        "Options Trading (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.01,
        format="%.3f",
        help="Transaction cost for option trades"
    )

with col3:
    futures_cost = st.number_input(
        "Futures Trading (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.02,
        step=0.01,
        format="%.3f",
        help="Transaction cost for futures trades"
    )

st.markdown("---")

# Detection Thresholds
st.subheader("Detection Thresholds")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Put-Call Parity**")
    pcp_min_profit = st.number_input("Minimum Profit (₹)", value=5.0, step=1.0, key="pcp_profit")
    pcp_min_deviation = st.slider("Minimum Deviation (%)", 0.01, 1.0, 0.1, 0.01, key="pcp_dev")

with col2:
    st.markdown("**Futures Basis**")
    fb_min_profit = st.number_input("Minimum Profit (₹)", value=5.0, step=1.0, key="fb_profit")
    fb_min_deviation = st.slider("Minimum Basis Deviation (%)", 0.01, 1.0, 0.05, 0.01, key="fb_dev")

st.markdown("---")

# Data Sources
st.subheader("Data Sources")

data_source = st.radio(
    "Primary Data Source",
    options=["NSE Python API", "Yahoo Finance", "Dummy Data (Testing)"],
    index=0,
    help="Select the primary data source for market data"
)

cache_duration = st.slider(
    "Cache Duration (seconds)",
    min_value=10,
    max_value=300,
    value=60,
    step=10,
    help="How long to cache market data before refreshing"
)

st.markdown("---")

# Display Configuration
st.subheader("Display Settings")

col1, col2 = st.columns(2)

with col1:
    show_debug_logs = st.checkbox("Show Debug Logs", value=False)
    show_metadata = st.checkbox("Show Opportunity Metadata", value=True)

with col2:
    auto_refresh = st.checkbox("Enable Auto-Refresh", value=False)
    if auto_refresh:
        refresh_interval = st.slider("Refresh Interval (sec)", 10, 120, 30, 10)

st.markdown("---")

# Summary
st.subheader("Current Configuration Summary")

config_summary = f"""
**Transaction Costs:**
- Equity: {equity_cost}%
- Options: {options_cost}%
- Futures: {futures_cost}%

**Detection Thresholds:**
- PCP Min Profit: ₹{pcp_min_profit}
- PCP Min Deviation: {pcp_min_deviation}%
- Futures Min Profit: ₹{fb_min_profit}
- Futures Min Deviation: {fb_min_deviation}%

**Data Settings:**
- Primary Source: {data_source}
- Cache Duration: {cache_duration}s

**Display Settings:**
- Debug Logs: {'Enabled' if show_debug_logs else 'Disabled'}
- Auto-Refresh: {'Enabled' if auto_refresh else 'Disabled'}
"""

st.code(config_summary, language=None)

st.info("💡 Note: Settings are session-specific and reset when you close the browser.")