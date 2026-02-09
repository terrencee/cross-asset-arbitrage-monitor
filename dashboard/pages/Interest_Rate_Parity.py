"""
Covered Interest Rate Parity Monitor Page
True cross-asset arbitrage: FX + Interest Rates
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_acquisition.indian_markets import get_cip_arbitrage_data
from src.arbitrage_monitors.covered_interest_parity import CoveredInterestParityMonitor

st.set_page_config(page_title="Interest Rate Parity Monitor", page_icon="💱", layout="wide")

st.title("💱 Covered Interest Rate Parity Monitor")
st.markdown("**True cross-asset arbitrage:** Detect mispricing between FX markets and interest rate differentials")
st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    currency_pair = st.selectbox("Currency Pair", ['USDINR', 'EURINR', 'GBPINR'], index=0)
    time_period = st.slider("Time Period (months)", min_value=1, max_value=24, value=12, step=1)
    notional = st.number_input("Notional Amount ($)", min_value=1000, value=10000, step=1000)
    
    min_profit = st.number_input("Min Profit (₹)", min_value=0.0, value=100.0, step=50.0)
    min_deviation = st.slider("Min Deviation %", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Fetch data
@st.cache_data(ttl=60)
def fetch_cip_data(pair, months, notional_amt):
    data = get_cip_arbitrage_data(pair, time_period_months=months)
    if data:
        data['notional'] = notional_amt
    return data

try:
    with st.spinner('Fetching FX and interest rate data...'):
        cip_data = fetch_cip_data(currency_pair, time_period, notional)
    
    if cip_data is None:
        st.error("Unable to fetch CIP data")
        st.stop()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Spot Rate", f"{cip_data['spot_rate']:.4f}")
    with col2:
        st.metric("Forward Rate", f"{cip_data['forward_rate']:.4f}")
    with col3:
        st.metric("INR Rate", f"{cip_data['domestic_rate']*100:.2f}%")
    with col4:
        foreign_currency = currency_pair[:3]
        st.metric(f"{foreign_currency} Rate", f"{cip_data['foreign_rate']*100:.2f}%")
    
    st.markdown("---")
    
    # Fair value calculation
    import numpy as np
    T = cip_data['time_period']
    r_d = cip_data['domestic_rate']
    r_f = cip_data['foreign_rate']
    spot = cip_data['spot_rate']
    
    fair_forward = spot * ((1 + r_d * T) / (1 + r_f * T))
    actual_forward = cip_data['forward_rate']
    deviation = actual_forward - fair_forward
    deviation_pct = (deviation / fair_forward) * 100
    
    # Fair value analysis
    st.subheader("Fair Value Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Fair Forward Rate", f"{fair_forward:.4f}")
        st.metric("Actual Forward Rate", f"{actual_forward:.4f}")
    
    with col2:
        st.metric("Deviation", f"{deviation:.4f}", delta=f"{deviation_pct:.3f}%")
        if deviation > 0:
            st.info("💹 Forward is **overvalued** - Borrow-Invest opportunity")
        else:
            st.info("📉 Forward is **undervalued** - Invest-Borrow opportunity")
    
    st.markdown("---")
    
    # Initialize monitor
    monitor = CoveredInterestParityMonitor(
        transaction_costs={
            'fx_spot': 0.0005,
            'fx_forward': 0.0005,
            'borrowing_spread': 0.001
        },
        min_profit_threshold=min_profit,
        min_deviation_pct=min_deviation
    )
    
    # Detect opportunity
    opportunity = monitor.check_arbitrage(cip_data)
    
    if opportunity and opportunity.is_viable:
        st.success(f"✅ CIP Arbitrage Opportunity: {opportunity.strategy_name}")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Gross Profit", f"₹{opportunity.gross_profit:.2f}")
        with col2:
            st.metric("Transaction Costs", f"₹{opportunity.metadata['transaction_costs']:.2f}")
        with col3:
            st.metric("Net Profit", f"₹{opportunity.net_profit:.2f}")
        with col4:
            st.metric("Annualized Return", f"{opportunity.metadata['annualized_return']:.2f}%")
        
        # Execution steps
        st.markdown("### Execution Strategy")
        for i, step in enumerate(opportunity.execution_steps, 1):
            st.markdown(f"**Step {i}:** {step}")
        
        # Position details
        st.markdown("### Position Details")
        positions_data = []
        for pos in opportunity.positions:
            positions_data.append({
                'Action': pos['action'],
                'Instrument': pos['instrument_type'],
                'Amount': f"{pos.get('amount', pos.get('from_amount', 'N/A')):,.2f}",
                'Rate': f"{pos.get('rate', 'N/A')}"
            })
        
        df = pd.DataFrame(positions_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
    else:
        st.info("No CIP arbitrage opportunity. Markets are in equilibrium.")
    
    # Visualization
    st.markdown("### Rate Comparison")
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Spot',
        x=['Exchange Rate'],
        y=[spot],
        text=[f"{spot:.4f}"],
        textposition='auto',
        marker_color='#1976d2'
    ))
    
    fig.add_trace(go.Bar(
        name='Fair Forward',
        x=['Exchange Rate'],
        y=[fair_forward],
        text=[f"{fair_forward:.4f}"],
        textposition='auto',
        marker_color='#4caf50'
    ))
    
    fig.add_trace(go.Bar(
        name='Actual Forward',
        x=['Exchange Rate'],
        y=[actual_forward],
        text=[f"{actual_forward:.4f}"],
        textposition='auto',
        marker_color='#f44336' if deviation > 0 else '#ff9800'
    ))
    
    fig.update_layout(
        barmode='group',
        height=400,
        yaxis_title=f"{currency_pair} Rate",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.exception(e)

# Theory section
with st.expander("ℹ️ Covered Interest Rate Parity Theory"):
    st.markdown("""
    ### True Cross-Asset Arbitrage
    
    CIP links **two asset classes:**
    - **Foreign Exchange Market** (spot & forward rates)
    - **Fixed Income Market** (interest rates)
    
    ### Mathematical Relationship
```
    F/S = (1 + r_domestic × T) / (1 + r_foreign × T)
```
    
    Where:
    - F = Forward exchange rate
    - S = Spot exchange rate
    - r_domestic = Domestic interest rate (e.g., India)
    - r_foreign = Foreign interest rate (e.g., USA)
    - T = Time period in years
    
    ### Arbitrage Strategies
    
    **Borrow-Invest** (when forward is overvalued):
    1. Borrow domestic currency (INR) at r_domestic
    2. Convert to foreign currency (USD) at spot rate
    3. Invest USD at r_foreign
    4. Sell USD forward to lock in return
    5. At maturity: Deliver USD, receive INR, repay loan
    6. Profit = Forward proceeds - Loan repayment
    
    **Invest-Borrow** (when forward is undervalued):
    1. Borrow foreign currency (USD) at r_foreign
    2. Convert to domestic currency (INR) at spot rate
    3. Invest INR at r_domestic
    4. Buy USD forward to lock in repayment cost
    5. At maturity: Receive INR, buy USD via forward, repay loan
    6. Profit = INR investment - Forward cost
    
    ### Why "Cross-Asset"?
    
    Unlike Put-Call Parity (options + equity) or Futures Basis (spot + futures),
    CIP arbitrage requires simultaneous positions in:
    - **FX derivatives** (spot & forward)
    - **Money markets** (deposits & loans)
    
    This makes it truly **cross-asset class** arbitrage.
    """)