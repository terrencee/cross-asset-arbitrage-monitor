"""
Futures Basis Monitor Page
Dedicated page for spot-futures arbitrage detection
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_acquisition.indian_markets import get_futures_arbitrage_data
from src.arbitrage_monitors.futures_basis import FuturesBasisMonitor

st.set_page_config(page_title="Futures Basis Monitor", page_icon="📈", layout="wide")

st.title("📈 Futures Basis Arbitrage Monitor")
st.markdown("Detect mispricing between spot and futures markets (Cost-of-Carry)")
st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    symbol = st.selectbox("Select Index", ['NIFTY', 'BANKNIFTY'], index=0)
    min_profit = st.number_input("Min Profit (₹)", min_value=0.0, value=5.0, step=5.0)
    min_basis_deviation = st.slider("Min Basis Deviation %", min_value=0.01, max_value=1.0, value=0.05, step=0.01)
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Fetch data
@st.cache_data(ttl=60)
def fetch_futures_data(symbol):
    return get_futures_arbitrage_data(symbol)

try:
    with st.spinner('Fetching futures data...'):
        futures_data = fetch_futures_data(symbol)
    
    if futures_data is None:
        st.error("Unable to fetch futures data")
        st.stop()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Spot Price", f"₹{futures_data['spot_price']:,.2f}")
    with col2:
        st.metric("Futures Price", f"₹{futures_data['futures_price']:,.2f}")
    with col3:
        st.metric("Time to Expiry", f"{int(futures_data['time_to_expiry']*365)} days")
    with col4:
        st.metric("Risk-Free Rate", f"{futures_data['risk_free_rate']*100:.2f}%")
    
    st.markdown("---")
    
    # Initialize monitor
    monitor = FuturesBasisMonitor(
        transaction_costs={'equity': 0.0005, 'futures': 0.0002},
        min_profit_threshold=min_profit,
        min_basis_deviation_pct=min_basis_deviation
    )
    
    # Calculate fair value
    import numpy as np
    r = futures_data['risk_free_rate']
    T = futures_data['time_to_expiry']
    spot = futures_data['spot_price']
    fair_futures = spot * np.exp(r * T)
    basis = futures_data['futures_price'] - fair_futures
    basis_pct = (basis / fair_futures) * 100
    
    # Display fair value analysis
    st.subheader("Fair Value Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Fair Futures Price", f"₹{fair_futures:.2f}")
        st.metric("Actual Futures Price", f"₹{futures_data['futures_price']:.2f}")
    
    with col2:
        st.metric("Basis (Actual - Fair)", f"₹{basis:.2f}", delta=f"{basis_pct:.3f}%")
        if basis > 0:
            st.info("📈 Futures are **overpriced** - Cash-and-Carry opportunity")
        else:
            st.info("📉 Futures are **underpriced** - Reverse Cash-and-Carry opportunity")
    
    st.markdown("---")
    
    # Detect opportunity
    opportunity = monitor.check_arbitrage(futures_data)
    
    if opportunity and opportunity.is_viable:
        st.success(f"✅ Arbitrage Opportunity Found: {opportunity.strategy_name}")
        
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
                'Amount': f"₹{pos.get('price', pos.get('amount', 0)):,.2f}",
                'Details': f"{pos.get('quantity', 'N/A')} units" if 'quantity' in pos else f"{pos.get('rate', 'N/A')} rate"
            })
        
        df = pd.DataFrame(positions_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
    else:
        st.info("No arbitrage opportunity. Futures are fairly priced within transaction cost bounds.")
    
    # Visualization
    st.markdown("### Price Comparison")
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Spot',
        x=['Current Price'],
        y=[spot],
        text=[f"₹{spot:.2f}"],
        textposition='auto',
        marker_color='#1976d2'
    ))
    
    fig.add_trace(go.Bar(
        name='Fair Futures',
        x=['Current Price'],
        y=[fair_futures],
        text=[f"₹{fair_futures:.2f}"],
        textposition='auto',
        marker_color='#4caf50'
    ))
    
    fig.add_trace(go.Bar(
        name='Actual Futures',
        x=['Current Price'],
        y=[futures_data['futures_price']],
        text=[f"₹{futures_data['futures_price']:.2f}"],
        textposition='auto',
        marker_color='#f44336' if basis > 0 else '#ff9800'
    ))
    
    fig.update_layout(
        barmode='group',
        height=400,
        xaxis_title="",
        yaxis_title="Price (₹)",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.exception(e)

# Theory section
with st.expander("ℹ Futures Basis Theory"):
    st.markdown("""
    ### Cost-of-Carry Model
```
    F = S × e^((r - q) × T)
```
    
    Where:
    - F = Fair futures price
    - S = Spot price
    - r = Risk-free interest rate
    - q = Dividend yield
    - T = Time to expiry
    
    ### Arbitrage Strategies
    
    **Cash-and-Carry** (when F > Fair):
    1. Borrow money at rate r
    2. Buy underlying at spot price S
    3. Sell futures at F
    4. At expiry: Deliver underlying, receive F
    5. Repay loan with interest
    6. Profit = F - S×e^(rT) - costs
    
    **Reverse Cash-and-Carry** (when F < Fair):
    1. Sell underlying short at S
    2. Lend proceeds at rate r
    3. Buy futures at F
    4. At expiry: Receive S×e^(rT) from loan
    5. Take delivery via futures at F
    6. Profit = S×e^(rT) - F - costs
    """)