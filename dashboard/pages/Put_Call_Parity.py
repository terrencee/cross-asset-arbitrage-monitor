"""
Put-Call Parity Monitor Page
Dedicated page for PCP arbitrage detection
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_acquisition.indian_markets import NSEDataFetcher
from src.data_acquisition.risk_free_rates import RiskFreeRateFetcher
from src.arbitrage_monitors.put_call_parity import PutCallParityMonitor

st.set_page_config(page_title="Put-Call Parity Monitor", page_icon="📊", layout="wide")

st.title("📊 Put-Call Parity Arbitrage Monitor")
st.markdown("Detect mispricing between call options, put options, and underlying stock")
st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    symbol = st.selectbox("Select Index", ['NIFTY', 'BANKNIFTY'], index=0)
    min_profit = st.number_input("Min Profit (₹)", min_value=0.0, value=5.0, step=5.0)
    min_deviation = st.slider("Min Deviation %", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Fetch data
@st.cache_data(ttl=60)
def fetch_pcp_data(symbol):
    fetcher = NSEDataFetcher()
    spot = fetcher.get_spot_price(symbol)
    pcp_data = fetcher.get_option_data_for_arbitrage(symbol)
    rate_fetcher = RiskFreeRateFetcher()
    rfr = rate_fetcher.get_india_rate('91day')
    return spot, pcp_data, rfr

try:
    with st.spinner('Fetching options data...'):
        spot, pcp_data, rfr = fetch_pcp_data(symbol)
    
    if spot is None:
        st.error("Unable to fetch spot price. Please try again.")
        st.stop()
    
    if not pcp_data:
        st.warning("No options data available. Please check data sources.")
        st.stop()
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Spot Price", f"₹{spot:,.2f}")
    with col2:
        st.metric("Risk-Free Rate", f"{rfr*100:.2f}%")
    with col3:
        st.metric("Strikes Analyzed", len(pcp_data))
    
    st.markdown("---")
    
    # Initialize monitor
    monitor = PutCallParityMonitor(
        transaction_costs={'equity': 0.0005, 'options': 0.0005},
        min_profit_threshold=min_profit,
        min_deviation_pct=min_deviation
    )
    
    # Detect opportunities
    opportunities = []
    for data in pcp_data:
        opp = monitor.check_arbitrage(data)
        if opp and opp.is_viable and opp.net_profit >= min_profit:
            opportunities.append(opp)
    
    opportunities.sort(key=lambda x: x.net_profit, reverse=True)
    
    # Display results
    if len(opportunities) == 0:
        st.info("No PCP arbitrage opportunities found. Markets are efficient!")
    else:
        st.success(f"Found {len(opportunities)} Put-Call Parity arbitrage opportunities")
        
        # Table view
        table_data = []
        for opp in opportunities:
            table_data.append({
                'Strike': f"₹{opp.metadata['strike']:,.0f}",
                'Strategy': opp.strategy_name,
                'Call': f"₹{[p for p in opp.positions if p['instrument_type'] == 'Call Option'][0]['price']:.2f}",
                'Put': f"₹{[p for p in opp.positions if p['instrument_type'] == 'Put Option'][0]['price']:.2f}",
                'Net Profit': f"₹{opp.net_profit:.2f}",
                'Profit %': f"{opp.profit_pct:.3f}%",
                'Annualized': f"{opp.metadata['annualized_return']:.2f}%"
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Detailed view
        st.markdown("### Detailed Opportunities")
        for i, opp in enumerate(opportunities):
            with st.expander(f"#{i+1} - {opp.strategy_name} | Strike: ₹{opp.metadata['strike']:,.0f} | Profit: ₹{opp.net_profit:.2f}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Execution Steps:**")
                    for step in opp.execution_steps:
                        st.markdown(f"- {step}")
                
                with col2:
                    st.markdown("**Positions:**")
                    for pos in opp.positions:
                        st.markdown(f"- {pos['action']} {pos['instrument_type']} @ ₹{pos['price']:.2f}")
        
        # Chart
        if len(opportunities) > 0:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[opp.metadata['strike'] for opp in opportunities],
                y=[opp.net_profit for opp in opportunities],
                text=[f"₹{opp.net_profit:.2f}" for opp in opportunities],
                textposition='auto',
                marker_color='#1976d2'
            ))
            fig.update_layout(
                title="Profit by Strike",
                xaxis_title="Strike Price",
                yaxis_title="Net Profit (₹)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)
    st.info("Please try refreshing the page or check your data sources.")

# Theory section
with st.expander("ℹ️ Put-Call Parity Theory"):
    st.markdown("""
    ### Mathematical Relationship
```
    C - P = S·e^(-qT) - K·e^(-rT)
```
    
    Where:
    - C = Call option price
    - P = Put option price  
    - S = Spot price
    - K = Strike price
    - r = Risk-free rate
    - q = Dividend yield
    - T = Time to expiry
    
    ### Arbitrage Strategies
    
    **Conversion Arbitrage** (when call is overpriced):
    1. Buy underlying stock
    2. Buy put option
    3. Sell call option
    4. Result: Guaranteed payoff of K at expiry
    
    **Reversal Arbitrage** (when put is overpriced):
    1. Sell underlying stock short
    2. Buy call option
    3. Sell put option
    4. Result: Guaranteed obligation to buy stock at K
    """)