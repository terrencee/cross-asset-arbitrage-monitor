"""
Documentation and Help Page
"""

import streamlit as st

st.set_page_config(page_title="Documentation", page_icon="📚", layout="wide")

st.title("📚 Documentation")
st.markdown("Complete guide to the Cross-Asset Arbitrage Monitor")
st.markdown("---")

# Table of Contents
st.sidebar.markdown("## Contents")
st.sidebar.markdown("- [Overview](#overview)")
st.sidebar.markdown("- [Arbitrage Strategies](#arbitrage-strategies)")
st.sidebar.markdown("- [How to Use](#how-to-use)")
st.sidebar.markdown("- [Technical Details](#technical-details)")
st.sidebar.markdown("- [FAQ](#faq)")

# Overview
st.header("Overview")
st.markdown("""
The **Cross-Asset Arbitrage Monitor** is a real-time system that detects mispricing across
options and futures markets by checking fundamental parity relationships.

**Key Features:**
- ✅ Put-Call Parity arbitrage detection
- ✅ Futures Basis (Cost-of-Carry) arbitrage detection
- ✅ Real-time data from NSE
- ✅ Automatic profit calculation including transaction costs
- ✅ Detailed execution steps for each opportunity
- ✅ Professional analytics and visualization

**Developed by:** Adhiraj | **Institution:** IIT Roorkee | **Course:** Financial Engineering
""")

st.markdown("---")

# Arbitrage Strategies
st.header("Arbitrage Strategies")

tab1, tab2 = st.tabs(["Put-Call Parity", "Futures Basis"])

with tab1:
    st.markdown("""
    ### Put-Call Parity
    
    **Fundamental Relationship:**
    
    The price of a call option minus the price of a put option should equal the present value
    of the difference between the spot price and strike price:
```
    C - P = S·e^(-qT) - K·e^(-rT)
```
    
    **Variables:**
    - `C` = Call option price
    - `P` = Put option price
    - `S` = Current spot price
    - `K` = Strike price
    - `r` = Risk-free interest rate
    - `q` = Dividend yield
    - `T` = Time to expiry (in years)
    
    ---
    
    ### Strategy 1: Conversion Arbitrage
    
    **When:** Call is overpriced relative to put
    
    **Action:**
    1. Buy the underlying stock at spot price S
    2. Buy a put option at strike K
    3. Sell a call option at strike K
    
    **Result:** At expiry, you deliver the stock and receive K (guaranteed)
    
    **Profit:** K·e^(-rT) - (S + P - C) - transaction costs
    
    ---
    
    ### Strategy 2: Reversal Arbitrage
    
    **When:** Put is overpriced relative to call
    
    **Action:**
    1. Sell (short) the underlying stock at spot price S
    2. Buy a call option at strike K
    3. Sell a put option at strike K
    
    **Result:** At expiry, you buy back the stock at K (guaranteed)
    
    **Profit:** (S - C + P) - K·e^(-rT) - transaction costs
    """)

with tab2:
    st.markdown("""
    ### Futures Basis (Cost-of-Carry)
    
    **Fundamental Relationship:**
    
    The fair price of a futures contract should equal the spot price adjusted for the cost
    of carrying the underlying asset until expiry:
```
    F = S × e^((r - q) × T)
```
    
    **Variables:**
    - `F` = Futures price
    - `S` = Current spot price
    - `r` = Risk-free interest rate
    - `q` = Dividend yield
    - `T` = Time to expiry (in years)
    
    ---
    
    ### Strategy 1: Cash-and-Carry Arbitrage
    
    **When:** Futures price > Fair futures price (futures overpriced)
    
    **Action:**
    1. Borrow money at risk-free rate r
    2. Buy the underlying asset at spot price S
    3. Sell futures contract at price F
    
    **Result:** At expiry, deliver the asset and receive F, repay loan
    
    **Profit:** F - S·e^(rT) - transaction costs
    
    ---
    
    ### Strategy 2: Reverse Cash-and-Carry Arbitrage
    
    **When:** Futures price < Fair futures price (futures underpriced)
    
    **Action:**
    1. Sell (short) the underlying asset at spot price S
    2. Lend the proceeds at risk-free rate r
    3. Buy futures contract at price F
    
    **Result:** At expiry, receive S·e^(rT) from loan, buy asset via futures at F
    
    **Profit:** S·e^(rT) - F - transaction costs
    """)

st.markdown("---")

# How to Use
st.header("How to Use")

st.markdown("""
### Quick Start Guide

1. **Navigate to a Monitor Page**
   - Use the sidebar to select "Put-Call Parity" or "Futures Basis"
   - Or use "All Opportunities" to see everything

2. **Configure Settings**
   - Adjust minimum profit threshold
   - Set deviation percentage for detection sensitivity
   - Enable/disable specific monitors

3. **Review Opportunities**
   - Opportunities are listed by profitability
   - Click to expand for detailed execution steps
   - Check confidence score and annualized return

4. **Understand the Metrics**
   - **Gross Profit:** Total profit before costs
   - **Transaction Costs:** Fees for executing trades
   - **Net Profit:** Final profit after all costs
   - **Profit %:** Return as percentage of capital employed
   - **Annualized Return:** Profit extrapolated to annual rate

5. **Execute (In Real Trading)**
   - Follow the step-by-step execution guide
   - Monitor positions until expiry
   - Realize the guaranteed profit
""")

st.markdown("---")

# Technical Details
st.header("Technical Details")

st.markdown("""
### Data Sources

**Primary: NSE Python API**
- Real-time options chain data
- Futures contract prices
- Most reliable during market hours

**Backup: Yahoo Finance**
- Used when NSE API fails
- Spot prices for major indices
- Less real-time but more stable

**Fallback: Dummy Data**
- Generated for testing
- Uses realistic pricing models
- Allows development without live connection

### Transaction Costs (Default)

| Asset Class | Cost |
|------------|------|
| Equity | 0.05% |
| Options | 0.05% |
| Futures | 0.02% |

### Detection Algorithm

1. Fetch market data (spot, options, futures)
2. Calculate theoretical fair prices
3. Compare with actual market prices
4. Identify deviations above threshold
5. Calculate profit including costs
6. Generate execution strategy
7. Score opportunity confidence

### Limitations

- **Execution Risk:** Prices may move before execution
- **Impact Costs:** Large trades affect market prices
- **Funding Costs:** Borrowing rates may vary
- **Slippage:** Bid-ask spreads reduce profits
- **Dummy Data:** May show artificial opportunities
""")

st.markdown("---")

# FAQ
st.header("FAQ")

with st.expander("❓ Why am I seeing 'No opportunities found'?"):
    st.markdown("""
    This is normal! Modern financial markets are very efficient, meaning arbitrage
    opportunities are rare and quickly eliminated. You might see opportunities when:
    - Markets are volatile
    - Using dummy data (for testing)
    - Setting low profit thresholds
    - Enabling all monitors
    """)

with st.expander("❓ How accurate is the dummy data?"):
    st.markdown("""
    Dummy data uses simplified pricing models (Black-Scholes for options, cost-of-carry
    for futures). It's designed to be realistic enough for testing but may show arbitrage
    opportunities that wouldn't exist with real market data.
    """)

with st.expander("❓ Can I use this for real trading?"):
    st.markdown("""
    This system is designed for **educational purposes**. For real trading:
    - Verify opportunities with live market data
    - Account for execution risk and slippage
    - Consider your funding costs
    - Factor in margin requirements
    - Consult with financial advisors
    """)

with st.expander("❓ What is 'confidence score'?"):
    st.markdown("""
    Confidence score (0-100%) indicates how reliable an opportunity is based on:
    - Magnitude of deviation from fair value
    - Bid-ask spread (tighter = more confident)
    - Time to expiry (shorter = more confident)
    - Market liquidity indicators
    """)

with st.expander("❓ Why do transaction costs matter so much?"):
    st.markdown("""
    Arbitrage profits are typically very small (< 1%). Transaction costs of 0.05-0.10%
    can easily eliminate the entire profit. This is why:
    1. We calculate net profit after costs
    2. We set minimum profit thresholds
    3. Real arbitrage requires low-cost execution
    """)

st.markdown("---")

st.info("""
💡 **Need Help?** 
For technical support or questions about the system, refer to the project documentation
or contact the development team.
""")