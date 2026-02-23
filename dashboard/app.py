"""
Cross-Asset Arbitrage Monitor Dashboard
Main Streamlit Application - Three Monitors

Author: Adhiraj - IIT Roorkee
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_acquisition.indian_markets import NSEDataFetcher, get_futures_arbitrage_data, get_cip_arbitrage_data
from src.data_acquisition.risk_free_rates import RiskFreeRateFetcher
from src.arbitrage_monitors.put_call_parity import PutCallParityMonitor
from src.arbitrage_monitors.futures_basis import FuturesBasisMonitor
from src.arbitrage_monitors.covered_interest_parity import CoveredInterestParityMonitor

# Page configuration
st.set_page_config(
    page_title="Cross-Asset Arbitrage Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .monitor-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 0.3rem;
        font-size: 0.85rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .pcp-badge {
        background-color: #e3f2fd;
        color: #1976d2;
        border: 1px solid #1976d2;
    }
    .futures-badge {
        background-color: #f3e5f5;
        color: #7b1fa2;
        border: 1px solid #7b1fa2;
    }
    .cip-badge {
        background-color: #e8f5e9;
        color: #388e3c;
        border: 1px solid #388e3c;
    }
</style>
""", unsafe_allow_html=True)


def _ensure_defaults():
    # Costs are entered as % in UI, converted to decimals for engine.
    st.session_state.setdefault("equity_cost_pct", 0.05)
    st.session_state.setdefault("options_cost_pct", 0.05)
    st.session_state.setdefault("futures_cost_pct", 0.02)
    st.session_state.setdefault("pcp_min_profit", 5.0)
    st.session_state.setdefault("pcp_min_dev", 0.10)
    st.session_state.setdefault("fb_min_profit", 5.0)
    st.session_state.setdefault("fb_min_dev", 0.05)
    st.session_state.setdefault("cip_min_profit", 100.0)
    st.session_state.setdefault("cip_min_dev", 0.10)
    st.session_state.setdefault("cache_duration", 60)
    st.session_state.setdefault("allow_dummy", True)  # Allow dummy data by default for testing till we get proper api to fetch real option data

def _txn_costs():
    return {
        "equity": st.session_state["equity_cost_pct"] / 100.0,
        "options": st.session_state["options_cost_pct"] / 100.0,
        "futures": st.session_state["futures_cost_pct"] / 100.0,
        "fx_spot": 0.0005,
        "fx_forward": 0.0005,
        "borrowing_spread": 0.001,
    }

_ensure_defaults()


@st.cache_data(ttl=60)
def fetch_all_market_data(symbol='NIFTY', currency_pair='USDINR'):
    """Fetch all market data with caching"""
    # fetcher = NSEDataFetcher(cache_duration_seconds=60)
    fetcher = NSEDataFetcher(
        cache_duration_seconds=int(st.session_state.get("cache_duration", 60)),
        allow_dummy=bool(st.session_state.get("allow_dummy", False))
    )
    
    # Get spot price
    spot = fetcher.get_spot_price(symbol)
    
    # Get options data for PCP
    pcp_data = fetcher.get_option_data_for_arbitrage(symbol)
    
    # Get futures data
    futures_data = get_futures_arbitrage_data(symbol)
    
    # Get CIP data
    cip_data = get_cip_arbitrage_data(currency_pair, time_period_months=12)
    
    # Get risk-free rate
    rate_fetcher = RiskFreeRateFetcher()
    rfr = rate_fetcher.get_india_rate('91day')
    
    return spot, pcp_data, futures_data, cip_data, rfr


@st.cache_resource
def get_arbitrage_monitors():
    """Create all arbitrage monitor instances"""
    '''
    transaction_costs = {
        'equity': 0.0005,
        'options': 0.0005,
        'futures': 0.0002,
        'fx_spot': 0.0005,
        'fx_forward': 0.0005,
        'borrowing_spread': 0.001
    }
    '''
    transaction_costs = _txn_costs()
    
    pcp_monitor = PutCallParityMonitor(
        transaction_costs=transaction_costs,
        # min_profit_threshold=5.0,
        # min_deviation_pct=0.05

        min_profit_threshold=float(st.session_state.get("pcp_min_profit", 5.0)),
        min_deviation_pct=float(st.session_state.get("pcp_min_dev", 0.10))
    )
    
    futures_monitor = FuturesBasisMonitor(
        transaction_costs=transaction_costs,
        # min_profit_threshold=5.0,
        # min_basis_deviation_pct=0.05

        min_profit_threshold=float(st.session_state.get("fb_min_profit", 5.0)),
        min_basis_deviation_pct=float(st.session_state.get("fb_min_dev", 0.05))
    )
    
    cip_monitor = CoveredInterestParityMonitor(
        transaction_costs=transaction_costs,
        # min_profit_threshold=100.0,
        # min_deviation_pct=0.1

        min_profit_threshold=float(st.session_state.get("cip_min_profit", 100.0)),
        min_deviation_pct=float(st.session_state.get("cip_min_dev", 0.10))
    )
    
    return pcp_monitor, futures_monitor, cip_monitor


def display_market_overview(spot, rfr, num_pcp, num_futures, num_cip):
    """Display market overview metrics"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Nifty Spot Price",
            value=f"₹{spot:,.2f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Risk-Free Rate",
            value=f"{rfr*100:.2f}%",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Total Opportunities",
            value=num_pcp + num_futures + num_cip,
            delta=None,
            help=f"PCP: {num_pcp} | Futures: {num_futures} | CIP: {num_cip}"
        )
    
    with col4:
        st.metric(
            label="Last Updated",
            value=datetime.now().strftime("%H:%M:%S"),
            delta=None
        )


def display_opportunity_card(opp, index, monitor_type):
    """Display individual arbitrage opportunity with monitor badge"""
    
    # Badge HTML
    badge_configs = {
        'PCP': ('pcp-badge', 'Put-Call Parity'),
        'Futures': ('futures-badge', 'Futures Basis'),
        'CIP': ('cip-badge', 'Interest Rate Parity')
    }
    badge_class, badge_text = badge_configs[monitor_type]
    
    title = f"#{index+1} - {opp.strategy_name}"
    if monitor_type == "PCP":
        title += f" | Strike: ₹{opp.metadata['strike']:,.0f}"
    elif monitor_type == "CIP":
        title += f" | {opp.metadata['currency_pair']}"
    title += f" | Profit: ₹{opp.net_profit:.2f}"
    
    with st.expander(title, expanded=False):
        # Monitor badge
        st.markdown(f'<span class="monitor-badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)
        
        # Two columns: details and execution
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Opportunity Details")
            
            if monitor_type == "PCP":
                details_df = pd.DataFrame({
                    'Metric': ['Strategy', 'Instrument', 'Spot Price', 'Strike Price', 'Call Price', 'Put Price', 'Time to Expiry'],
                    'Value': [
                        opp.strategy_name,
                        opp.instrument,
                        f"₹{opp.metadata['spot']:,.2f}",
                        f"₹{opp.metadata['strike']:,.2f}",
                        f"₹{[p for p in opp.positions if p['instrument_type'] == 'Call Option'][0]['price']:.2f}",
                        f"₹{[p for p in opp.positions if p['instrument_type'] == 'Put Option'][0]['price']:.2f}",
                        f"{opp.metadata['time_to_expiry_days']:.0f} days"
                    ]
                })
            elif monitor_type == "Futures":
                details_df = pd.DataFrame({
                    'Metric': ['Strategy', 'Instrument', 'Spot Price', 'Futures Price', 'Fair Futures', 'Basis Deviation', 'Time to Expiry'],
                    'Value': [
                        opp.strategy_name,
                        opp.instrument,
                        f"₹{opp.metadata['spot']:,.2f}",
                        f"₹{opp.metadata['futures']:,.2f}",
                        f"₹{opp.metadata['fair_futures']:,.2f}",
                        f"{opp.metadata['basis_deviation_pct']:.3f}%",
                        f"{opp.metadata['time_to_expiry_days']:.0f} days"
                    ]
                })
            else:  # CIP
                details_df = pd.DataFrame({
                    'Metric': ['Strategy', 'Currency Pair', 'Spot Rate', 'Forward Rate', 'Fair Forward', 'Deviation', 'Time Period'],
                    'Value': [
                        opp.strategy_name,
                        opp.metadata['currency_pair'],
                        f"{opp.metadata['spot_rate']:.4f}",
                        f"{opp.metadata['forward_rate']:.4f}",
                        f"{opp.metadata['fair_forward']:.4f}",
                        f"{opp.metadata['deviation_pct']:.3f}%",
                        f"{opp.metadata['time_period_days']:.0f} days"
                    ]
                })

            details_df = details_df.astype(str)
            

            
            st.dataframe(details_df, use_container_width=True, hide_index=True)
            
            # Profitability metrics
            st.markdown("### Profitability")
            profit_col1, profit_col2, profit_col3 = st.columns(3)
            
            with profit_col1:
                st.metric("Gross Profit", f"₹{opp.gross_profit:.2f}")
            with profit_col2:
                st.metric("Transaction Cost", f"₹{opp.metadata['transaction_costs']:.2f}")
            with profit_col3:
                st.metric("Net Profit", f"₹{opp.net_profit:.2f}")
            
            st.metric("Profit %", f"{opp.profit_pct:.3f}%")
            st.metric("Annualized Return", f"{opp.metadata['annualized_return']:.2f}%")
            st.metric("Confidence Score", f"{opp.confidence_score:.1%}")
        
        with col2:
            st.markdown("### Execution Steps")
            
            for i, step in enumerate(opp.execution_steps, 1):
                st.markdown(f"**{i}.** {step}")
            
            st.markdown("---")
            st.markdown("### Position Details")
            
            positions_data = []
            for pos in opp.positions:
                positions_data.append({
                    'Action': pos['action'],
                    'Instrument': pos['instrument_type'],
                    'Amount': f"₹{pos.get('price', pos.get('amount', pos.get('from_amount', 0))):,.2f}",
                    'Details': str(pos.get('rate', pos.get('quantity', 'N/A')))
                })
            
            positions_df = pd.DataFrame(positions_data)
            positions_df = positions_df.astype(str)
            st.dataframe(positions_df, use_container_width=True, hide_index=True)


def create_combined_profit_chart(pcp_opportunities, futures_opportunities, cip_opportunities):
    """Create combined profit distribution chart"""
    
    if not pcp_opportunities and not futures_opportunities and not cip_opportunities:
        return None
    
    fig = go.Figure()
    
    # Add PCP opportunities
    if pcp_opportunities:
        pcp_strikes = [opp.metadata['strike'] for opp in pcp_opportunities]
        pcp_profits = [opp.net_profit for opp in pcp_opportunities]
        
        fig.add_trace(go.Scatter(
            x=pcp_strikes,
            y=pcp_profits,
            mode='markers',
            name='Put-Call Parity',
            marker=dict(size=10, color='#1976d2'),
            hovertemplate='<b>PCP</b><br>Strike: ₹%{x:,.0f}<br>Profit: ₹%{y:.2f}<extra></extra>'
        ))
    
    # Add futures opportunities
    if futures_opportunities:
        futures_x = [25000 + i*100 for i in range(len(futures_opportunities))]  # Dummy x-values
        futures_profits = [opp.net_profit for opp in futures_opportunities]
        
        fig.add_trace(go.Scatter(
            x=futures_x,
            y=futures_profits,
            mode='markers',
            name='Futures Basis',
            marker=dict(size=12, color='#7b1fa2', symbol='diamond'),
            hovertemplate='<b>Futures</b><br>Profit: ₹%{y:.2f}<extra></extra>'
        ))
    
    # Add CIP opportunities
    if cip_opportunities:
        cip_x = [26000 + i*100 for i in range(len(cip_opportunities))]  # Dummy x-values
        cip_profits = [opp.net_profit for opp in cip_opportunities]
        
        fig.add_trace(go.Scatter(
            x=cip_x,
            y=cip_profits,
            mode='markers',
            name='Interest Rate Parity',
            marker=dict(size=12, color='#388e3c', symbol='square'),
            hovertemplate='<b>CIP</b><br>Profit: ₹%{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Profit Distribution by Strategy",
        xaxis_title="Strike/Index",
        yaxis_title="Net Profit (₹)",
        template="plotly_white",
        height=400,
        showlegend=True
    )
    
    return fig


def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<div class="main-header">Cross-Asset Arbitrage Monitor</div>', unsafe_allow_html=True)
    st.markdown("**Real-time arbitrage detection across Options, Futures, and FX markets**")
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        
        # Symbol selection
        symbol = st.selectbox("Select Index", ['NIFTY', 'BANKNIFTY'], index=0)
        currency_pair = st.selectbox("Select Currency Pair", ['USDINR', 'EURINR', 'GBPINR'], index=0)
        
        # Monitor selection
        st.subheader("Active Monitors")
        enable_pcp = st.checkbox("Put-Call Parity", value=True)
        enable_futures = st.checkbox("Futures Basis", value=True)
        enable_cip = st.checkbox("Interest Rate Parity", value=True)
        
        # Refresh controls
        st.markdown("---")
        auto_refresh = st.checkbox("Auto-refresh", value=False)
        refresh_interval = st.slider(
            "Refresh interval (seconds)",
            min_value=10,
            max_value=120,
            value=30,
            step=10,
            disabled=not auto_refresh
        )
        
        # Manual refresh button
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Filter controls
        st.subheader("Filters")
        min_profit = st.number_input("Minimum Profit (₹)", min_value=0.0, value=5.0, step=5.0)
        show_only_viable = st.checkbox("Show only profitable", value=True)
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        **Developed by:** Adhiraj  
        **Institution:** IIT Roorkee  
        **Course:** Financial Engineering
        
        **Active Monitors:**
        - ✅ Put-Call Parity (Options)
        - ✅ Futures Basis (Spot-Futures)
        - ✅ Interest Rate Parity (FX + Rates)
        """)
    
    # Main content area
    try:
        # Fetch data
        with st.spinner('Fetching market data...'):
            spot, pcp_data, futures_data, cip_data, rfr = fetch_all_market_data(symbol, currency_pair)
        
        if spot is None:
            st.error("Unable to fetch spot price. Please try again.")
            return
        
        # Get monitors
        pcp_monitor, futures_monitor, cip_monitor = get_arbitrage_monitors()
        pcp_monitor.min_profit_threshold = min_profit
        futures_monitor.min_profit_threshold = min_profit
        cip_monitor.min_profit_threshold = max(min_profit, 100.0)  # CIP has higher threshold
        
        # Clear previous opportunities
        pcp_monitor.clear_opportunities()
        futures_monitor.clear_opportunities()
        cip_monitor.clear_opportunities()
        
        # Detect PCP opportunities
        pcp_opportunities = []
        if enable_pcp and pcp_data:
            for data in pcp_data:
                opp = pcp_monitor.check_arbitrage(data)
                if opp is not None:
                    if show_only_viable and not opp.is_viable:
                        continue
                    if opp.net_profit >= min_profit:
                        pcp_opportunities.append(opp)
        
        # Detect Futures opportunities
        futures_opportunities = []
        if enable_futures and futures_data:
            opp = futures_monitor.check_arbitrage(futures_data)
            if opp is not None:
                if show_only_viable and not opp.is_viable:
                    pass
                elif opp.net_profit >= min_profit:
                    futures_opportunities.append(opp)
        
        # Detect CIP opportunities
        cip_opportunities = []
        if enable_cip and cip_data:
            opp = cip_monitor.check_arbitrage(cip_data)
            if opp is not None:
                if show_only_viable and not opp.is_viable:
                    pass
                elif opp.net_profit >= cip_monitor.min_profit_threshold:
                    cip_opportunities.append(opp)
        
        # Sort all opportunities by profit
        pcp_opportunities.sort(key=lambda x: x.net_profit, reverse=True)
        futures_opportunities.sort(key=lambda x: x.net_profit, reverse=True)
        cip_opportunities.sort(key=lambda x: x.net_profit, reverse=True)
        
        total_opportunities = len(pcp_opportunities) + len(futures_opportunities) + len(cip_opportunities)
        
        # Display market overview
        display_market_overview(spot, rfr, len(pcp_opportunities), len(futures_opportunities), len(cip_opportunities))
        
        st.markdown("---")
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 All Opportunities", "📈 Analytics", "ℹ️ Documentation"])
        
        with tab1:
            if total_opportunities == 0:
                st.info("""
                No arbitrage opportunities found meeting your criteria.
                
                This is normal in efficient markets. Try:
                - Lowering the minimum profit threshold
                - Enabling all monitors
                - Checking during market hours for live data
                """)
            else:
                st.success(f"Found {total_opportunities} arbitrage opportunities ({len(pcp_opportunities)} PCP, {len(futures_opportunities)} Futures, {len(cip_opportunities)} CIP)")
                
                # Combine and display all opportunities
                all_opportunities = []
                
                for opp in pcp_opportunities:
                    all_opportunities.append(('PCP', opp))
                for opp in futures_opportunities:
                    all_opportunities.append(('Futures', opp))
                for opp in cip_opportunities:
                    all_opportunities.append(('CIP', opp))
                
                # Sort by profit
                all_opportunities.sort(key=lambda x: x[1].net_profit, reverse=True)
                
                # Display each opportunity
                for i, (monitor_type, opp) in enumerate(all_opportunities):
                    display_opportunity_card(opp, i, monitor_type)
        
        with tab2:
            st.subheader("Analytics Dashboard")
            
            if total_opportunities > 0:
                # Combined profit chart
                fig = create_combined_profit_chart(pcp_opportunities, futures_opportunities, cip_opportunities)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Summary Statistics")
                    
                    all_opps = pcp_opportunities + futures_opportunities + cip_opportunities
                    total_profit = sum(opp.net_profit for opp in all_opps)
                    avg_profit = total_profit / len(all_opps)
                    max_profit = max(opp.net_profit for opp in all_opps)
                    
                    summary_df = pd.DataFrame({
                        'Metric': [
                            'Total Opportunities',
                            'PCP Opportunities',
                            'Futures Opportunities',
                            'CIP Opportunities',
                            'Total Potential Profit',
                            'Average Profit',
                            'Maximum Profit'
                        ],
                        'Value': [
                            total_opportunities,
                            len(pcp_opportunities),
                            len(futures_opportunities),
                            len(cip_opportunities),
                            f"₹{total_profit:.2f}",
                            f"₹{avg_profit:.2f}",
                            f"₹{max_profit:.2f}"
                        ]
                    })
                    summary_df["Value"] = summary_df["Value"].astype(str)

                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("### Monitor Distribution")
                    
                    monitor_df = pd.DataFrame({
                        'Monitor': ['Put-Call Parity', 'Futures Basis', 'Interest Rate Parity'],
                        'Count': [len(pcp_opportunities), len(futures_opportunities), len(cip_opportunities)]
                    })
                    
                    fig_pie = px.pie(
                        monitor_df,
                        values='Count',
                        names='Monitor',
                        title='Opportunities by Monitor',
                        color_discrete_sequence=['#1976d2', '#7b1fa2', '#388e3c']
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No data available for analytics")
        
        with tab3:
            st.subheader("Arbitrage Strategies")
            
            st.markdown("""
            ### Three Parity Relationships
            
            This system monitors **three fundamental parity conditions** across different asset classes:
            
            1. **Put-Call Parity** (Options Market)
            2. **Futures Basis** (Spot-Futures)
            3. **Covered Interest Rate Parity** (FX + Interest Rates) ← True Cross-Asset
            """)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **Put-Call Parity**
```
                C - P = S·e^(-qT) - K·e^(-rT)
```
                Options vs Equity
                """)
            
            with col2:
                st.markdown("""
                **Futures Basis**
```
                F = S·e^((r-q)T)
```
                Spot vs Futures
                """)
            
            with col3:
                st.markdown("""
                **Interest Rate Parity**
```
                F/S = (1+r_d)/(1+r_f)
```
                FX + Interest Rates
                """)
            
            st.markdown("---")
            st.markdown("""
            ### Transaction Costs
            - Equity: 0.05% | Options: 0.05% | Futures: 0.02%
            - FX Spot: 0.05% | FX Forward: 0.05% | Borrowing Spread: 0.10%
            
            ### Data Sources
            - **NSE API** (primary), **Yahoo Finance** (backup), **Dummy data** (testing)
            """)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()