"""
Put-Call Parity Arbitrage Monitor

Detects violations of put-call parity:
C - P = S - K·e^(-rT)

Implements two strategies:
1. Conversion: Buy stock + Buy put + Sell call (when call overpriced)
2. Reversal: Sell stock + Buy call + Sell put (when put overpriced)
"""

import numpy as np
from datetime import datetime
from typing import Dict, Optional, List
from src.arbitrage_monitors.base_monitor import BaseArbitrageMonitor, ArbitrageOpportunity
from src.pricing_models.options_pricing import BlackScholesModel, SyntheticPositions
from src.utils.validators import DataValidator
from src.utils.logger import log


class PutCallParityMonitor(BaseArbitrageMonitor):
    """
    Monitor put-call parity violations for arbitrage
    """
    
    def __init__(self, transaction_costs: Dict[str, float], 
                 min_profit_threshold: float = 10.0,
                 min_deviation_pct: float = 0.1):
        """
        Args:
            transaction_costs: Transaction cost rates by instrument
            min_profit_threshold: Minimum net profit to flag (₹)
            min_deviation_pct: Minimum % deviation from parity to investigate
        """
        super().__init__(transaction_costs, min_profit_threshold)
        self.min_deviation_pct = min_deviation_pct
        self.bs_model = BlackScholesModel()
        self.synthetic_model = SyntheticPositions(self.bs_model)
        
        log.info(f"PutCallParity monitor initialized with {min_deviation_pct}% deviation threshold")
    
    def check_arbitrage(self, market_data: Dict) -> Optional[ArbitrageOpportunity]:
        """
        Check for put-call parity arbitrage
        
        Expected market_data structure:
        {
            'spot_price': float,
            'strike': float,
            'call_price': float,
            'put_price': float,
            'time_to_expiry': float,  # years
            'risk_free_rate': float,   # annualized
            'dividend_yield': float,   # annualized (optional, default 0)
            'market': str,             # e.g., 'NSE', 'NYSE'
            'instrument': str,         # e.g., 'NIFTY', 'SPY'
            'expiry_date': str,        # e.g., '2025-02-13'
            'call_bid': float,         # (optional) for better execution modeling
            'call_ask': float,
            'put_bid': float,
            'put_ask': float
        }
        """
        # Extract data
        S = market_data['spot_price']
        K = market_data['strike']
        C_market = market_data['call_price']
        P_market = market_data['put_price']
        T = market_data['time_to_expiry']
        r = market_data['risk_free_rate']
        q = market_data.get('dividend_yield', 0)
        F = market_data.get('futures_price', None)
        
        market = market_data.get('market', 'Unknown')
        instrument = market_data.get('instrument', 'Unknown')
        
        # Validate data
        is_valid, error_msg = DataValidator.validate_option_data(
            S, K, C_market, P_market, T
        )
        
        if not is_valid:
            log.warning(f"Invalid data for {instrument}: {error_msg}")
            return None
        
        # Calculate put-call parity components
        parity_lhs = C_market - P_market  # Left-hand side
        # parity_rhs = S * np.exp(-q * T) - K * np.exp(-r * T)  # Right-hand side
        # Right-hand side:
        # If futures price is provided, use futures-based parity (more appropriate for index options).
        # C - P ≈ e^{-rT}(F - K)
        if F is not None:
            parity_rhs = np.exp(-r * T) * (float(F) - K)
        else:
            parity_rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        
        deviation = parity_lhs - parity_rhs
        # deviation_pct = abs(deviation) / C_market * 100  # As % of call price
        # Safer scaling than dividing by C (C can be tiny for OTM calls)
        scale = max(abs(parity_rhs), 1.0)
        deviation_pct = abs(deviation) / scale * 100
        
        # Check if deviation exceeds minimum threshold
        if deviation_pct < self.min_deviation_pct:
            log.debug(f"{instrument} PCP deviation {deviation_pct:.3f}% < threshold {self.min_deviation_pct}%")
            return None
        
        # Calculate synthetic prices for verification
        C_synthetic = self.synthetic_model.synthetic_call(P_market, S, K, T, r)
        P_synthetic = self.synthetic_model.synthetic_put(C_market, S, K, T, r)
        
        # Determine arbitrage strategy
        if deviation > 0:
            # Call overpriced relative to put → CONVERSION
            opportunity = self._conversion_arbitrage(
                S, K, C_market, P_market, T, r, q,
                market, instrument, market_data
            )
        else:
            # Put overpriced relative to call → REVERSAL
            opportunity = self._reversal_arbitrage(
                S, K, C_market, P_market, T, r, q,
                market, instrument, market_data
            )
        
        # Log and return
        if opportunity:
            self.log_opportunity(opportunity)
            return opportunity
        
        return None
    
    def _conversion_arbitrage(self, S: float, K: float, C: float, P: float,
                             T: float, r: float, q: float,
                             market: str, instrument: str,
                             market_data: Dict) -> Optional[ArbitrageOpportunity]:
        """
        Conversion Arbitrage (when call is overpriced)
        
        Strategy:
        1. BUY stock at S
        2. BUY put at P (protective put)
        3. SELL call at C (covered call)
        
        Payoff at expiry:
        - If S_T > K: Stock sold at K (call exercised), put expires worthless
        - If S_T < K: Stock sold at K (put exercised), call expires worthless
        - Either way: Guaranteed K at expiry
        
        Profit = K·e^(-rT) - (S - C + P) - transaction_costs
        """
        # Calculate position costs
        stock_cost = S
        put_cost = P
        call_proceeds = C  # We receive this
        
        # Net investment (what we pay upfront)
        net_investment = stock_cost + put_cost - call_proceeds
        
        # Transaction costs
        tc_stock = self.calculate_transaction_cost(S, 'equity')
        tc_put = self.calculate_transaction_cost(P, 'options')
        tc_call = self.calculate_transaction_cost(C, 'options')
        total_tc = tc_stock + tc_put + tc_call
        
        # Guaranteed payoff at expiry
        guaranteed_payoff = K
        
        # Present value of guaranteed payoff
        pv_payoff = guaranteed_payoff * np.exp(-r * T)
        
        # Gross profit (before costs)
        gross_profit = pv_payoff - net_investment
        
        # Net profit (after costs)
        net_profit = gross_profit - total_tc
        
        # Profit percentage
        profit_pct = (net_profit / net_investment) * 100 if net_investment > 0 else 0
        
        # Check if profitable
        if net_profit < self.min_profit_threshold:
            return None
        
        # Build positions list
        positions = [
            {
                'action': 'BUY',
                'instrument_type': 'Stock',
                'symbol': instrument,
                'quantity': 1,
                'price': S,
                'cost': stock_cost + tc_stock
            },
            {
                'action': 'BUY',
                'instrument_type': 'Put Option',
                'symbol': f"{instrument} {K}P",
                'strike': K,
                'expiry': market_data.get('expiry_date', 'N/A'),
                'quantity': 1,
                'price': P,
                'cost': put_cost + tc_put
            },
            {
                'action': 'SELL',
                'instrument_type': 'Call Option',
                'symbol': f"{instrument} {K}C",
                'strike': K,
                'expiry': market_data.get('expiry_date', 'N/A'),
                'quantity': 1,
                'price': C,
                'proceeds': call_proceeds - tc_call
            }
        ]
        
        # Build execution steps
        execution_steps = [
            f"Step 1: BUY 1 lot of {instrument} at ₹{S:.2f}",
            f"Step 2: BUY 1 {K} Put option at ₹{P:.2f}",
            f"Step 3: SELL 1 {K} Call option at ₹{C:.2f}",
            f"Step 4: Net cash outflow: ₹{net_investment:.2f}",
            f"Step 5: At expiry (T={T:.3f}y): Guaranteed receipt of ₹{guaranteed_payoff:.2f}",
            f"Step 6: Net profit: ₹{net_profit:.2f} ({profit_pct:.2f}%)"
        ]
        
        # Metadata
        metadata = {
            'parity_deviation': gross_profit,
            'transaction_costs': total_tc,
            'annualized_return': (net_profit / net_investment) / T * 100 if T > 0 else 0,
            'risk_free_rate': r,
            'time_to_expiry_days': T * 365,
            'strike': K,
            'spot': S
        }
        
        # Create opportunity object
        opportunity = ArbitrageOpportunity(
            strategy_name="Conversion Arbitrage (PCP)",
            market=market,
            instrument=instrument,
            detection_time=datetime.now(),
            gross_profit=gross_profit,
            net_profit=net_profit,
            profit_pct=profit_pct,
            positions=positions,
            execution_steps=execution_steps,
            confidence_score=self._calculate_confidence(market_data),
            metadata=metadata
        )
        
        return opportunity
    
    def _reversal_arbitrage(self, S: float, K: float, C: float, P: float,
                           T: float, r: float, q: float,
                           market: str, instrument: str,
                           market_data: Dict) -> Optional[ArbitrageOpportunity]:
        """
        Reversal Arbitrage (when put is overpriced)
        
        Strategy:
        1. SELL (short) stock at S
        2. BUY call at C
        3. SELL put at P
        
        Payoff at expiry:
        - If S_T > K: Stock bought at K (call exercised), put expires worthless
        - If S_T < K: Stock bought at K (put assigned), call expires worthless
        - Either way: Guaranteed to buy back stock at K
        
        Profit = (S - C + P) - K·e^(-rT) - transaction_costs
        """
        # Calculate position proceeds/costs
        stock_proceeds = S  # We receive this from short sale
        call_cost = C
        put_proceeds = P  # We receive this
        
        # Net credit (what we receive upfront)
        net_credit = stock_proceeds - call_cost + put_proceeds
        
        # Transaction costs
        tc_stock = self.calculate_transaction_cost(S, 'equity')
        tc_call = self.calculate_transaction_cost(C, 'options')
        tc_put = self.calculate_transaction_cost(P, 'options')
        total_tc = tc_stock + tc_call + tc_put
        
        # Guaranteed obligation at expiry (must buy back stock at K)
        guaranteed_obligation = K
        
        # Present value of obligation
        pv_obligation = guaranteed_obligation * np.exp(-r * T)
        
        # Gross profit
        gross_profit = net_credit - pv_obligation
        
        # Net profit
        net_profit = gross_profit - total_tc
        
        # Profit percentage
        profit_pct = (net_profit / pv_obligation) * 100 if pv_obligation > 0 else 0
        
        # Check viability
        if net_profit < self.min_profit_threshold:
            return None
        
        # Build positions
        positions = [
            {
                'action': 'SELL (Short)',
                'instrument_type': 'Stock',
                'symbol': instrument,
                'quantity': 1,
                'price': S,
                'proceeds': stock_proceeds - tc_stock
            },
            {
                'action': 'BUY',
                'instrument_type': 'Call Option',
                'symbol': f"{instrument} {K}C",
                'strike': K,
                'expiry': market_data.get('expiry_date', 'N/A'),
                'quantity': 1,
                'price': C,
                'cost': call_cost + tc_call
            },
            {
                'action': 'SELL',
                'instrument_type': 'Put Option',
                'symbol': f"{instrument} {K}P",
                'strike': K,
                'expiry': market_data.get('expiry_date', 'N/A'),
                'quantity': 1,
                'price': P,
                'proceeds': put_proceeds - tc_put
            }
        ]
        
        # Execution steps
        execution_steps = [
            f"Step 1: SHORT SELL 1 lot of {instrument} at ₹{S:.2f} (receive cash)",
            f"Step 2: BUY 1 {K} Call option at ₹{C:.2f} (pay premium)",
            f"Step 3: SELL 1 {K} Put option at ₹{P:.2f} (receive premium)",
            f"Step 4: Net cash inflow: ₹{net_credit:.2f}",
            f"Step 5: At expiry (T={T:.3f}y): Must buy back stock at ₹{guaranteed_obligation:.2f}",
            f"Step 6: Net profit: ₹{net_profit:.2f} ({profit_pct:.2f}%)"
        ]
        
        # Metadata
        metadata = {
            'parity_deviation': abs(gross_profit),
            'transaction_costs': total_tc,
            'annualized_return': (net_profit / pv_obligation) / T * 100 if T > 0 else 0,
            'risk_free_rate': r,
            'time_to_expiry_days': T * 365,
            'strike': K,
            'spot': S
        }
        
        opportunity = ArbitrageOpportunity(
            strategy_name="Reversal Arbitrage (PCP)",
            market=market,
            instrument=instrument,
            detection_time=datetime.now(),
            gross_profit=gross_profit,
            net_profit=net_profit,
            profit_pct=profit_pct,
            positions=positions,
            execution_steps=execution_steps,
            confidence_score=self._calculate_confidence(market_data),
            metadata=metadata
        )
        
        return opportunity
    
    def _calculate_confidence(self, market_data: Dict) -> float:
        """
        Calculate confidence score for the opportunity (0-1)
        
        Factors affecting confidence:
        - Bid-ask spread (tighter = more confident)
        - Time to expiry (more time = less confident due to market changes)
        - Liquidity indicators
        """
        confidence = 1.0
        
        # Penalize if bid-ask spreads are too wide
        if 'call_bid' in market_data and 'call_ask' in market_data:
            call_spread = (market_data['call_ask'] - market_data['call_bid']) / market_data['call_price']
            if call_spread > 0.02:  # >2% spread
                confidence *= 0.9
        
        # Penalize longer expirations (more time for market to close gap)
        T = market_data['time_to_expiry']
        if T > 0.25:  # >3 months
            confidence *= 0.95
        
        return confidence