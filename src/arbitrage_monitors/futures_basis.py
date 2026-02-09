"""
Futures Basis / Cost-of-Carry Arbitrage Monitor

Detects mispricing between spot and futures markets based on the
cost-of-carry relationship.

Author: Adhiraj - IIT Roorkee
"""

import numpy as np
from typing import Dict, Optional
from datetime import datetime

from src.arbitrage_monitors.base_monitor import BaseArbitrageMonitor, ArbitrageOpportunity
from src.utils.logger import log


class FuturesBasisMonitor(BaseArbitrageMonitor):
    """
    Monitor for futures basis arbitrage (cost-of-carry relationship)
    
    Fundamental relationship:
        F = S * e^((r - q) * T)
    
    Where:
        F = Futures price
        S = Spot price
        r = Risk-free rate
        q = Dividend yield
        T = Time to expiry
    
    Arbitrage strategies:
        1. Cash-and-Carry: Buy spot, sell futures (when F > fair value)
        2. Reverse Cash-and-Carry: Sell spot, buy futures (when F < fair value)
    """
    
    def __init__(
        self,
        transaction_costs: Dict[str, float],
        min_profit_threshold: float = 10.0,
        min_basis_deviation_pct: float = 0.1
    ):
        """
        Initialize Futures Basis Monitor
        
        Args:
            transaction_costs: Dict with keys 'equity', 'futures'
            min_profit_threshold: Minimum profit (in rupees) to consider viable
            min_basis_deviation_pct: Minimum basis deviation % to trigger arbitrage
        """
        super().__init__(transaction_costs, min_profit_threshold)
        self.min_basis_deviation_pct = min_basis_deviation_pct
        
        log.info(f"FuturesBasis monitor initialized with {min_basis_deviation_pct}% deviation threshold")
    
    def calculate_fair_futures_price(
        self,
        spot_price: float,
        risk_free_rate: float,
        dividend_yield: float,
        time_to_expiry: float
    ) -> float:
        """
        Calculate theoretical fair futures price using cost-of-carry model
        
        Args:
            spot_price: Current spot price
            risk_free_rate: Annual risk-free rate (e.g., 0.07 for 7%)
            dividend_yield: Annual dividend yield (typically 0 for indices)
            time_to_expiry: Time to expiry in years
        
        Returns:
            Fair futures price
        """
        # F = S * e^((r - q) * T)
        carry_cost = (risk_free_rate - dividend_yield) * time_to_expiry
        fair_price = spot_price * np.exp(carry_cost)
        
        return fair_price
    
    def check_arbitrage(self, market_data: Dict) -> Optional[ArbitrageOpportunity]:
        """
        Check for futures basis arbitrage opportunity
        
        Args:
            market_data: Dictionary containing:
                - spot_price: Current spot price
                - futures_price: Current futures price
                - time_to_expiry: Time to expiry (years)
                - risk_free_rate: Risk-free rate
                - dividend_yield: Dividend yield (default 0)
                - market: Market identifier (e.g., 'NSE')
                - instrument: Instrument name (e.g., 'NIFTY')
                - expiry_date: Expiry date string
        
        Returns:
            ArbitrageOpportunity if found, None otherwise
        """
        # Extract data
        spot = market_data['spot_price']
        futures = market_data['futures_price']
        T = market_data['time_to_expiry']
        r = market_data['risk_free_rate']
        q = market_data.get('dividend_yield', 0.0)
        market = market_data.get('market', 'NSE')
        instrument = market_data.get('instrument', 'Unknown')
        
        # Calculate fair futures price
        fair_futures = self.calculate_fair_futures_price(spot, r, q, T)
        
        # Calculate basis deviation
        basis_deviation = futures - fair_futures
        basis_deviation_pct = (basis_deviation / fair_futures) * 100
        
        log.debug(f"Futures basis check: Spot={spot:.2f}, Futures={futures:.2f}, Fair={fair_futures:.2f}, Deviation={basis_deviation_pct:.3f}%")
        
        # Check if deviation exceeds threshold
        if abs(basis_deviation_pct) < self.min_basis_deviation_pct:
            return None
        
        # Determine strategy
        if futures > fair_futures:
            # Cash-and-Carry: Futures overpriced
            return self._create_cash_and_carry_opportunity(
                spot, futures, fair_futures, T, r, market_data
            )
        else:
            # Reverse Cash-and-Carry: Futures underpriced
            return self._create_reverse_cash_and_carry_opportunity(
                spot, futures, fair_futures, T, r, market_data
            )
    
    def _create_cash_and_carry_opportunity(
        self,
        spot: float,
        futures: float,
        fair_futures: float,
        T: float,
        r: float,
        market_data: Dict
    ) -> Optional[ArbitrageOpportunity]:
        """
        Create cash-and-carry arbitrage opportunity
        
        Strategy:
        1. Buy spot at S
        2. Sell futures at F
        3. Borrow S at rate r
        
        At expiry:
        - Deliver spot, receive F
        - Repay loan: S * e^(rT)
        - Profit = F - S * e^(rT)
        """
        instrument = market_data.get('instrument', 'Unknown')
        
        # Calculate costs
        spot_cost = spot
        borrowing_cost = spot * (np.exp(r * T) - 1)  # Interest on borrowed amount
        
        # Transaction costs
        equity_tc = self.calculate_transaction_cost(spot, 'equity')
        futures_tc = self.calculate_transaction_cost(futures, 'futures')
        total_tc = equity_tc + futures_tc
        
        # Gross profit
        gross_profit = futures - spot - borrowing_cost
        
        # Net profit
        net_profit = gross_profit - total_tc
        
        # Check viability
        if net_profit < self.min_profit_threshold:
            return None
        
        # Calculate profit percentage (relative to capital employed)
        capital_employed = spot + equity_tc
        profit_pct = (net_profit / capital_employed) * 100
        
        # Annualized return
        annualized_return = (net_profit / capital_employed) * (365.0 / (T * 365)) * 100
        
        # Execution steps
        execution_steps = [
            f"1. Borrow ₹{spot:.2f} at {r*100:.2f}% annual rate",
            f"2. Buy {instrument} spot at ₹{spot:.2f} (Cost: ₹{equity_tc:.2f})",
            f"3. Sell {instrument} futures at ₹{futures:.2f} (Cost: ₹{futures_tc:.2f})",
            f"4. At expiry ({market_data.get('expiry_date', 'expiry')}): Deliver spot, receive ₹{futures:.2f}",
            f"5. Repay loan: ₹{spot * np.exp(r * T):.2f} (principal + interest)",
            f"6. Net profit: ₹{net_profit:.2f}"
        ]
        
        # Positions
        positions = [
            {
                'action': 'BUY',
                'instrument_type': 'Spot',
                'price': spot,
                'quantity': 1,
                'cost': spot + equity_tc
            },
            {
                'action': 'SELL',
                'instrument_type': 'Futures',
                'price': futures,
                'quantity': 1,
                'proceeds': futures - futures_tc
            },
            {
                'action': 'BORROW',
                'instrument_type': 'Cash',
                'amount': spot,
                'rate': r,
                'repayment': spot * np.exp(r * T)
            }
        ]
        
        # Confidence score (based on deviation magnitude and time to expiry)
        deviation_factor = min(abs((futures - fair_futures) / fair_futures), 0.05) / 0.05
        time_factor = min(T, 0.25) / 0.25  # Prefer shorter maturities
        confidence = 0.5 + (0.3 * deviation_factor) + (0.2 * time_factor)
        
        # Metadata
        metadata = {
            'spot': spot,
            'futures': futures,
            'fair_futures': fair_futures,
            'basis_deviation': futures - fair_futures,
            'basis_deviation_pct': ((futures - fair_futures) / fair_futures) * 100,
            'time_to_expiry_days': T * 365,
            'borrowing_cost': borrowing_cost,
            'transaction_costs': total_tc,
            'annualized_return': annualized_return,
            'expiry_date': market_data.get('expiry_date', 'N/A')
        }
        
        opportunity = ArbitrageOpportunity(
            strategy_name="Cash-and-Carry (Futures Basis)",
            instrument=instrument,
            market=market_data.get('market', 'NSE'),
            gross_profit=gross_profit,
            net_profit=net_profit,
            profit_pct=profit_pct,
            execution_steps=execution_steps,
            positions=positions,
            confidence_score=confidence,
            metadata=metadata,
            detection_time=datetime.now()  # Add this line
        )
        
        self.log_opportunity(opportunity)
        return opportunity
    
    def _create_reverse_cash_and_carry_opportunity(
        self,
        spot: float,
        futures: float,
        fair_futures: float,
        T: float,
        r: float,
        market_data: Dict
    ) -> Optional[ArbitrageOpportunity]:
        """
        Create reverse cash-and-carry arbitrage opportunity
        
        Strategy:
        1. Sell spot at S
        2. Buy futures at F
        3. Lend S at rate r
        
        At expiry:
        - Receive S * e^(rT) from lending
        - Buy spot at F via futures
        - Profit = S * e^(rT) - F
        """
        instrument = market_data.get('instrument', 'Unknown')
        
        # Calculate proceeds
        spot_proceeds = spot
        lending_proceeds = spot * (np.exp(r * T) - 1)  # Interest earned
        
        # Transaction costs
        equity_tc = self.calculate_transaction_cost(spot, 'equity')
        futures_tc = self.calculate_transaction_cost(futures, 'futures')
        total_tc = equity_tc + futures_tc
        
        # Gross profit
        gross_profit = spot + lending_proceeds - futures
        
        # Net profit
        net_profit = gross_profit - total_tc
        
        # Check viability
        if net_profit < self.min_profit_threshold:
            return None
        
        # Calculate profit percentage
        capital_employed = futures + futures_tc  # Capital needed to buy back
        profit_pct = (net_profit / capital_employed) * 100
        
        # Annualized return
        annualized_return = (net_profit / capital_employed) * (365.0 / (T * 365)) * 100
        
        # Execution steps
        execution_steps = [
            f"1. Sell {instrument} spot at ₹{spot:.2f} (Cost: ₹{equity_tc:.2f})",
            f"2. Lend ₹{spot:.2f} at {r*100:.2f}% annual rate",
            f"3. Buy {instrument} futures at ₹{futures:.2f} (Cost: ₹{futures_tc:.2f})",
            f"4. At expiry ({market_data.get('expiry_date', 'expiry')}): Receive ₹{spot * np.exp(r * T):.2f} from lending",
            f"5. Receive spot via futures at ₹{futures:.2f}",
            f"6. Net profit: ₹{net_profit:.2f}"
        ]
        
        # Positions
        positions = [
            {
                'action': 'SELL',
                'instrument_type': 'Spot',
                'price': spot,
                'quantity': 1,
                'proceeds': spot - equity_tc
            },
            {
                'action': 'BUY',
                'instrument_type': 'Futures',
                'price': futures,
                'quantity': 1,
                'cost': futures + futures_tc
            },
            {
                'action': 'LEND',
                'instrument_type': 'Cash',
                'amount': spot,
                'rate': r,
                'maturity_value': spot * np.exp(r * T)
            }
        ]
        
        # Confidence score
        deviation_factor = min(abs((futures - fair_futures) / fair_futures), 0.05) / 0.05
        time_factor = min(T, 0.25) / 0.25
        confidence = 0.5 + (0.3 * deviation_factor) + (0.2 * time_factor)
        
        # Metadata
        metadata = {
            'spot': spot,
            'futures': futures,
            'fair_futures': fair_futures,
            'basis_deviation': futures - fair_futures,
            'basis_deviation_pct': ((futures - fair_futures) / fair_futures) * 100,
            'time_to_expiry_days': T * 365,
            'lending_proceeds': lending_proceeds,
            'transaction_costs': total_tc,
            'annualized_return': annualized_return,
            'expiry_date': market_data.get('expiry_date', 'N/A')
        }
        
        opportunity = ArbitrageOpportunity(
            strategy_name="Reverse Cash-and-Carry (Futures Basis)",
            instrument=instrument,
            market=market_data.get('market', 'NSE'),
            gross_profit=gross_profit,
            net_profit=net_profit,
            profit_pct=profit_pct,
            execution_steps=execution_steps,
            positions=positions,
            confidence_score=confidence,
            metadata=metadata,
            detection_time=datetime.now()
        )
        
        self.log_opportunity(opportunity)
        return opportunity