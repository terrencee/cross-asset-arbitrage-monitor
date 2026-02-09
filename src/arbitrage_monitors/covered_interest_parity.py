"""
Covered Interest Rate Parity (CIP) Arbitrage Monitor

Detects mispricing between FX spot, FX forward, and interest rate differentials.

True cross-asset arbitrage: Foreign Exchange + Fixed Income

Author: Adhiraj - IIT Roorkee
"""

import numpy as np
from typing import Dict, Optional
from datetime import datetime

from src.arbitrage_monitors.base_monitor import BaseArbitrageMonitor, ArbitrageOpportunity
from src.utils.logger import log


class CoveredInterestParityMonitor(BaseArbitrageMonitor):
    """
    Monitor for Covered Interest Rate Parity arbitrage
    
    Fundamental relationship:
        F/S = (1 + r_domestic) / (1 + r_foreign)
    
    Or equivalently:
        F = S × [(1 + r_domestic) / (1 + r_foreign)]
    
    Where:
        F = Forward exchange rate
        S = Spot exchange rate
        r_domestic = Domestic interest rate
        r_foreign = Foreign interest rate
    
    Arbitrage strategies:
        1. Borrow-Invest: When forward is overvalued
        2. Invest-Borrow: When forward is undervalued
    """
    
    def __init__(
        self,
        transaction_costs: Dict[str, float],
        min_profit_threshold: float = 100.0,
        min_deviation_pct: float = 0.1
    ):
        """
        Initialize CIP Monitor
        
        Args:
            transaction_costs: Dict with 'fx_spot', 'fx_forward', 'borrowing_spread'
            min_profit_threshold: Minimum profit to consider viable
            min_deviation_pct: Minimum deviation % to trigger arbitrage
        """
        super().__init__(transaction_costs, min_profit_threshold)
        self.min_deviation_pct = min_deviation_pct
        
        log.info(f"CIP monitor initialized with {min_deviation_pct}% deviation threshold")
    
    def calculate_fair_forward(
        self,
        spot_rate: float,
        domestic_rate: float,
        foreign_rate: float,
        time_period: float
    ) -> float:
        """
        Calculate theoretical fair forward rate using CIP
        
        Args:
            spot_rate: Current spot exchange rate (domestic/foreign, e.g., INR/USD)
            domestic_rate: Domestic interest rate (e.g., India)
            foreign_rate: Foreign interest rate (e.g., USA)
            time_period: Time period in years
        
        Returns:
            Fair forward exchange rate
        """
        # F = S × [(1 + r_d × T) / (1 + r_f × T)]
        # Using simple interest for clarity
        fair_forward = spot_rate * ((1 + domestic_rate * time_period) / 
                                    (1 + foreign_rate * time_period))
        
        return fair_forward
    
    def check_arbitrage(self, market_data: Dict) -> Optional[ArbitrageOpportunity]:
        """
        Check for CIP arbitrage opportunity
        
        Args:
            market_data: Dictionary containing:
                - spot_rate: Current spot FX rate
                - forward_rate: Current forward FX rate
                - domestic_rate: Domestic interest rate
                - foreign_rate: Foreign interest rate
                - time_period: Time to maturity (years)
                - currency_pair: E.g., 'USD/INR'
                - notional: Amount to trade (e.g., $10,000)
        
        Returns:
            ArbitrageOpportunity if found, None otherwise
        """
        # Extract data
        spot = market_data['spot_rate']
        forward = market_data['forward_rate']
        r_domestic = market_data['domestic_rate']
        r_foreign = market_data['foreign_rate']
        T = market_data['time_period']
        currency_pair = market_data.get('currency_pair', 'FX')
        notional = market_data.get('notional', 10000)  # Default $10,000
        
        # Calculate fair forward
        fair_forward = self.calculate_fair_forward(spot, r_domestic, r_foreign, T)
        
        # Calculate deviation
        deviation = forward - fair_forward
        deviation_pct = (deviation / fair_forward) * 100
        
        log.debug(f"CIP check: Spot={spot:.4f}, Forward={forward:.4f}, Fair={fair_forward:.4f}, Deviation={deviation_pct:.3f}%")
        
        # Check if deviation exceeds threshold
        if abs(deviation_pct) < self.min_deviation_pct:
            return None
        
        # Determine strategy
        if forward > fair_forward:
            # Forward is overvalued - sell forward
            return self._create_borrow_invest_opportunity(
                spot, forward, fair_forward, r_domestic, r_foreign, T, 
                notional, deviation, deviation_pct, market_data
            )
        else:
            # Forward is undervalued - buy forward
            return self._create_invest_borrow_opportunity(
                spot, forward, fair_forward, r_domestic, r_foreign, T,
                notional, deviation, deviation_pct, market_data
            )
    
    def _create_borrow_invest_opportunity(
        self,
        spot: float,
        forward: float,
        fair_forward: float,
        r_d: float,
        r_f: float,
        T: float,
        notional: float,
        deviation: float,
        deviation_pct: float,
        market_data: Dict
    ) -> Optional[ArbitrageOpportunity]:
        """
        Create borrow-invest arbitrage (forward overvalued)
        
        Strategy:
        1. Borrow domestic currency (INR) at r_d
        2. Convert to foreign currency (USD) at spot rate
        3. Invest foreign currency at r_f
        4. Sell foreign currency forward at F
        5. At maturity: receive foreign investment, deliver via forward, repay domestic loan
        
        Profit = Proceeds from forward - Domestic loan repayment
        """
        currency_pair = market_data.get('currency_pair', 'FX')
        
        # Amount to borrow in domestic currency
        domestic_borrow = notional * spot
        
        # Convert to foreign currency at spot
        foreign_amount = notional
        
        # Invest foreign currency
        foreign_maturity = foreign_amount * (1 + r_f * T)
        
        # Convert back at forward rate (locked in)
        domestic_proceeds = foreign_maturity * forward
        
        # Repay domestic loan
        domestic_repay = domestic_borrow * (1 + r_d * T)
        
        # Calculate transaction costs
        spot_tc = self.calculate_transaction_cost(domestic_borrow, 'fx_spot')
        forward_tc = self.calculate_transaction_cost(domestic_proceeds, 'fx_forward')
        # Borrowing spread (additional cost over base rate)
        borrow_spread = domestic_borrow * self.transaction_costs.get('borrowing_spread', 0.001) * T
        
        total_tc = spot_tc + forward_tc + borrow_spread
        
        # Gross profit
        gross_profit = domestic_proceeds - domestic_repay
        
        # Net profit
        net_profit = gross_profit - total_tc
        
        # Check viability
        if net_profit < self.min_profit_threshold:
            return None
        
        # Profit percentage
        profit_pct = (net_profit / domestic_borrow) * 100
        
        # Annualized return
        annualized_return = (net_profit / domestic_borrow) * (1 / T) * 100
        
        # Execution steps
        execution_steps = [
            f"1. Borrow ₹{domestic_borrow:,.2f} at {r_d*100:.2f}% p.a. for {T*12:.1f} months",
            f"2. Convert to ${notional:,.2f} at spot rate {spot:.4f} (Cost: ₹{spot_tc:.2f})",
            f"3. Invest ${notional:,.2f} in USD market at {r_f*100:.2f}% p.a.",
            f"4. Sell ${foreign_maturity:,.2f} forward at rate {forward:.4f} (Cost: ₹{forward_tc:.2f})",
            f"5. At maturity: Receive ${foreign_maturity:,.2f} from USD investment",
            f"6. Deliver USD via forward contract, receive ₹{domestic_proceeds:,.2f}",
            f"7. Repay INR loan: ₹{domestic_repay:,.2f}",
            f"8. Net profit: ₹{net_profit:,.2f}"
        ]
        
        # Positions
        positions = [
            {
                'action': 'BORROW',
                'instrument_type': 'INR Loan',
                'amount': domestic_borrow,
                'rate': r_d,
                'maturity': domestic_repay
            },
            {
                'action': 'CONVERT',
                'instrument_type': 'FX Spot',
                'from_amount': domestic_borrow,
                'to_amount': notional,
                'rate': spot
            },
            {
                'action': 'INVEST',
                'instrument_type': 'USD Deposit',
                'amount': notional,
                'rate': r_f,
                'maturity': foreign_maturity
            },
            {
                'action': 'SELL',
                'instrument_type': 'FX Forward',
                'amount': foreign_maturity,
                'rate': forward,
                'proceeds': domestic_proceeds
            }
        ]
        
        # Confidence score
        deviation_factor = min(abs((forward - fair_forward) / fair_forward), 0.05) / 0.05
        time_factor = min(T, 1.0) / 1.0
        confidence = 0.5 + (0.3 * deviation_factor) + (0.2 * time_factor)
        
        # Metadata
        metadata = {
            'spot_rate': spot,
            'forward_rate': forward,
            'fair_forward': fair_forward,
            'deviation': deviation,
            'deviation_pct': deviation_pct,
            'domestic_rate': r_d,
            'foreign_rate': r_f,
            'time_period_years': T,
            'time_period_days': T * 365,
            'notional_foreign': notional,
            'notional_domestic': domestic_borrow,
            'transaction_costs': total_tc,
            'annualized_return': annualized_return,
            'currency_pair': currency_pair
        }
        
        opportunity = ArbitrageOpportunity(
            strategy_name=f"Borrow-Invest CIP ({currency_pair})",
            instrument=currency_pair,
            market='FX',
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
    
    def _create_invest_borrow_opportunity(
        self,
        spot: float,
        forward: float,
        fair_forward: float,
        r_d: float,
        r_f: float,
        T: float,
        notional: float,
        deviation: float,
        deviation_pct: float,
        market_data: Dict
    ) -> Optional[ArbitrageOpportunity]:
        """
        Create invest-borrow arbitrage (forward undervalued)
        
        Strategy:
        1. Borrow foreign currency (USD) at r_f
        2. Convert to domestic currency (INR) at spot rate
        3. Invest domestic currency at r_d
        4. Buy foreign currency forward at F
        5. At maturity: receive domestic investment, convert via forward, repay foreign loan
        """
        currency_pair = market_data.get('currency_pair', 'FX')
        
        # Borrow foreign currency
        foreign_borrow = notional
        
        # Convert to domestic at spot
        domestic_amount = notional * spot
        
        # Invest domestic currency
        domestic_maturity = domestic_amount * (1 + r_d * T)
        
        # Buy foreign forward (to repay loan)
        foreign_repay = foreign_borrow * (1 + r_f * T)
        domestic_needed = foreign_repay * forward
        
        # Calculate transaction costs
        spot_tc = self.calculate_transaction_cost(domestic_amount, 'fx_spot')
        forward_tc = self.calculate_transaction_cost(domestic_needed, 'fx_forward')
        borrow_spread = foreign_borrow * spot * self.transaction_costs.get('borrowing_spread', 0.001) * T
        
        total_tc = spot_tc + forward_tc + borrow_spread
        
        # Gross profit
        gross_profit = domestic_maturity - domestic_needed
        
        # Net profit
        net_profit = gross_profit - total_tc
        
        # Check viability
        if net_profit < self.min_profit_threshold:
            return None
        
        # Profit percentage
        profit_pct = (net_profit / domestic_amount) * 100
        
        # Annualized return
        annualized_return = (net_profit / domestic_amount) * (1 / T) * 100
        
        # Execution steps
        execution_steps = [
            f"1. Borrow ${foreign_borrow:,.2f} at {r_f*100:.2f}% p.a. for {T*12:.1f} months",
            f"2. Convert to ₹{domestic_amount:,.2f} at spot rate {spot:.4f} (Cost: ₹{spot_tc:.2f})",
            f"3. Invest ₹{domestic_amount:,.2f} in INR market at {r_d*100:.2f}% p.a.",
            f"4. Buy ${foreign_repay:,.2f} forward at rate {forward:.4f} (Cost: ₹{forward_tc:.2f})",
            f"5. At maturity: Receive ₹{domestic_maturity:,.2f} from INR investment",
            f"6. Pay ₹{domestic_needed:,.2f} to receive ${foreign_repay:,.2f} via forward",
            f"7. Repay USD loan: ${foreign_repay:,.2f}",
            f"8. Net profit: ₹{net_profit:,.2f}"
        ]
        
        # Positions
        positions = [
            {
                'action': 'BORROW',
                'instrument_type': 'USD Loan',
                'amount': foreign_borrow,
                'rate': r_f,
                'maturity': foreign_repay
            },
            {
                'action': 'CONVERT',
                'instrument_type': 'FX Spot',
                'from_amount': foreign_borrow,
                'to_amount': domestic_amount,
                'rate': spot
            },
            {
                'action': 'INVEST',
                'instrument_type': 'INR Deposit',
                'amount': domestic_amount,
                'rate': r_d,
                'maturity': domestic_maturity
            },
            {
                'action': 'BUY',
                'instrument_type': 'FX Forward',
                'amount': foreign_repay,
                'rate': forward,
                'cost': domestic_needed
            }
        ]
        
        # Confidence score
        deviation_factor = min(abs((forward - fair_forward) / fair_forward), 0.05) / 0.05
        time_factor = min(T, 1.0) / 1.0
        confidence = 0.5 + (0.3 * deviation_factor) + (0.2 * time_factor)
        
        # Metadata
        metadata = {
            'spot_rate': spot,
            'forward_rate': forward,
            'fair_forward': fair_forward,
            'deviation': deviation,
            'deviation_pct': deviation_pct,
            'domestic_rate': r_d,
            'foreign_rate': r_f,
            'time_period_years': T,
            'time_period_days': T * 365,
            'notional_foreign': notional,
            'notional_domestic': domestic_amount,
            'transaction_costs': total_tc,
            'annualized_return': annualized_return,
            'currency_pair': currency_pair
        }
        
        opportunity = ArbitrageOpportunity(
            strategy_name=f"Invest-Borrow CIP ({currency_pair})",
            instrument=currency_pair,
            market='FX',
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