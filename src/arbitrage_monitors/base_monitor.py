"""
Base class for all arbitrage monitors
Provides common functionality and interface
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
from src.utils.logger import log

class ArbitrageOpportunity:
    """
    Data class representing an arbitrage opportunity
    """
    
    def __init__(self,
                 strategy_name: str,
                 market: str,
                 instrument: str,
                 detection_time: datetime,
                 gross_profit: float,
                 net_profit: float,
                 profit_pct: float,
                 positions: List[Dict],
                 execution_steps: List[str],
                 confidence_score: float = 1.0,
                 metadata: Optional[Dict] = None):
        """
        Args:
            strategy_name: Type of arbitrage (e.g., "Put-Call Parity", "Box Spread")
            market: Market identifier (e.g., "NSE", "NYSE")
            instrument: Underlying instrument (e.g., "NIFTY", "SPY")
            detection_time: When opportunity was detected
            gross_profit: Profit before transaction costs
            net_profit: Profit after transaction costs
            profit_pct: Profit as percentage of investment
            positions: List of positions to execute
            execution_steps: Human-readable execution instructions
            confidence_score: Reliability score (0-1)
            metadata: Additional information
        """
        self.strategy_name = strategy_name
        self.market = market
        self.instrument = instrument
        self.detection_time = detection_time
        self.gross_profit = gross_profit
        self.net_profit = net_profit
        self.profit_pct = profit_pct
        self.positions = positions
        self.execution_steps = execution_steps
        self.confidence_score = confidence_score
        self.metadata = metadata or {}
        
        # Calculate additional metrics
        self.is_viable = self.net_profit > 0
        self.priority_score = self._calculate_priority()
    
    def _calculate_priority(self) -> float:
        """
        Calculate priority score for opportunity ranking
        Higher score = more attractive opportunity
        
        Factors:
        - Net profit percentage (primary)
        - Confidence score
        - Size of opportunity
        """
        priority = (
            self.profit_pct * 0.6 +
            self.confidence_score * 30 * 0.3 +
            min(self.net_profit / 1000, 10) * 0.1  # Capped contribution from absolute profit
        )
        return priority
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage/display"""
        return {
            'strategy_name': self.strategy_name,
            'market': self.market,
            'instrument': self.instrument,
            'detection_time': self.detection_time.isoformat(),
            'gross_profit': self.gross_profit,
            'net_profit': self.net_profit,
            'profit_pct': self.profit_pct,
            'positions': self.positions,
            'execution_steps': self.execution_steps,
            'confidence_score': self.confidence_score,
            'is_viable': self.is_viable,
            'priority_score': self.priority_score,
            'metadata': self.metadata
        }
    
    def __repr__(self):
        return (f"ArbitrageOpportunity(strategy={self.strategy_name}, "
                f"instrument={self.instrument}, profit=₹{self.net_profit:.2f}, "
                f"profit_pct={self.profit_pct:.2f}%)")


class BaseArbitrageMonitor(ABC):
    """
    Abstract base class for arbitrage monitors
    All specific monitors (PCP, CIP, etc.) inherit from this
    """
    
    def __init__(self, transaction_costs: Dict[str, float], 
                 min_profit_threshold: float = 0.0):
        """
        Args:
            transaction_costs: Dictionary of transaction costs by instrument type
                              e.g., {'equity': 0.0005, 'options': 0.001}
            min_profit_threshold: Minimum profit (after costs) to flag opportunity
        """
        self.transaction_costs = transaction_costs
        self.min_profit_threshold = min_profit_threshold
        self.opportunities_detected = []
        self.monitor_name = self.__class__.__name__
        
        log.info(f"Initialized {self.monitor_name} with min profit threshold: ₹{min_profit_threshold}")
    
    @abstractmethod
    def check_arbitrage(self, market_data: Dict) -> Optional[ArbitrageOpportunity]:
        """
        Main method to check for arbitrage opportunities
        Must be implemented by each specific monitor
        
        Args:
            market_data: Dictionary containing relevant market data
            
        Returns:
            ArbitrageOpportunity if found, None otherwise
        """
        pass
    
    def calculate_transaction_cost(self, price: float, 
                                   instrument_type: str,
                                   quantity: float = 1.0) -> float:
        """
        Calculate transaction cost for a trade
        
        Args:
            price: Price of instrument
            instrument_type: Type (e.g., 'equity', 'options', 'futures')
            quantity: Number of units
            
        Returns:
            Total transaction cost
        """
        cost_rate = self.transaction_costs.get(instrument_type, 0.001)  # Default 0.1%
        return price * quantity * cost_rate
    
    def log_opportunity(self, opportunity: ArbitrageOpportunity):
        """
        Log detected opportunity
        """
        self.opportunities_detected.append(opportunity)
        
        if opportunity.is_viable:
            log.info(f"✓ VIABLE ARBITRAGE: {opportunity}")
        else:
            log.debug(f"✗ Unprofitable arbitrage: {opportunity}")
    
    def get_opportunities(self, 
                         min_priority: float = 0.0,
                         limit: int = 50) -> List[ArbitrageOpportunity]:
        """
        Retrieve detected opportunities sorted by priority
        
        Args:
            min_priority: Minimum priority score
            limit: Maximum number of opportunities to return
            
        Returns:
            List of opportunities sorted by priority (highest first)
        """
        filtered = [
            opp for opp in self.opportunities_detected 
            if opp.priority_score >= min_priority and opp.is_viable
        ]
        
        sorted_opps = sorted(filtered, key=lambda x: x.priority_score, reverse=True)
        return sorted_opps[:limit]
    
    def clear_opportunities(self):
        """Clear stored opportunities (e.g., for new session)"""
        self.opportunities_detected = []
        log.info(f"Cleared opportunities for {self.monitor_name}")
    
    def get_statistics(self) -> Dict:
        """
        Get summary statistics of detected opportunities
        """
        viable = [opp for opp in self.opportunities_detected if opp.is_viable]
        
        if not viable:
            return {
                'total_opportunities': len(self.opportunities_detected),
                'viable_opportunities': 0,
                'avg_profit': 0,
                'max_profit': 0,
                'total_profit_potential': 0
            }
        
        return {
            'total_opportunities': len(self.opportunities_detected),
            'viable_opportunities': len(viable),
            'avg_profit': sum(opp.net_profit for opp in viable) / len(viable),
            'max_profit': max(opp.net_profit for opp in viable),
            'total_profit_potential': sum(opp.net_profit for opp in viable),
            'avg_priority_score': sum(opp.priority_score for opp in viable) / len(viable)
        }