"""
Test Covered Interest Rate Parity Monitor
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.arbitrage_monitors.covered_interest_parity import CoveredInterestParityMonitor
from src.data_acquisition.indian_markets import get_cip_arbitrage_data


def test_cip():
    """Test CIP arbitrage detection"""
    
    print("\n" + "="*70)
    print("COVERED INTEREST RATE PARITY TEST")
    print("="*70)
    
    # Get FX data
    cip_data = get_cip_arbitrage_data('USDINR', time_period_months=12)
    
    if cip_data is None:
        print("\nFailed to fetch FX data")
        return False
    
    print(f"\nMarket Data:")
    print(f"  Currency Pair: {cip_data['currency_pair']}")
    print(f"  Spot Rate: {cip_data['spot_rate']:.4f}")
    print(f"  Forward Rate: {cip_data['forward_rate']:.4f}")
    print(f"  Domestic Rate (INR): {cip_data['domestic_rate']*100:.2f}%")
    print(f"  Foreign Rate (USD): {cip_data['foreign_rate']*100:.2f}%")
    print(f"  Time Period: {cip_data['time_period']*12:.0f} months")
    print(f"  Notional: ${cip_data['notional']:,.2f}")
    
    # Initialize monitor
    monitor = CoveredInterestParityMonitor(
        transaction_costs={
            'fx_spot': 0.0005,
            'fx_forward': 0.0005,
            'borrowing_spread': 0.001
        },
        min_profit_threshold=100.0,
        min_deviation_pct=0.1
    )
    
    # Check for arbitrage
    opportunity = monitor.check_arbitrage(cip_data)
    
    if opportunity and opportunity.is_viable:
        print(f"\n{'='*70}")
        print(f"CIP ARBITRAGE OPPORTUNITY FOUND!")
        print(f"{'='*70}")
        print(f"Strategy: {opportunity.strategy_name}")
        print(f"Gross Profit: ₹{opportunity.gross_profit:.2f}")
        print(f"Net Profit: ₹{opportunity.net_profit:.2f}")
        print(f"Profit %: {opportunity.profit_pct:.3f}%")
        print(f"Annualized Return: {opportunity.metadata['annualized_return']:.2f}%")
        
        print(f"\nFair Forward: {opportunity.metadata['fair_forward']:.4f}")
        print(f"Actual Forward: {opportunity.metadata['forward_rate']:.4f}")
        print(f"Deviation: {opportunity.metadata['deviation_pct']:.3f}%")
        
        print(f"\nExecution Steps:")
        for step in opportunity.execution_steps:
            print(f"  {step}")
        
        return True
    else:
        print("\nNo CIP arbitrage opportunity (normal in efficient markets)")
        return True


if __name__ == "__main__":
    success = test_cip()
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70 + "\n")