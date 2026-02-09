"""
Test Futures Basis Monitor
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.arbitrage_monitors.futures_basis import FuturesBasisMonitor
from src.data_acquisition.indian_markets import get_futures_arbitrage_data


def test_futures_basis():
    """Test futures basis arbitrage detection"""
    
    print("\n" + "="*70)
    print("FUTURES BASIS ARBITRAGE TEST")
    print("="*70)
    
    # Get futures data
    futures_data = get_futures_arbitrage_data('NIFTY')
    
    if futures_data is None:
        print("\nFailed to fetch futures data")
        return False
    
    print(f"\nMarket Data:")
    print(f"  Spot: ₹{futures_data['spot_price']:,.2f}")
    print(f"  Futures: ₹{futures_data['futures_price']:,.2f}")
    print(f"  Time to Expiry: {futures_data['time_to_expiry']*365:.0f} days")
    print(f"  Risk-Free Rate: {futures_data['risk_free_rate']*100:.2f}%")
    print(f"  Expiry: {futures_data['expiry_date']}")
    
    # Initialize monitor
    monitor = FuturesBasisMonitor(
        transaction_costs={'equity': 0.0005, 'futures': 0.0002},
        min_profit_threshold=5.0,
        min_basis_deviation_pct=0.05
    )
    
    # Check for arbitrage
    opportunity = monitor.check_arbitrage(futures_data)
    
    if opportunity and opportunity.is_viable:
        print(f"\n{'='*70}")
        print(f"ARBITRAGE OPPORTUNITY FOUND!")
        print(f"{'='*70}")
        print(f"Strategy: {opportunity.strategy_name}")
        print(f"Gross Profit: ₹{opportunity.gross_profit:.2f}")
        print(f"Net Profit: ₹{opportunity.net_profit:.2f}")
        print(f"Profit %: {opportunity.profit_pct:.3f}%")
        print(f"Annualized Return: {opportunity.metadata['annualized_return']:.2f}%")
        
        print(f"\nExecution Steps:")
        for step in opportunity.execution_steps:
            print(f"  {step}")
        
        return True
    else:
        print("\nNo arbitrage opportunity (normal in efficient markets)")
        return True


if __name__ == "__main__":
    success = test_futures_basis()
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70 + "\n")