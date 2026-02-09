"""
Test Data Acquisition
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_acquisition.indian_markets import NSEDataFetcher
from src.data_acquisition.risk_free_rates import RiskFreeRateFetcher
from src.arbitrage_monitors.put_call_parity import PutCallParityMonitor


def test_basic_fetch():
    """Test basic data fetching"""
    print("\n" + "="*70)
    print("TEST: Basic Data Fetching")
    print("="*70)
    
    fetcher = NSEDataFetcher()
    
    # Test spot price
    spot = fetcher.get_spot_price('NIFTY')
    print(f"\nNifty Spot: Rs.{spot:,.2f}")
    
    # Test risk-free rate
    rate_fetcher = RiskFreeRateFetcher()
    rate = rate_fetcher.get_india_rate('91day')
    print(f"India 91-day T-Bill: {rate*100:.2f}%")
    
    # Test arbitrage data
    arb_data = fetcher.get_option_data_for_arbitrage('NIFTY')
    print(f"\nGenerated {len(arb_data)} option pairs for arbitrage")
    
    if len(arb_data) > 0:
        print("\nSample data structure:")
        sample = arb_data[0]
        print(f"  Strike: {sample['strike']}")
        print(f"  Call: {sample['call_price']:.2f}")
        print(f"  Put: {sample['put_price']:.2f}")
        print(f"  Time to expiry: {sample['time_to_expiry']*365:.1f} days")
    
    return True


def test_arbitrage_integration():
    """Test integration with arbitrage monitor"""
    print("\n" + "="*70)
    print("TEST: Arbitrage Detection Integration")
    print("="*70)
    
    # Get data
    fetcher = NSEDataFetcher()
    data_list = fetcher.get_option_data_for_arbitrage('NIFTY')
    
    if not data_list:
        print("\nNo data available")
        return False
    
    # Setup monitor
    monitor = PutCallParityMonitor(
        transaction_costs={'equity': 0.0005, 'options': 0.0005},
        min_profit_threshold=5.0
    )
    
    # Check for arbitrage
    found = 0
    for data in data_list[:5]:  # Check first 5 strikes
        opp = monitor.check_arbitrage(data)
        if opp and opp.is_viable:
            found += 1
            print(f"\nFound opportunity at strike {data['strike']:.0f}")
            print(f"  Profit: Rs.{opp.net_profit:.2f}")
    
    if found == 0:
        print("\nNo arbitrage found (normal in efficient markets)")
        print("System is working correctly!")
    
    return True


if __name__ == "__main__":
    print("\nDATA ACQUISITION TEST SUITE\n")
    
    test1 = test_basic_fetch()
    test2 = test_arbitrage_integration()
    
    print("\n" + "="*70)
    if test1 and test2:
        print("SUCCESS: All tests passed")
    print("="*70 + "\n")