"""
Test Put-Call Parity Monitor with real example
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.arbitrage_monitors.put_call_parity import PutCallParityMonitor
from datetime import datetime

def test_pcp_reversal():
    """Test with reversal arbitrage example (your manual calculation)"""
    
    print("\n" + "="*70)
    print("TEST 1: REVERSAL ARBITRAGE (Put Overpriced)")
    print("="*70)
    
    # Your example data
    market_data = {
        'spot_price': 22000,
        'strike': 22000,
        'call_price': 250,
        'put_price': 230,
        'time_to_expiry': 7/365,
        'risk_free_rate': 0.07,
        'dividend_yield': 0,
        'market': 'NSE',
        'instrument': 'NIFTY',
        'expiry_date': '2025-02-14'
    }

    # Transaction costs (typical for Indian markets)
    transaction_costs = {
        'equity': 0.0005,   # 0.05%
        'options': 0.0005,  # 0.05%
        'futures': 0.0002   # 0.02%
    }

    # Initialize monitor
    monitor = PutCallParityMonitor(
        transaction_costs=transaction_costs,
        min_profit_threshold=5.0,  # Minimum ₹5 profit
        min_deviation_pct=0.05      # Minimum 0.05% deviation
    )

    # Check for arbitrage
    opportunity = monitor.check_arbitrage(market_data)

    if opportunity:
        print(f"\n ARBITRAGE DETECTED!")
        print(f"   Strategy: {opportunity.strategy_name}")
        print(f"   Gross Profit: ₹{opportunity.gross_profit:.2f}")
        print(f"   Net Profit: ₹{opportunity.net_profit:.2f}")
        print(f"   Profit %: {opportunity.profit_pct:.2f}%")
        
        print(f"\n   Execution Steps:")
        for i, step in enumerate(opportunity.execution_steps, 1):
            print(f"      {i}. {step}")
        
        # Verify it's reversal
        if "Reversal" in opportunity.strategy_name:
            print(f"\n    Correct strategy: Reversal")
            return True
        else:
            print(f"\n    Wrong strategy: Expected Reversal, got {opportunity.strategy_name}")
            return False
    else:
        print("\n No arbitrage detected (FAILED)")
        return False


def test_pcp_conversion():
    """Test conversion arbitrage (call overpriced scenario)"""
    
    print("\n" + "="*70)
    print("TEST 2: CONVERSION ARBITRAGE (Call Overpriced)")
    print("="*70)
    
    market_data = {
        'spot_price': 22000,
        'strike': 22000,
        'call_price': 280,  # Overpriced call
        'put_price': 230,
        'time_to_expiry': 7/365,
        'risk_free_rate': 0.07,
        'dividend_yield': 0,
        'market': 'NSE',
        'instrument': 'NIFTY',
        'expiry_date': '2025-02-14'
    }
    
    transaction_costs = {
        'equity': 0.0005,
        'options': 0.0005,
        'futures': 0.0002
    }
    
    monitor = PutCallParityMonitor(
        transaction_costs=transaction_costs,
        min_profit_threshold=5.0,
        min_deviation_pct=0.05
    )
    
    opportunity = monitor.check_arbitrage(market_data)
    
    if opportunity:
        print(f"\n ARBITRAGE DETECTED!")
        print(f"   Strategy: {opportunity.strategy_name}")
        print(f"   Net Profit: ₹{opportunity.net_profit:.2f}")
        
        if "Conversion" in opportunity.strategy_name:
            print(f"    Correct strategy: Conversion")
            return True
        else:
            print(f"    Wrong strategy: Expected Conversion")
            return False
    else:
        print("\n No arbitrage detected (FAILED)")
        return False


def test_no_arbitrage():
    """Test when no arbitrage exists (prices in equilibrium)"""
    
    print("\n" + "="*70)
    print("TEST 3: NO ARBITRAGE (Perfect Parity)")
    print("="*70)
    
    # Perfectly balanced prices
    market_data = {
        'spot_price': 22000,
        'strike': 22000,
        'call_price': 220,
        'put_price': 200,  # Exactly matching put-call parity
        'time_to_expiry': 7/365,
        'risk_free_rate': 0.07,
        'dividend_yield': 0,
        'market': 'NSE',
        'instrument': 'NIFTY',
        'expiry_date': '2025-02-14'
    }
    
    transaction_costs = {
        'equity': 0.0005,
        'options': 0.0005,
        'futures': 0.0002
    }
    
    monitor = PutCallParityMonitor(
        transaction_costs=transaction_costs,
        min_profit_threshold=5.0,
        min_deviation_pct=0.05
    )
    
    opportunity = monitor.check_arbitrage(market_data)
    
    if opportunity is None:
        print(f"\n Correctly identified no arbitrage")
        return True
    else:
        print(f"\n False positive: Detected arbitrage when there shouldn't be")
        print(f"   Net Profit: ₹{opportunity.net_profit:.2f}")
        return False


if __name__ == "__main__":
    print("\n" + "."*35)
    print("PUT-CALL PARITY ARBITRAGE MONITOR TEST SUITE")
    print("."*35 + "\n")
    
    # Run all tests
    test1_passed = test_pcp_reversal()
    test2_passed = test_pcp_conversion()
    test3_passed = test_no_arbitrage()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"  Test 1 (Reversal):   {' PASSED' if test1_passed else ' FAILED'}")
    print(f"  Test 2 (Conversion): {' PASSED' if test2_passed else ' FAILED'}")
    print(f"  Test 3 (No Arb):     {' PASSED' if test3_passed else ' FAILED'}")
    
    total_passed = sum([test1_passed, test2_passed, test3_passed])
    print(f"\n  Overall: {total_passed}/3 tests passed")
    print("="*70 + "\n")
    
    if total_passed == 3:
        print(" ALL TESTS PASSED! Your PCP monitor is working correctly!\n")
    else:
        print(" Some tests failed. Review the output above.\n")
