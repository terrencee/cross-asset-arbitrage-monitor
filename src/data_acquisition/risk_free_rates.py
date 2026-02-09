"""
Risk-Free Interest Rates Data Acquisition
"""

from typing import Optional, Dict
from datetime import datetime
import time

try:
    import pandas_datareader.data as web
except ImportError:
    web = None

from src.utils.logger import log


class RiskFreeRateFetcher:
    """Fetch risk-free rates for option pricing"""
    
    # Fallback rates (updated Feb 2025)
    FALLBACK_RATES = {
        'india_91day': 0.0685,    # 6.85%
        'india_1year': 0.0700,    # 7.00%
        'us_3month': 0.0425,      # 4.25%
        'us_1year': 0.0410        # 4.10%
    }
    
    def __init__(self, cache_duration_seconds: int = 3600):
        """
        Initialize with 1-hour cache (rates don't change that fast)
        """
        self.cache_duration = cache_duration_seconds
        self.cache = {}
        self.last_fetch_time = {}
        log.info("RiskFreeRateFetcher initialized")
    
    def get_india_rate(self, tenor: str = '91day') -> float:
        """
        Get Indian T-Bill rate
        
        Args:
            tenor: '91day', '182day', '364day', or '1year'
            
        Returns:
            Annualized rate (e.g., 0.07 for 7%)
        """
        cache_key = f"india_{tenor}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        # For now, use fallback (RBI doesn't have easy API)
        rate = self.FALLBACK_RATES.get(cache_key, 0.07)
        log.info(f"India {tenor} rate: {rate*100:.2f}%")
        
        self._update_cache(cache_key, rate)
        return rate
    
    def get_us_rate(self, tenor: str = '3month') -> float:
        """
        Get US Treasury rate
        
        Could fetch from FRED API but fallback works fine
        """
        cache_key = f"us_{tenor}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        rate = self.FALLBACK_RATES.get(cache_key, 0.04)
        log.info(f"US {tenor} rate: {rate*100:.2f}%")
        
        self._update_cache(cache_key, rate)
        return rate
    
    def get_rate_for_options(self, time_to_expiry_years: float, market: str = 'NSE') -> float:
        """
        Get appropriate rate based on option maturity
        
        This is the KEY method - matches rate tenor to option tenor
        
        Args:
            time_to_expiry_years: Time to expiry (e.g., 0.0192 for 7 days)
            market: 'NSE' or 'NYSE'
        
        Returns:
            Appropriate annualized rate
        """
        # Choose tenor based on time to expiry
        if time_to_expiry_years < 0.5:  # < 6 months
            tenor = 'short'
        elif time_to_expiry_years < 2.0:  # < 2 years
            tenor = 'medium'
        else:
            tenor = 'long'
        
        # Get rate for market
        if market in ['NSE', 'BSE']:
            if tenor == 'short':
                return self.get_india_rate('91day')
            else:
                return self.get_india_rate('1year')
        else:  # US markets
            if tenor == 'short':
                return self.get_us_rate('3month')
            else:
                return self.get_us_rate('1year')
    
    def _is_cache_valid(self, key: str) -> bool:
        if key not in self.cache or key not in self.last_fetch_time:
            return False
        elapsed = time.time() - self.last_fetch_time[key]
        return elapsed < self.cache_duration
    
    def _update_cache(self, key: str, value: float):
        self.cache[key] = value
        self.last_fetch_time[key] = time.time()


# Convenience function
def get_rate(market: str = 'NSE', time_to_expiry: float = 0.0192) -> float:
    """
    Quick function to get rate
    
    Usage:
        rate = get_rate('NSE', 7/365)  # For 7-day Nifty option
    """
    fetcher = RiskFreeRateFetcher()
    return fetcher.get_rate_for_options(time_to_expiry, market)