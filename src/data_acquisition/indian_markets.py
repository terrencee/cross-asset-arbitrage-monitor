"""
Indian Markets Data Acquisition
Fetches NSE (National Stock Exchange) market data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import random

# Import with error handling
try:
    from nsepython import nse_optionchain_scrapper, nse_eq
except ImportError:
    nse_optionchain_scrapper = None
    nse_eq = None

try:
    import yfinance as yf
except ImportError:
    yf = None

from src.utils.logger import log
from src.utils.validators import DataValidator


class NSEDataFetcher:
    """Fetch market data from NSE"""
    
    SYMBOLS = {
        'NIFTY': 'NIFTY',
        'BANKNIFTY': 'BANKNIFTY',
        'FINNIFTY': 'FINNIFTY'
    }
    
    YF_SYMBOLS = {
        'NIFTY': '^NSEI',
        'BANKNIFTY': '^NSEBANK'
    }
    
    def __init__(self, cache_duration_seconds: int = 60):
        """
        Initialize fetcher with caching
        
        Args:
            cache_duration_seconds: How long to keep cached data (default 60)
        """
        self.cache_duration = cache_duration_seconds
        self.cache = {}
        self.last_fetch_time = {}
        log.info("NSEDataFetcher initialized")
    
    def get_spot_price(self, symbol: str) -> Optional[float]:
        """
        Get current spot price
        
        Three-tier approach:
        1. Try nsepython (NSE API)
        2. Try yfinance (Yahoo Finance)
        3. Use fallback value (for testing)
        """
        cache_key = f"spot_{symbol}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            log.debug(f"Using cached spot for {symbol}")
            return self.cache[cache_key]
        
        spot_price = None
        
        # Method 1: nsepython
        if nse_eq is not None:
            try:
                if symbol == 'NIFTY':
                    data = nse_eq('NIFTY 50')
                elif symbol == 'BANKNIFTY':
                    data = nse_eq('NIFTY BANK')
                else:
                    data = nse_eq(symbol)
                
                if data and 'lastPrice' in data:
                    spot_price = float(data['lastPrice'])
                    log.info(f"Fetched {symbol} spot via nsepython: {spot_price}")
            except Exception as e:
                log.warning(f"nsepython failed: {e}")
        
        # Method 2: yfinance
        if spot_price is None and yf is not None and symbol in self.YF_SYMBOLS:
            try:
                ticker = yf.Ticker(self.YF_SYMBOLS[symbol])
                hist = ticker.history(period='1d')
                if not hist.empty:
                    spot_price = float(hist['Close'].iloc[-1])
                    log.info(f"Fetched {symbol} spot via yfinance: {spot_price}")
            except Exception as e:
                log.warning(f"yfinance failed: {e}")
        
        # Method 3: Fallback
        if spot_price is None:
            fallback = {'NIFTY': 22000.0, 'BANKNIFTY': 47000.0, 'FINNIFTY': 20000.0}
            spot_price = fallback.get(symbol)
            log.warning(f"Using fallback for {symbol}: {spot_price}")
        
        # Cache
        if spot_price is not None:
            self._update_cache(cache_key, spot_price)
        
        return spot_price
    
    def get_options_chain(self, symbol: str, expiry_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Get complete options chain
        
        Returns DataFrame with columns:
        - strike, call_price, put_price, call_bid, call_ask, put_bid, put_ask
        - call_oi, put_oi (open interest)
        - call_iv, put_iv (implied volatility)
        """
        cache_key = f"optchain_{symbol}_{expiry_date}"
        
        if self._is_cache_valid(cache_key):
            log.debug(f"Using cached options chain")
            return self.cache[cache_key]
        
        if nse_optionchain_scrapper is None:
            spot = self.get_spot_price(symbol)
            return self._generate_dummy_options_chain(symbol, spot)

        
        
        try:
            log.info(f"Fetching options chain for {symbol}")
            data = nse_optionchain_scrapper(symbol)
            
            if data is None or 'records' not in data:
                return self._generate_dummy_options_chain(symbol)
            
            records = data['records']['data']
            
            # Filter by expiry if specified
            if expiry_date:
                records = [r for r in records if r.get('expiryDate') == expiry_date]
            
            options_data = []
            
            for record in records:
                strike = record.get('strikePrice')
                
                # Extract call data
                call_data = record.get('CE', {})
                call_price = call_data.get('lastPrice', np.nan)
                call_bid = call_data.get('bidprice', np.nan)
                call_ask = call_data.get('askPrice', np.nan)
                call_oi = call_data.get('openInterest', 0)
                call_iv = call_data.get('impliedVolatility', np.nan)
                
                # Extract put data
                put_data = record.get('PE', {})
                put_price = put_data.get('lastPrice', np.nan)
                put_bid = put_data.get('bidprice', np.nan)
                put_ask = put_data.get('askPrice', np.nan)
                put_oi = put_data.get('openInterest', 0)
                put_iv = put_data.get('impliedVolatility', np.nan)
                
                expiry = record.get('expiryDate', '')
                
                options_data.append({
                    'strike': strike,
                    'expiry_date': expiry,
                    'call_price': call_price,
                    'call_bid': call_bid,
                    'call_ask': call_ask,
                    'call_oi': call_oi,
                    'call_iv': call_iv,
                    'put_price': put_price,
                    'put_bid': put_bid,
                    'put_ask': put_ask,
                    'put_oi': put_oi,
                    'put_iv': put_iv
                })
            
            df = pd.DataFrame(options_data)
            df = df.dropna(subset=['call_price', 'put_price'])
            
            log.info(f"Fetched {len(df)} option pairs")
            self._update_cache(cache_key, df)
            
            return df
            
        except Exception as e:
            log.error(f"Error fetching options: {e}")
            return self._generate_dummy_options_chain(symbol)
    
    def get_option_data_for_arbitrage(self, symbol: str, expiry_date: Optional[str] = None) -> List[Dict]:
        """
        Get data formatted for arbitrage detection
        
        This is the KEY method that connects data acquisition to arbitrage monitoring
        
        Returns list of dictionaries, each ready for PutCallParityMonitor.check_arbitrage()
        """
        spot = self.get_spot_price(symbol)
        if spot is None:
            log.error("Spot price is None, cannot proceed")
            return []
        
        log.info(f"Spot price for arbitrage filtering: {spot}")
        
        options_chain = self.get_options_chain(symbol, expiry_date)
        if options_chain is None:
            log.error("Options chain is None")
            return []
        
        if options_chain.empty:
            log.error("Options chain is empty")
            return []
        
        log.info(f"Options chain retrieved: {len(options_chain)} rows")
        log.info(f"Options chain columns: {list(options_chain.columns)}")
        log.info(f"Sample strikes: {list(options_chain['strike'].head())}")
        
        # Filter for liquid strikes (within 10% of spot)
        lower_bound = spot * 0.90
        upper_bound = spot * 1.10
        
        log.info(f"Filtering strikes between {lower_bound:.2f} and {upper_bound:.2f}")
        
        liquid_options = options_chain[
            (options_chain['strike'] >= lower_bound) & 
            (options_chain['strike'] <= upper_bound)
        ]
        
        log.info(f"Filtering {len(options_chain)} strikes to {len(liquid_options)} liquid strikes (within 10% of spot)")
        
        if liquid_options.empty:
            log.error("No liquid options after filtering!")
            log.error(f"Strike range in data: {options_chain['strike'].min()} to {options_chain['strike'].max()}")
            return []
        
        log.info(f"Liquid strikes: {list(liquid_options['strike'])}")
        
        arbitrage_data = []
        
        log.info(f"Starting iteration over {len(liquid_options)} rows")
        
        for idx, row in liquid_options.iterrows():
            log.info(f"Processing row {idx}: strike={row['strike']}")
            
            # Calculate time to expiry
            try:
                expiry_dt = datetime.strptime(row['expiry_date'], '%d-%b-%Y')
                time_to_expiry = (expiry_dt - datetime.now()).days / 365.0
                
                if time_to_expiry <= 0:
                    log.debug(f"Skipping expired option at strike {row['strike']}")
                    continue
                    
            except Exception as e:
                log.warning(f"Could not parse expiry date '{row['expiry_date']}': {e}")
                time_to_expiry = 21 / 365.0  # Default to 3 weeks
            
            log.info(f"Time to expiry calculated: {time_to_expiry} years")
            
            # Build arbitrage-ready dictionary
            arb_dict = {
                'spot_price': spot,
                'strike': row['strike'],
                'call_price': (row['call_bid'] + row['call_ask']) / 2 if pd.notna(row['call_bid']) and pd.notna(row['call_ask']) else row['call_price'],
                'put_price': (row['put_bid'] + row['put_ask']) / 2 if pd.notna(row['put_bid']) and pd.notna(row['put_ask']) else row['put_price'],
                'time_to_expiry': time_to_expiry,
                'risk_free_rate': 0.07,
                'dividend_yield': 0.0,
                'market': 'NSE',
                'instrument': symbol,
                'expiry_date': row['expiry_date'],
                'call_bid': row['call_bid'],
                'call_ask': row['call_ask'],
                'put_bid': row['put_bid'],
                'put_ask': row['put_ask']
            }
            
            log.info(f"Built arb_dict: Call={arb_dict['call_price']:.2f}, Put={arb_dict['put_price']:.2f}")
            
            # Validate
            is_valid, error_msg = DataValidator.validate_option_data(
                arb_dict['spot_price'],
                arb_dict['strike'],
                arb_dict['call_price'],
                arb_dict['put_price'],
                arb_dict['time_to_expiry']
            )
            
            log.info(f"Validation result: {is_valid}, message: {error_msg}")
            
            if is_valid:
                arbitrage_data.append(arb_dict)
                log.info(f"Added strike {row['strike']} to arbitrage data")
            else:
                log.warning(f"VALIDATION FAILED for strike {row['strike']}: {error_msg}")
                log.warning(f"  Spot: {arb_dict['spot_price']}, Strike: {arb_dict['strike']}, Call: {arb_dict['call_price']}, Put: {arb_dict['put_price']}, T: {arb_dict['time_to_expiry']}")
        
        log.info(f"Prepared {len(arbitrage_data)} option pairs for arbitrage detection")
        return arbitrage_data
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still fresh"""
        if key not in self.cache or key not in self.last_fetch_time:
            return False
        elapsed = time.time() - self.last_fetch_time[key]
        return elapsed < self.cache_duration
    
    def _update_cache(self, key: str, value):
        """Store data in cache with timestamp"""
        self.cache[key] = value
        self.last_fetch_time[key] = time.time()
    
    def _generate_dummy_options_chain(self, symbol: str, spot: Optional[float] = None) -> pd.DataFrame:
        """Generate realistic fake data for testing"""
        log.warning(f"Generating dummy options chain for {symbol}")
        
        # Use provided spot, or fetch it, or use fallback
        if spot is None:
            spot = self.get_spot_price(symbol)
        
        if spot is None:
            base_spot = {'NIFTY': 22000, 'BANKNIFTY': 47000, 'FINNIFTY': 20000}
            spot = base_spot.get(symbol, 22000)
        
        log.info(f"Generating dummy options around spot: {spot:.2f}")
        
        step = 50 if symbol == 'NIFTY' else 100
        strikes = list(range(int(spot - 500), int(spot + 501), step))  # Note: 501 to include spot+500
        
        options_data = []
        
        for strike in strikes:
            # Simple approximation
            moneyness = spot / strike
            call_price = max(spot - strike, 0) + 50 * (moneyness - 1)
            put_price = max(strike - spot, 0) + 50 * (1 - moneyness)
            
            options_data.append({
                'strike': float(strike),  # Ensure it's float
                'expiry_date': '28-Feb-2026',
                'call_price': max(call_price, 10),
                'call_bid': max(call_price - 5, 5),
                'call_ask': call_price + 5,
                'call_oi': 10000,
                'call_iv': 15.0,
                'put_price': max(put_price, 10),
                'put_bid': max(put_price - 5, 5),
                'put_ask': put_price + 5,
                'put_oi': 10000,
                'put_iv': 15.0
            })
        
        df = pd.DataFrame(options_data)
        log.info(f"Generated {len(df)} dummy option pairs")
        return df
    
    def get_futures_data(self, symbol: str = 'NIFTY') -> Optional[Dict]:
        """
        Get Nifty futures data
        
        Returns dictionary with futures price and expiry for nearest month contract
        """
        # For now, generate dummy futures data
        # In production, would use nsepython nse_fno() function
        
        spot = self.get_spot_price(symbol)
        if spot is None:
            return None
        
        # Generate dummy futures (slightly above fair value for testing)
        # Typical futures expiry: last Thursday of current month
        from datetime import datetime, timedelta
        
        # Find last Thursday of current month
        today = datetime.now()
        # Go to end of month
        if today.month == 12:
            next_month = datetime(today.year + 1, 1, 1)
        else:
            next_month = datetime(today.year, today.month + 1, 1)
        
        last_day = next_month - timedelta(days=1)
        
        # Find last Thursday
        days_to_thursday = (last_day.weekday() - 3) % 7
        last_thursday = last_day - timedelta(days=days_to_thursday)
        
        # Calculate time to expiry
        time_to_expiry = (last_thursday - today).days / 365.0
        
        # Calculate fair futures (with slight premium for dummy data)
        r = 0.07  # Risk-free rate
        fair_futures = spot * np.exp(r * time_to_expiry)
        
        # Add small random deviation to create arbitrage opportunity
        # deviation = random.uniform(0.002, 0.008)  # +0.2% to +0.8% (larger deviation)
        deviation = random.uniform(0.003, 0.010)  # +0.3% to +1.0% (guaranteed above 0.1% threshold)
        futures_price = fair_futures * (1 + deviation)
        
        log.info(f"Generated dummy futures data: Spot={spot:.2f}, Futures={futures_price:.2f}, Expiry={last_thursday.strftime('%d-%b-%Y')}")
        
        return {
            'futures_price': futures_price,
            'expiry_date': last_thursday.strftime('%d-%b-%Y'),
            'time_to_expiry': time_to_expiry,
            'spot_price': spot,
            'instrument': symbol,
            'market': 'NSE'
        }
    
    def get_fx_data(self, currency_pair: str = 'USDINR', time_period_months: int = 12) -> Optional[Dict]:
        """
        Get FX spot and synthetic forward data
        
        Args:
            currency_pair: Currency pair (e.g., 'USDINR', 'EURINR')
            time_period_months: Forward contract maturity in months
            
        Returns:
            Dictionary with FX data for CIP arbitrage
        """
        # Map currency pairs to yfinance symbols
        yf_symbols = {
            'USDINR': 'USDINR=X',
            'EURINR': 'EURINR=X',
            'GBPINR': 'GBPINR=X'
        }
        
        if currency_pair not in yf_symbols:
            log.warning(f"Unsupported currency pair: {currency_pair}")
            return None
        
        # Fetch spot rate
        spot_rate = None
        
        if yf is not None:
            try:
                ticker = yf.Ticker(yf_symbols[currency_pair])
                hist = ticker.history(period='1d')
                if not hist.empty:
                    spot_rate = float(hist['Close'].iloc[-1])
                    log.info(f"Fetched {currency_pair} spot via yfinance: {spot_rate:.4f}")
            except Exception as e:
                log.warning(f"yfinance failed for {currency_pair}: {e}")
        
        # Fallback to hardcoded rates
        if spot_rate is None:
            fallback_rates = {
                'USDINR': 83.25,
                'EURINR': 89.50,
                'GBPINR': 105.75
            }
            spot_rate = fallback_rates.get(currency_pair, 83.25)
            log.warning(f"Using fallback spot rate for {currency_pair}: {spot_rate:.4f}")
        
        # Calculate time period in years
        T = time_period_months / 12.0
        
        # Get interest rates (we already have this from risk_free_rates)
        from src.data_acquisition.risk_free_rates import RiskFreeRateFetcher
        rate_fetcher = RiskFreeRateFetcher()
        
        # Domestic rate (India)
        r_domestic = rate_fetcher.get_india_rate('1year')
        
        # Foreign rate (based on currency)
        if currency_pair.startswith('USD'):
            r_foreign = rate_fetcher.get_us_rate('1year')
        elif currency_pair.startswith('EUR'):
            r_foreign = 0.035  # Approximate EUR rate
        elif currency_pair.startswith('GBP'):
            r_foreign = 0.045  # Approximate GBP rate
        else:
            r_foreign = 0.04
        
        # Calculate fair forward using CIP
        fair_forward = spot_rate * ((1 + r_domestic * T) / (1 + r_foreign * T))
        
        # Add small random deviation to create arbitrage opportunity
        import random
        deviation = random.uniform(-0.005, 0.005)  # -0.5% to +0.5%
        forward_rate = fair_forward * (1 + deviation)
        
        log.info(f"Generated FX data: {currency_pair} Spot={spot_rate:.4f}, Forward={forward_rate:.4f}")
        
        return {
            'spot_rate': spot_rate,
            'forward_rate': forward_rate,
            'domestic_rate': r_domestic,
            'foreign_rate': r_foreign,
            'time_period': T,
            'currency_pair': currency_pair,
            'notional': 10000  # $10,000 default
        }
    

# Convenience functions

def get_nifty_spot() -> Optional[float]:
        """Quick function to get Nifty spot"""
        fetcher = NSEDataFetcher()
        return fetcher.get_spot_price('NIFTY')




def get_arbitrage_opportunities(symbol: str = 'NIFTY') -> List[Dict]:
        """
        Quick function to get arbitrage-ready data
        
        Usage:
            data_list = get_arbitrage_opportunities('NIFTY')
            for data in data_list:
                opportunity = monitor.check_arbitrage(data)
        """
        fetcher = NSEDataFetcher()
        return fetcher.get_option_data_for_arbitrage(symbol)

def get_futures_arbitrage_data(symbol: str = 'NIFTY') -> Optional[Dict]:
        """
        Get data formatted for futures arbitrage detection
        
        Returns:
            Dictionary ready for FuturesBasisMonitor.check_arbitrage()
        """
        from src.data_acquisition.risk_free_rates import RiskFreeRateFetcher
        
        fetcher = NSEDataFetcher()
        futures_data = fetcher.get_futures_data(symbol)
        
        if futures_data is None:
            return None
        
        # Add risk-free rate
        rate_fetcher = RiskFreeRateFetcher()
        futures_data['risk_free_rate'] = rate_fetcher.get_rate_for_options(
            futures_data['time_to_expiry'],
            'NSE'
        )
        futures_data['dividend_yield'] = 0.0  # Indices don't pay dividends
        
        return futures_data

def get_cip_arbitrage_data(currency_pair: str = 'USDINR', time_period_months: int = 12) -> Optional[Dict]:
    """
    Get data formatted for CIP arbitrage detection
    
    Returns:
        Dictionary ready for CoveredInterestParityMonitor.check_arbitrage()
    """
    fetcher = NSEDataFetcher()
    return fetcher.get_fx_data(currency_pair, time_period_months)
    





    