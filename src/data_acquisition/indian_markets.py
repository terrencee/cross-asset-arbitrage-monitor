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
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
    
    
    def __init__(self, cache_duration_seconds: int = 60, allow_dummy: bool = False):
        """
        Initialize fetcher with caching
        
        Args:
            cache_duration_seconds: How long to keep cached data (default 60)
        """
        self.cache_duration = cache_duration_seconds
        self.cache = {}
        self.last_fetch_time = {}
        self.allow_dummy = allow_dummy
        self._session = None
        log.info("NSEDataFetcher initialized")

    def _nse_session(self) -> requests.Session:
        """Create a robust NSE session with headers + retries + cookies."""
        if self._session is not None:
            return self._session    
        s = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.7,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",)
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain",
    "Origin": "https://www.nseindia.com",
    "Connection": "keep-alive",
    "DNT": "1",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
        })
        # set cookies
        try:
            s.get("https://www.nseindia.com", timeout=15)
        except Exception:
            pass

        self._session = s
        return self._session

    def _nse_get_json(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        s = self._nse_session()

        # Refresh cookies once if needed
        if "nseappid" not in s.cookies.get_dict():
            try:
                s.get("https://www.nseindia.com", timeout=10)
            except Exception:
                pass

        try:
            r = s.get(url, params=params, timeout=20)

            ct = (r.headers.get("Content-Type") or "").lower()
            log.info(f"[NSE] GET {r.url} -> {r.status_code} | ct={ct}")

            # Log a short snippet ALWAYS (super useful for Cloudflare/HTML blocks)
            try:
                snippet = (r.text or "")[:200].replace("\n", " ")
                log.info(f"[NSE] snippet: {snippet}")
            except Exception:
                pass

            if r.status_code != 200:
                return None

            # If NSE returns HTML with 200, JSON parsing will fail -> catch below
            return r.json()

        except Exception as e:
            log.exception(f"[NSE] Exception for {url} params={params}: {e}")
            return None

        
    def _prime_option_chain(self) -> None:
        """Hit option-chain page to set additional cookies NSE expects."""
        """Prime NSE cookies/headers for option chain endpoints."""
        try:
            s = self._nse_session()
            s.get("https://www.nseindia.com", timeout=10)
            s.get("https://www.nseindia.com/option-chain", timeout=10)
            # sometimes this helps set additional cookies used by api calls
            # s.get("https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY", timeout=10)
        except Exception:
            pass
       

    
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
        
        '''
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
                '''
        # Method 1: NSE Index API (robust for indices; avoids nse_eq misuse)
        try:
            index_map = {
                "NIFTY": "NIFTY 50",
                "BANKNIFTY": "NIFTY BANK",
                "FINNIFTY": "NIFTY FIN SERVICE"
            }
            idx_name = index_map.get(symbol, symbol)
            j = self._nse_get_json(
                "https://www.nseindia.com/api/equity-stockIndices",
                params={"index": idx_name}
            )
            if j == {}:
                log.warning("[OC] option-chain-indices returned empty JSON {}. Resetting session + retrying once...")
                self._session = None  # force a fresh cookie jar
                self._prime_option_chain()
                j = self._nse_get_json(
                    "https://www.nseindia.com/api/option-chain-indices",
                    params={"symbol": symbol}
                )

            if j and "data" in j and len(j["data"]) > 0:
                row0 = j["data"][0]
                # NSE sometimes uses 'last' field for index quotes
                spot_price = float(row0.get("last", row0.get("lastPrice", 0)))
                if spot_price > 0:
                    log.info(f"Fetched {symbol} spot via NSE index API: {spot_price}")
        except Exception as e:
            log.warning(f"NSE index API failed: {e}")
        
        
        # Method 2: yfinance (fallback; may be delayed/close)
        if spot_price is None and yf is not None and symbol in self.YF_SYMBOLS:
            try:
                ticker = yf.Ticker(self.YF_SYMBOLS[symbol])
                '''hist = ticker.history(period='1d')
                if not hist.empty:
                    spot_price = float(hist['Close'].iloc[-1])
                    log.info(f"Fetched {symbol} spot via yfinance: {spot_price}")'''
                # Try intraday first
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    spot_price = float(hist["Close"].iloc[-1])
                else:
                    hist = ticker.history(period="5d")
                    if not hist.empty:
                        spot_price = float(hist["Close"].iloc[-1])
                if spot_price is not None:
                    log.info(f"Fetched {symbol} spot via yfinance: {spot_price}")
            except Exception as e:
                log.warning(f"yfinance failed: {e}")
        
        # Method 3: Explicit dummy/fallback only (never silently in live mode)
        if spot_price is None:
            if self.allow_dummy:
                fallback = {'NIFTY': 22000.0, 'BANKNIFTY': 47000.0, 'FINNIFTY': 20000.0}
                spot_price = fallback.get(symbol)
                log.warning(f"Using fallback for {symbol}: {spot_price}")
            else:
                log.error(f"Spot fetch failed for {symbol} (dummy disabled).")
                return None
            # fallback = {'NIFTY': 22000.0, 'BANKNIFTY': 47000.0, 'FINNIFTY': 20000.0}
            # spot_price = fallback.get(symbol)
            # log.warning(f"Using fallback for {symbol}: {spot_price}")
        
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
        
        #if nse_optionchain_scrapper is None:
         #   spot = self.get_spot_price(symbol)
          #  return self._generate_dummy_options_chain(symbol, spot)

        # Prefer NSE official endpoint (more reliable than scrapper)
        data = None

        log.info(f"[OC] Fetching option chain for {symbol} expiry={expiry_date} (allow_dummy={self.allow_dummy})")

        try:
            self._prime_option_chain()
            log.info("[OC] Priming done. Calling option-chain-indices API...")
            j = self._nse_get_json(
                "https://www.nseindia.com/api/option-chain-indices",
                params={"symbol": symbol}
            )
            log.info(f"[OC] option-chain-indices returned: {'OK' if j else 'None'}")

            if j and "records" in j and "data" in j["records"]:
                data = j
                try:
                    underlying = float(data["records"]["underlyingValue"])
                    self._update_cache(f"underlying_{symbol}", underlying)
                    log.info(f"[OC] underlyingValue={underlying}")
                except Exception:
                    log.warning("[OC] Could not parse underlyingValue")

        except Exception as e:
            log.exception(f"[OC] Exception while fetching option-chain-indices: {e}")
            data = None

        # Fallback to nsepython scrapper if needed
        if data is None and nse_optionchain_scrapper is not None:
            try:
                data = nse_optionchain_scrapper(symbol)
                try:
                    underlying = float(data["records"]["underlyingValue"])
                    self._update_cache(f"underlying_{symbol}", underlying)
                except Exception:
                    pass
            except Exception as e:
                log.warning(f"nse_optionchain_scrapper failed: {e}")

        # Fallback: try derivatives quote endpoint (sometimes less blocked)
        if data is None:
            try:
                self._prime_option_chain()
                q = self._nse_get_json(
                    "https://www.nseindia.com/api/quote-derivative",
                    params={"symbol": symbol}
                )
                # This endpoint structure differs; if present, you can at least detect failure reason
                if q:
                    log.warning("quote-derivative returned data but not parsed into chain yet.")
            except Exception:
                pass


        # If no live data, dummy only if enabled
        if data is None:
            if not self.allow_dummy:
                log.error("Options chain unavailable (dummy disabled).")
                return None
            spot = self.get_spot_price(symbol)
            return self._generate_dummy_options_chain(symbol, spot)

        
        
        try:
            # log.info(f"Fetching options chain for {symbol}")
            # data = nse_optionchain_scrapper(symbol)
            
            # if data is None or 'records' not in data:
             #   return self._generate_dummy_options_chain(symbol)
            
            if data is None or 'records' not in data:
                if not self.allow_dummy:
                    return None
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
        # spot = self.get_spot_price(symbol)
        # Prefer option-chain underlying (most consistent for PCP/ATM filtering)
        spot = None
       # Fetch chain first so underlying cache can populate
        options_chain = self.get_options_chain(symbol, expiry_date)
        if options_chain is None:
            log.error("Options chain is None")
            return []

        if self._is_cache_valid(f"underlying_{symbol}"):
            spot = self.cache.get(f"underlying_{symbol}")

        if spot is None:
            spot = self.get_spot_price(symbol)
        if spot is None:
            log.error("Spot price is None, cannot proceed")
            return []
        
        log.info(f"Spot price for arbitrage filtering: {spot}")
        
        '''options_chain = self.get_options_chain(symbol, expiry_date)
        if options_chain is None:
            log.error("Options chain is None")
            return []
            '''
        
        
        if options_chain.empty:
            log.error("Options chain is empty")
            return []
        
        log.info(f"Options chain retrieved: {len(options_chain)} rows")
        log.info(f"Options chain columns: {list(options_chain.columns)}")
        log.info(f"Sample strikes: {list(options_chain['strike'].head())}")

        # Filter for liquid strikes (within 10% of forward, not spot)
        # Choose a reference expiry (nearest expiry in chain if not provided)
        ref_expiry = expiry_date
        if ref_expiry is None:
            # pick the nearest (earliest) expiry present in the chain
            expiries = sorted(options_chain["expiry_date"].dropna().unique().tolist())
            if expiries:
                ref_expiry = expiries[0]

        # Compute a reference T for forward centering
        T_ref = None
        try:
            if ref_expiry:
                ref_dt = datetime.strptime(ref_expiry, "%d-%b-%Y")
                delta_sec = (ref_dt - datetime.now()).total_seconds()
                T_ref = max(delta_sec, 0) / (365.0 * 24 * 3600)
        except Exception:
            T_ref = None

        if T_ref is None or T_ref <= 0:
            # fallback (about 2–3 weeks)
            T_ref = 21 / 365.0

        # Pull a sensible risk-free rate for this maturity
        try:
            from src.data_acquisition.risk_free_rates import RiskFreeRateFetcher
            rate_fetcher = RiskFreeRateFetcher()
            r_ref = rate_fetcher.get_rate_for_options(T_ref, "NSE")
        except Exception:
            r_ref = 0.07

        q_ref = 0.0  # you can later improve this using implied carry from futures if you add live futures
        forward_ref = spot * np.exp((r_ref - q_ref) * T_ref)

        log.info(f"Forward reference for strike filtering: F={forward_ref:.2f} (Spot={spot:.2f}, r={r_ref:.4f}, T={T_ref:.4f})")

        lower_bound = forward_ref * 0.90
        upper_bound = forward_ref * 1.10

        log.info(f"Filtering strikes between {lower_bound:.2f} and {upper_bound:.2f} (±10% of forward)")
        
        '''
        # Filter for liquid strikes (within 10% of spot)
        lower_bound = spot * 0.90
        upper_bound = spot * 1.10
        
        log.info(f"Filtering strikes between {lower_bound:.2f} and {upper_bound:.2f}")
        '''
        
        liquid_options = options_chain[
            (options_chain['strike'] >= lower_bound) & 
            (options_chain['strike'] <= upper_bound)
        ]
        
        log.info(
            f"Filtering {len(options_chain)} strikes to {len(liquid_options)} liquid strikes (within 10% of the forward)")
        
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
                # expiry_dt = datetime.strptime(row['expiry_date'], '%d-%b-%Y')
                # time_to_expiry = (expiry_dt - datetime.now()).days / 365.0

                expiry_dt = datetime.strptime(row['expiry_date'], '%d-%b-%Y')
                delta_sec = (expiry_dt - datetime.now()).total_seconds()
                time_to_expiry = max(delta_sec, 0) / (365.0 * 24 * 3600)
                
                if time_to_expiry <= 0:
                    log.debug(f"Skipping expired option at strike {row['strike']}")
                    continue
                    
            except Exception as e:
                log.warning(f"Could not parse expiry date '{row['expiry_date']}': {e}")
                time_to_expiry = 21 / 365.0  # Default to 3 weeks
            
            log.info(f"Time to expiry calculated: {time_to_expiry} years")

            # Pull a sensible risk-free rate instead of hardcoding
            try:
                from src.data_acquisition.risk_free_rates import RiskFreeRateFetcher
                rate_fetcher = RiskFreeRateFetcher()
                rfr = rate_fetcher.get_rate_for_options(time_to_expiry, 'NSE')
            except Exception:
                rfr = 0.07

            # Forward proxy for parity (ATM-forward). PutCallParityMonitor uses futures_price if provided.
            forward_price = spot * np.exp((rfr - 0.0) * time_to_expiry)
            
            # Build arbitrage-ready dictionary
            arb_dict = {
                'spot_price': spot,
                'strike': row['strike'],
                'call_price': (row['call_bid'] + row['call_ask']) / 2 if pd.notna(row['call_bid']) and pd.notna(row['call_ask']) else row['call_price'],
                'put_price': (row['put_bid'] + row['put_ask']) / 2 if pd.notna(row['put_bid']) and pd.notna(row['put_ask']) else row['put_price'],
                'time_to_expiry': time_to_expiry,
                # 'risk_free_rate': 0.07,
                'risk_free_rate': rfr,
                'dividend_yield': 0.0,
                'market': 'NSE',
                'instrument': symbol,
                'expiry_date': row['expiry_date'],
                'call_bid': row['call_bid'],
                'call_ask': row['call_ask'],
                'put_bid': row['put_bid'],
                'put_ask': row['put_ask'],
                'forward_price': forward_price,  # Add forward price for PCP
                'futures_price': forward_price,  # Use forward as proxy for futures in PCP
            }
            
            log.info(f"Built arb_dict: Call={arb_dict['call_price']:.2f}, Put={arb_dict['put_price']:.2f}, F={arb_dict['futures_price']:.2f}")
            
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
    





    