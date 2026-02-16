"""
Data validation utilities
Ensures data quality before processing
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from datetime import datetime
from src.utils.logger import log

class DataValidator:
    """Validate market data for quality and completeness"""
    
    @staticmethod
    def validate_price(price: float, min_price: float = 0, 
                      max_price: Optional[float] = None) -> bool:
        """
        Validate that a price is reasonable
        
        Args:
            price: Price to validate
            min_price: Minimum acceptable price
            max_price: Maximum acceptable price (optional)
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(price, (int, float)):
            log.warning(f"Price is not numeric: {price}")
            return False
            
        if np.isnan(price) or np.isinf(price):
            log.warning(f"Price is NaN or Inf: {price}")
            return False
            
        if price <= min_price:
            log.warning(f"Price {price} is below minimum {min_price}")
            return False
            
        if max_price is not None and price > max_price:
            log.warning(f"Price {price} exceeds maximum {max_price}")
            return False
            
        return True
    
    @staticmethod
    def validate_option_data(spot: float, strike: float, 
                            call_price: float, put_price: float,
                            time_to_expiry: float) -> Tuple[bool, str]:
        """
        Validate option data for arbitrage calculations
        
        Args:
            spot: Underlying spot price
            strike: Strike price
            call_price: Call option price
            put_price: Put option price
            time_to_expiry: Time to expiry in years
            
        Returns:
            (is_valid, error_message) tuple
        """
        # Check all prices are positive
        if not all(DataValidator.validate_price(p) for p in 
                   [spot, strike, call_price, put_price]):
            return False, "Invalid price data (negative or NaN)"
        
        # Check time to expiry
        if time_to_expiry <= 0:
            return False, f"Invalid time to expiry: {time_to_expiry}"
        
        if time_to_expiry > 5:  # More than 5 years seems unreasonable
            return False, f"Time to expiry too large: {time_to_expiry}"
        
        # No-arbitrage bounds for call option: S - K <= C <= S
        if call_price > spot:
            return False, f"Call price {call_price} exceeds spot {spot}"
        
        # For in-the-money calls
        intrinsic_call = max(spot - strike, 0)
        #if call_price < intrinsic_call * 0.95:  # Allow 5% slack for spreads
        if call_price + 1e-9 < intrinsic_call * 0.80:  # Allow wider slack for wide spreads / stale lastPrice
            return False, f"Call price {call_price} below intrinsic {intrinsic_call}"
        
        # No-arbitrage bounds for put option: K - S <= P <= K
        if put_price > strike:
            return False, f"Put price {put_price} exceeds strike {strike}"
        
        # For in-the-money puts
        intrinsic_put = max(strike - spot, 0)
        # if put_price < intrinsic_put * 0.95:
        if put_price + 1e-9 < intrinsic_put * 0.80:
            return False, f"Put price {put_price} below intrinsic {intrinsic_put}"
        
        return True, "Valid"
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, 
                          required_columns: list) -> Tuple[bool, str]:
        """
        Validate DataFrame has required structure
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            (is_valid, error_message) tuple
        """
        if df is None or df.empty:
            return False, "DataFrame is empty or None"
        
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"
        
        # Check for sufficient data
        if len(df) == 0:
            return False, "DataFrame has no rows"
        
        return True, "Valid"
    
    @staticmethod
    def validate_fx_data(spot_rate: float, forward_rate: float,
                        domestic_rate: float, foreign_rate: float,
                        time_to_maturity: float) -> Tuple[bool, str]:
        """
        Validate FX parity data
        
        Args:
            spot_rate: Spot exchange rate
            forward_rate: Forward exchange rate
            domestic_rate: Domestic interest rate (annualized)
            foreign_rate: Foreign interest rate (annualized)
            time_to_maturity: Time to forward maturity (years)
            
        Returns:
            (is_valid, error_message) tuple
        """
        # Validate exchange rates
        if not all(DataValidator.validate_price(r) for r in [spot_rate, forward_rate]):
            return False, "Invalid exchange rate"
        
        # Validate interest rates (allow negative rates, but check reasonableness)
        if abs(domestic_rate) > 0.5 or abs(foreign_rate) > 0.5:  # 50% seems extreme
            return False, f"Interest rates seem unreasonable: {domestic_rate}, {foreign_rate}"
        
        # Check time to maturity
        if time_to_maturity <= 0 or time_to_maturity > 10:
            return False, f"Invalid time to maturity: {time_to_maturity}"
        
        return True, "Valid"