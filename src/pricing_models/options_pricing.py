"""
Options pricing models
Implements Black-Scholes and related calculations
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple
from src.utils.logger import log
from src.utils.validators import DataValidator

class BlackScholesModel:
    """
    Black-Scholes-Merton option pricing model
    
    Assumptions:
    - European options (exercise only at maturity)
    - No dividends (or continuous dividend yield)
    - Constant volatility and risk-free rate
    - Log-normal stock prices
    """
    
    def __init__(self):
        self.model_name = "Black-Scholes-Merton"
    
    def _d1(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """
        Calculate d1 parameter in Black-Scholes formula
        
        d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate (annualized)
            sigma: Volatility (annualized)
            q: Dividend yield (annualized, continuous)
        """
        numerator = np.log(S / K) + (r - q + 0.5 * sigma**2) * T
        denominator = sigma * np.sqrt(T)
        return numerator / denominator
    
    def _d2(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """
        Calculate d2 parameter in Black-Scholes formula
        
        d2 = d1 - σ√T
        """
        return self._d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)
    
    def call_price(self, S: float, K: float, T: float, r: float, 
                   sigma: float, q: float = 0) -> float:
        """
        Calculate European call option price
        
        C = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
        
        Args:
            S: Spot price of underlying
            K: Strike price
            T: Time to expiry in years
            r: Risk-free interest rate (annualized)
            sigma: Volatility (annualized standard deviation)
            q: Continuous dividend yield (default 0)
            
        Returns:
            Call option price
        """
        if T <= 0:
            # At expiry, call worth max(S-K, 0)
            return max(S - K, 0)
        
        d1 = self._d1(S, K, T, r, sigma, q)
        d2 = self._d2(S, K, T, r, sigma, q)
        
        call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        
        return max(call, 0)  # Ensure non-negative
    
    def put_price(self, S: float, K: float, T: float, r: float,
                  sigma: float, q: float = 0) -> float:
        """
        Calculate European put option price
        
        P = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)
        
        Can also use put-call parity: P = C - S·e^(-qT) + K·e^(-rT)
        """
        if T <= 0:
            # At expiry, put worth max(K-S, 0)
            return max(K - S, 0)
        
        d1 = self._d1(S, K, T, r, sigma, q)
        d2 = self._d2(S, K, T, r, sigma, q)
        
        put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
        return max(put, 0)
    
    def calculate_greeks(self, S: float, K: float, T: float, r: float,
                        sigma: float, q: float = 0, option_type: str = 'call') -> Dict:
        """
        Calculate option Greeks
        
        Greeks measure sensitivity to various parameters:
        - Delta: ∂V/∂S (price sensitivity to underlying)
        - Gamma: ∂²V/∂S² (rate of delta change)
        - Theta: ∂V/∂T (time decay)
        - Vega: ∂V/∂σ (volatility sensitivity)
        - Rho: ∂V/∂r (interest rate sensitivity)
        
        Returns:
            Dictionary with all Greeks
        """
        d1 = self._d1(S, K, T, r, sigma, q)
        d2 = self._d2(S, K, T, r, sigma, q)
        
        sqrt_T = np.sqrt(T)
        exp_qT = np.exp(-q * T)
        exp_rT = np.exp(-r * T)
        
        # Delta
        if option_type == 'call':
            delta = exp_qT * norm.cdf(d1)
        else:  # put
            delta = exp_qT * (norm.cdf(d1) - 1)
        
        # Gamma (same for calls and puts)
        gamma = (exp_qT * norm.pdf(d1)) / (S * sigma * sqrt_T)
        
        # Theta
        term1 = -(S * norm.pdf(d1) * sigma * exp_qT) / (2 * sqrt_T)
        if option_type == 'call':
            term2 = -r * K * exp_rT * norm.cdf(d2)
            term3 = q * S * exp_qT * norm.cdf(d1)
            theta = (term1 + term2 + term3) / 365  # Per day
        else:  # put
            term2 = r * K * exp_rT * norm.cdf(-d2)
            term3 = -q * S * exp_qT * norm.cdf(-d1)
            theta = (term1 + term2 + term3) / 365
        
        # Vega (same for calls and puts)
        vega = S * exp_qT * norm.pdf(d1) * sqrt_T / 100  # Per 1% volatility change
        
        # Rho
        if option_type == 'call':
            rho = K * T * exp_rT * norm.cdf(d2) / 100  # Per 1% rate change
        else:  # put
            rho = -K * T * exp_rT * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def implied_volatility(self, market_price: float, S: float, K: float, 
                          T: float, r: float, option_type: str = 'call',
                          q: float = 0, max_iterations: int = 100,
                          tolerance: float = 1e-5) -> float:
        """
        Calculate implied volatility using Newton-Raphson method
        
        Solves: market_price = BS_price(σ) for σ
        
        Args:
            market_price: Observed option price in market
            S, K, T, r, q: Black-Scholes parameters
            option_type: 'call' or 'put'
            max_iterations: Maximum Newton-Raphson iterations
            tolerance: Convergence tolerance
            
        Returns:
            Implied volatility (annualized)
        """
        # Initial guess: approximate formula
        sigma = np.sqrt(2 * np.pi / T) * (market_price / S)
        
        for i in range(max_iterations):
            # Calculate price and vega with current sigma
            if option_type == 'call':
                price = self.call_price(S, K, T, r, sigma, q)
            else:
                price = self.put_price(S, K, T, r, sigma, q)
            
            vega = self.calculate_greeks(S, K, T, r, sigma, q, option_type)['vega'] * 100
            
            # Price difference
            diff = market_price - price
            
            # Check convergence
            if abs(diff) < tolerance:
                log.debug(f"IV converged in {i+1} iterations: {sigma:.4f}")
                return sigma
            
            # Newton-Raphson update: σ_new = σ_old + (Price_diff / Vega)
            if vega < 1e-10:
                log.warning("Vega too small, IV calculation failed")
                return np.nan
            
            sigma += diff / vega
            
            # Keep sigma in reasonable range
            sigma = np.clip(sigma, 0.01, 5.0)
        
        log.warning(f"IV did not converge after {max_iterations} iterations")
        return sigma  # Return best estimate even if not converged


class SyntheticPositions:
    """
    Create synthetic positions using put-call parity relationships
    """
    
    def __init__(self, bs_model: BlackScholesModel):
        self.bs_model = bs_model
    
    def synthetic_call(self, put_price: float, spot_price: float, 
                      strike: float, time_to_expiry: float, 
                      risk_free_rate: float) -> float:
        """
        Synthetic call = Long put + Long stock + Short bond
        
        C = P + S - K·e^(-rT)
        
        This is the put-call parity relationship rearranged for call
        """
        synthetic_call = put_price + spot_price - strike * np.exp(-risk_free_rate * time_to_expiry)
        return synthetic_call
    
    def synthetic_put(self, call_price: float, spot_price: float,
                     strike: float, time_to_expiry: float,
                     risk_free_rate: float) -> float:
        """
        Synthetic put = Long call + Short stock + Long bond
        
        P = C - S + K·e^(-rT)
        """
        synthetic_put = call_price - spot_price + strike * np.exp(-risk_free_rate * time_to_expiry)
        return synthetic_put
    
    def synthetic_stock(self, call_price: float, put_price: float,
                       strike: float, time_to_expiry: float,
                       risk_free_rate: float) -> float:
        """
        Synthetic stock = Long call + Short put + Long bond
        
        S = C - P + K·e^(-rT)
        """
        synthetic_stock = call_price - put_price + strike * np.exp(-risk_free_rate * time_to_expiry)
        return synthetic_stock