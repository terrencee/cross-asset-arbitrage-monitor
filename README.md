# Cross-Asset Arbitrage Monitor

**Real-time arbitrage detection across Options, Futures, and FX markets**

Developed by **Adhiraj** | IIT Roorkee | Financial Engineering Course Project

**Link** : https://cross-asset-arbitrage-monitor-terradragonfe.streamlit.app/

##  Overview

This system monitors **three fundamental parity conditions** across different asset classes to identify risk-free arbitrage opportunities:

1. **Put-Call Parity** (Options + Equity)
2. **Futures Basis** (Spot + Futures)  
3. **Covered Interest Rate Parity** (FX + Interest Rates) ← True Cross-Asset

##  Live Demo

**Dashboard URL:** [Will be added after deployment]

##  Features

-  **Real-time arbitrage detection** across 3 different markets
-  **Multi-page interactive dashboard** built with Streamlit
-  **Automatic fair price calculation** using Black-Scholes and cost-of-carry models
-  **Transaction cost modeling** for realistic profit estimation
-  **Step-by-step execution instructions** for each opportunity
-  **Multi-source data acquisition** (NSE API, Yahoo Finance, synthetic data)
-  **Comprehensive analytics** with Plotly visualizations

##  Arbitrage Strategies

### 1. Put-Call Parity
**Relationship:** `C - P = S·e^(-qT) - K·e^(-rT)`

**Strategies:**
- Conversion Arbitrage (call overpriced)
- Reversal Arbitrage (put overpriced)

### 2. Futures Basis (Cost-of-Carry)
**Relationship:** `F = S·e^((r-q)T)`

**Strategies:**
- Cash-and-Carry (futures overpriced)
- Reverse Cash-and-Carry (futures underpriced)

### 3. Covered Interest Rate Parity
**Relationship:** `F/S = (1 + r_domestic) / (1 + r_foreign)`

**Strategies:**
- Borrow-Invest (forward overvalued)
- Invest-Borrow (forward undervalued)

##  Technology Stack

- **Python 3.11**
- **Streamlit** - Web dashboard framework
- **Plotly** - Interactive visualizations
- **NumPy & Pandas** - Data processing
- **yfinance** - Market data API
- **nsepython** - NSE data API

##  Project Structure
```
cross_asset_arbitrage_monitor/
├── src/
│   ├── arbitrage_monitors/     # Core arbitrage detection logic
│   ├── data_acquisition/       # Market data fetchers
│   ├── pricing/                # Black-Scholes pricing engine
│   └── utils/                  # Logging, validation
├── dashboard/
│   ├── app.py                  # Main dashboard
│   └── pages/                  # Individual monitor pages
├── tests/                      # Test suite
├── requirements.txt
└── README.md
```

##  Local Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/cross_asset_arbitrage_monitor.git
cd cross_asset_arbitrage_monitor

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard/app.py
```

##  Usage

1. **Select Markets:** Choose index (NIFTY/BANKNIFTY) and currency pair
2. **Enable Monitors:** Toggle which arbitrage strategies to run
3. **Adjust Thresholds:** Set minimum profit and deviation requirements
4. **View Opportunities:** Expandable cards show full details and execution steps
5. **Analyze:** Use analytics tab for profit distribution and summary statistics

##  Documentation

Complete documentation available in the dashboard:
- **Put-Call Parity** page - Options arbitrage theory
- **Futures Basis** page - Cost-of-carry explanation
- **Interest Rate Parity** page - Cross-asset arbitrage guide
- **Documentation** page - Full user manual and FAQ

##  Educational Value

This project demonstrates:
- Advanced derivative pricing (Black-Scholes model)
- Multi-asset arbitrage strategies
- Real-time data acquisition and processing
- Professional dashboard development
- Transaction cost modeling
- Error handling and data validation

##  Disclaimer

This system is for **educational purposes only**. 

- Not financial advice
- Dummy data may show artificial opportunities
- Real arbitrage requires consideration of execution risk, slippage, and market impact
- Consult financial professionals before trading

##  Author

**Adhiraj**  
MBA Student - IIT Roorkee  
Financial Engineering Course Project

##  License

This project is created for academic purposes.

##  Acknowledgments

- IIT Roorkee Financial Engineering Course
- NSE Python API developers
- Streamlit community