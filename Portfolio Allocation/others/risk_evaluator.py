import yfinance as yf
import numpy as np
import pandas as pd
from fredapi import Fred
import requests
import json # Import json library

# ===========================================
# Configuration - Replace API keys as needed.
# ===========================================
FRED_API_KEY = "945b726143dfe4b8c28cc7a69a15c8d0"  # Replace with your actual FRED API key
# IMF_API_KEY = "YOUR_IMF_API_KEY"    # Removed - No longer used

# ===========================================
# API Integration Modules
# ===========================================

class YahooFinanceAPI:
    @staticmethod
    def get_stock_data(ticker):
        """
        Retrieve key stock data from Yahoo Finance using yfinance.
        Returns a dict with keys:
          - beta: Relative volatility.
          - volatility: Annualized volatility computed from daily returns.
          - debt_to_equity: Debt-to-equity ratio.
          - earnings_volatility: (Simulated placeholder).
          - qualitative_score: (Simulated placeholder for qualitative factors).
        """
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get beta; default to 1 if missing.
        beta = info.get("beta", 1.0)

        # Download historical data for volatility calculation (last 1 year)
        hist = stock.history(period="1y")
        if hist.empty:
            volatility = 0.2  # default volatility if no history available
        else:
            hist["returns"] = hist["Close"].pct_change()
            volatility = np.sqrt(252) * hist["returns"].std()  # annualized volatility

        # Debt-to-equity ratio may come as a string or number; try converting.
        try:
            debt_to_equity = float(info.get("debtToEquity", 0))
        except Exception:
            debt_to_equity = 0

        # Earnings volatility and qualitative score are not provided by Yahoo Finance.
        # In a production system, these would be derived from additional data sources.
        earnings_volatility = 0.1  # Placeholder value
        qualitative_score = 7      # Placeholder score (scale 0-10)

        return {
            "beta": beta,
            "volatility": volatility,
            "debt_to_equity": debt_to_equity,
            "earnings_volatility": earnings_volatility,
            "qualitative_score": qualitative_score
        }


class FREDAPI:
    def __init__(self, api_key):
        self.fred = Fred(api_key=api_key)

    def get_indicator(self, series_id, start_date=None, end_date=None):
        """
        Retrieve a time series from FRED.
        """
        try:
            data = self.fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            return data
        except Exception as e:
            print(f"Error retrieving {series_id} from FRED: {e}")
            return None


# class IMFAPI: # Commenting out the entire class as it's no longer used
#     @staticmethod
#     def get_country_gdp_growth(country_code):
#         """
#         Dummy function to simulate fetching GDP growth data from the IMF API.
#         In a production system, you would use requests to call the IMF API endpoint.
#         """
#         dummy_data = {
#             "USA": 2.5,
#             "CHN": 4.0,
#             "ARG": 0.5
#         }
#         return dummy_data.get(country_code.upper(), 2.0)
# 
#     @staticmethod
#     def get_country_inflation(country_code):
#         """
#         Dummy function to simulate fetching inflation data from the IMF API.
#         """
#         dummy_data = {
#             "USA": 2.5,
#             "CHN": 3.0,
#             "ARG": 40.0
#         }
#         return dummy_data.get(country_code.upper(), 2.0)


class MarketDataAPI:
    def __init__(self, fred_api_key, config_file_path="fred_config.json"):
        self.fred_api = FREDAPI(fred_api_key)
        self.config_file_path = config_file_path
        self.fred_series_map = self._load_config()

    def _load_config(self):
        """Loads the FRED series configuration from a JSON file."""
        try:
            with open(self.config_file_path, 'r') as f:
                config = json.load(f)
                print(f"Loaded FRED configuration from {self.config_file_path}")
                return config
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {self.config_file_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.config_file_path}")
            return {}
        except Exception as e:
            print(f"An error occurred loading config: {e}")
            return {}

    def _get_latest_fred_value(self, series_id, default=0.0, calculate_pct_change=False):
        """Helper to get the latest non-NaN value or calculate pct change from a FRED series."""
        series = self.fred_api.get_indicator(series_id)
        if series is not None and not series.empty:
            series_clean = series.dropna()
            if series_clean.empty:
                print(f"    Series {series_id} empty after dropna. Using default: {default}")
                return default

            if calculate_pct_change:
                if len(series_clean) >= 2:
                    # Calculate percentage change from second-to-last to last
                    last_value = series_clean.iloc[-1]
                    prev_value = series_clean.iloc[-2]
                    if pd.notna(prev_value) and prev_value != 0:
                        pct_change = ((last_value / prev_value) - 1) * 100
                        print(f"    Fetched {series_id}: Calculated {pct_change:.2f}% change (from {series_clean.index[-2].date()} to {series_clean.index[-1].date()}) ")
                        return pct_change
                    else:
                        print(f"    Could not calculate pct change for {series_id} (prev value issue). Using default: {default}")
                        return default
                else:
                    print(f"    Not enough data points (need 2) to calculate pct change for {series_id}. Using default: {default}")
                    return default
            else:
                # Get the last non-NaN value directly
                last_value = series_clean.iloc[-1]
                print(f"    Fetched {series_id}: {last_value:.2f} (from {series_clean.index[-1].date()})")
                return last_value
        else:
            print(f"    Could not fetch or empty series for {series_id}. Using default: {default}")
            return default

    def get_market_data(self, country):
        """
        Assemble market-level data for a given country using FRED config.
          - GDP growth, inflation from FRED (based on config)
          - Political risk and Debt-to-GDP (simulated values)
        """
        country_code = country.upper()
        print(f"Fetching market data for {country_code}...")

        country_config = self.fred_series_map.get(country_code)

        if not country_config:
            print(f"Warning: No configuration found for country {country_code} in {self.config_file_path}. Using defaults.")
            gdp_growth = 2.0 # Default GDP growth
            inflation = 2.0 # Default inflation
        else:
            gdp_series_id = country_config.get("gdp_growth")
            inflation_series_id = country_config.get("inflation")
            gdp_is_growth_rate = country_config.get("gdp_source_is_growth_rate", False)

            if not gdp_series_id or not inflation_series_id:
                 print(f"Warning: Missing 'gdp_growth' or 'inflation' series ID for {country_code} in config. Using defaults.")
                 gdp_growth = 2.0
                 inflation = 2.0
            else:
                # Fetch/Calculate GDP Growth based on config flag
                gdp_growth = self._get_latest_fred_value(
                    gdp_series_id, 
                    default=2.0, 
                    calculate_pct_change=not gdp_is_growth_rate # Calculate if source is NOT already a growth rate
                )
                # Fetch Inflation directly
                inflation = self._get_latest_fred_value(inflation_series_id, default=2.0)

        # Simulated political risk (0-10 scale) and debt-to-GDP percentages
        political_risk = {"USA": 3, "CHN": 6, "ARG": 9}.get(country_code, 5)
        debt_to_gdp = {"USA": 105, "CHN": 80, "ARG": 120}.get(country_code, 100)

        print(f"  Using -> GDP Growth: {gdp_growth:.2f}%, Inflation: {inflation:.2f}%" )
        return {
            "gdp_growth": gdp_growth,
            "inflation": inflation,
            "political_risk": political_risk,
            "debt_to_gdp": debt_to_gdp
        }

# ===========================================
# Risk Evaluator Classes
# ===========================================

class StockRiskEvaluator:
    def __init__(self, weights=None):
        # Default weights for each risk factor (they sum to 1)
        if weights is None:
            self.weights = {
                "beta": 0.15,
                "volatility": 0.15,
                "debt": 0.25,
                "earnings_consistency": 0.25,
                "qualitative": 0.20
            }
        else:
            self.weights = weights

    def evaluate(self, stock_data):
        """
        Expects stock_data as a dict with:
          - beta, volatility, debt_to_equity, earnings_volatility, qualitative_score.
        Returns a risk score (0=low risk, 100=high risk) and a breakdown.
        """
        beta = stock_data.get("beta", 1)
        volatility = stock_data.get("volatility", 0.2)  # default 20%
        debt = stock_data.get("debt_to_equity", 50)       # percentage
        earnings_vol = stock_data.get("earnings_volatility", 0.1)
        qualitative = stock_data.get("qualitative_score", 5)  # scale 0-10

        # Calculate individual risk components
        beta_risk = max(beta - 1, 0) * 100
        base_volatility = 0.2
        volatility_risk = abs(volatility - base_volatility) / base_volatility * 50
        threshold_debt = 100
        debt_risk = (debt - threshold_debt) / threshold_debt * 100 if debt > threshold_debt else 0
        earnings_risk = earnings_vol * 100
        qualitative_risk = (10 - qualitative) * 10

        # Combine components with weights
        risk_score = (self.weights["beta"] * beta_risk +
                      self.weights["volatility"] * volatility_risk +
                      self.weights["debt"] * debt_risk +
                      self.weights["earnings_consistency"] * earnings_risk +
                      self.weights["qualitative"] * qualitative_risk)

        # Clamp score between 0 and 100
        risk_score = max(min(risk_score, 100), 0)

        breakdown = {
            "beta_risk": beta_risk,
            "volatility_risk": volatility_risk,
            "debt_risk": debt_risk,
            "earnings_risk": earnings_risk,
            "qualitative_risk": qualitative_risk
        }
        return {"risk_score": risk_score, "breakdown": breakdown}


class MarketRiskEvaluator:
    def __init__(self, weights=None):
        if weights is None:
            self.weights = {
                "gdp_growth": 0.30,
                "inflation": 0.30,
                "political": 0.20,
                "debt_to_gdp": 0.20
            }
        else:
            self.weights = weights

    def evaluate(self, market_data):
        """
        Expects market_data as a dict with:
          - gdp_growth, inflation, political_risk, debt_to_gdp.
        Returns a risk score (0=low risk, 100=high risk) and a breakdown.
        """
        gdp_growth = market_data.get("gdp_growth", 2)
        inflation = market_data.get("inflation", 2)
        political_risk = market_data.get("political_risk", 5)
        debt_to_gdp = market_data.get("debt_to_gdp", 60)

        gdp_risk = max(3 - gdp_growth, 0) * 20
        inflation_risk = abs(inflation - 2) * 20
        political_risk_scaled = political_risk * 10
        threshold = 100
        debt_risk = (debt_to_gdp - threshold) / threshold * 100 if debt_to_gdp > threshold else 0

        risk_score = (self.weights["gdp_growth"] * gdp_risk +
                      self.weights["inflation"] * inflation_risk +
                      self.weights["political"] * political_risk_scaled +
                      self.weights["debt_to_gdp"] * debt_risk)

        risk_score = max(min(risk_score, 100), 0)
        breakdown = {
            "gdp_risk": gdp_risk,
            "inflation_risk": inflation_risk,
            "political_risk": political_risk_scaled,
            "debt_risk": debt_risk
        }
        return {"risk_score": risk_score, "breakdown": breakdown}

# ===========================================
# Main Risk Engine: API Integration and Evaluation
# ===========================================

def main():
    # ----- STOCK RISK EVALUATION -----
    tickers = ["AAPL", "TSLA", "XYZ"]
    stock_evaluator = StockRiskEvaluator()
    print("=== Stock Risk Evaluations ===")
    for ticker in tickers:
        stock_data = YahooFinanceAPI.get_stock_data(ticker)
        result = stock_evaluator.evaluate(stock_data)
        print(f"Ticker: {ticker}")
        print(f"  Overall Risk Score: {result['risk_score']:.2f} (0=low risk, 100=high risk)")
        print("  Breakdown:")
        for factor, value in result["breakdown"].items():
            print(f"    {factor}: {value:.2f}")
        print("-" * 40)

    # ----- MARKET RISK EVALUATION -----
    market_data_api = MarketDataAPI(FRED_API_KEY) # Config file defaults to fred_config.json
    market_evaluator = MarketRiskEvaluator()
    # countries = ["USA", "CHN", "ARG"] # Removed hardcoded list

    print("\n=== Market Risk Evaluations ===")
    if not market_data_api.fred_series_map: # Check if config loaded successfully
        print("Could not load FRED config. Skipping market risk evaluations.")
        return
        
    # Get countries dynamically from the loaded config keys
    countries_to_evaluate = list(market_data_api.fred_series_map.keys())
    print(f"Evaluating market risk for countries found in config: {countries_to_evaluate}")

    for country in countries_to_evaluate:
        market_data = market_data_api.get_market_data(country)
        if market_data: # Check if data was retrieved
            result = market_evaluator.evaluate(market_data)
            print(f"Country: {country}")
            print(f"  Overall Risk Score: {result['risk_score']:.2f} (0=low risk, 100=high risk)")
            print("  Breakdown:")
            for factor, value in result["breakdown"].items():
                print(f"    {factor}: {value:.2f}")
        else:
            print(f"Could not retrieve market data for {country}.")
        print("-" * 40)

if __name__ == "__main__":
    main()
