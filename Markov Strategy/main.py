# 0. pip install yfinance scikit-learn pandas matplotlib numpy
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def sharpe(returns, periods=252):
    """
    Calculate the Sharpe ratio.
    Handles cases where the standard deviation of returns is zero.
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    
    std = returns.std()
    if std == 0:
        return 0.0
    return np.sqrt(periods) * returns.mean() / std

# --- 1. Fetch data ---
ticker = "SPY"
df = yf.download(ticker, start="2015-01-01", end="2025-05-18")

if df.empty:
    raise ValueError(f"No data downloaded for ticker {ticker}. Check ticker symbol or date range.")

# --- 2. Compute enhanced features ---
df["Returns"] = df["Close"].pct_change()
df["Volatility"] = df["Returns"].rolling(20).std()

# Moving Averages and related features
for period in [10, 20, 50]:
    sma = df["Close"].rolling(period).mean()
    df[f"SMA{period}"] = sma
    df[f"SMA{period}_dist"] = ((df["Close"] - sma) / sma * 100)
    df[f"SMA{period}_slope"] = sma.pct_change(5) * 100

# RSI with multiple periods
def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta).where(delta < 0, 0).rolling(window=period).mean()
    
    rs = gain / loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.replace([np.inf, -np.inf], [100, 0])
    return rsi

for period in [14, 28]:
    df[f"RSI{period}"] = calc_rsi(df["Close"], period)

# Enhanced MACD
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
df["MACD_slope"] = df["MACD_hist"].pct_change(3) * 100

# Bollinger Bands
period_bb = 20
bb_middle = df["Close"].rolling(window=period_bb).mean()
bb_std = df["Close"].rolling(window=period_bb).std()
bb_upper = bb_middle + (bb_std * 2)
bb_lower = bb_middle - (bb_std * 2)

df["BB_middle"] = bb_middle
df["BB_std"] = bb_std
df["BB_upper"] = bb_upper
df["BB_lower"] = bb_lower
df["BB_width"] = ((bb_upper - bb_lower) / bb_middle * 100).values
df["BB_position"] = ((df["Close"] - bb_lower) / (bb_upper - bb_lower)).values

# Volume features
volume_ma = df["Volume"].rolling(20).mean()
df["Volume_MA"] = volume_ma
df["Volume_ratio"] = (df["Volume"] / volume_ma).values

# Clean up NaN values from feature engineering
df.dropna(inplace=True)

# --- 3. Enhanced target definition ---
threshold = 0.005  # 0.5% daily move
df["Next_return"] = df["Close"].pct_change().shift(-1)
df["Target"] = ((df["Next_return"] > threshold) | (df["Next_return"] < -threshold)).astype(int)
df.dropna(inplace=True)

# Feature selection
prefixes = ["SMA", "RSI", "MACD", "BB_", "Volume", "Volatility"]
feature_cols = [str(col) for col in df.columns if any(str(col).startswith(prefix) for prefix in prefixes)]
feature_cols = [f for f in feature_cols if f not in ["Next_return", "Target", "Returns"]]

X_unscaled = df[feature_cols].copy()
y = df["Target"].copy()

# --- 4. Time series cross-validation and optimization ---
tscv = TimeSeriesSplit(n_splits=5, test_size=252)  # Approximately 1 year of trading days

param_grid = {
    "n_estimators": [100, 200],  # Reduced for faster execution
    "max_depth": [3, 5],
    "min_samples_leaf": [50, 100],
    "class_weight": ["balanced"]
}

best_score = -np.inf
best_params = None
cv_results = []

print("Starting Cross-Validation...")
cv_results_list = []

for params in ParameterGrid(param_grid):
    print(f"\nEvaluating params: {params}")
    fold_sharpe_scores = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_unscaled), 1):
        print(f"  Fold {fold_idx}/{tscv.n_splits}")
        
        # Split data
        X_train = X_unscaled.iloc[train_idx].astype(float)
        X_test = X_unscaled.iloc[test_idx].astype(float)
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        
        if len(X_train.index) == 0 or len(X_test.index) == 0:
            print("Warning: Empty train or test set, skipping fold")
            continue
            
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train.values),
            index=X_train.index,
            columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test.values),
            index=X_test.index,
            columns=X_test.columns
        )
        
        # Train and evaluate
        model = RandomForestClassifier(random_state=42, **params)
        model.fit(X_train_scaled, y_train)
        
        # Get predictions and calculate returns
        predictions = model.predict(X_test_scaled)
        next_returns = df.loc[X_test.index, "Next_return"]
        strategy_returns = predictions * next_returns
        
        # Calculate Sharpe ratio
        sr = sharpe(strategy_returns)
        if np.isnan(sr):
            sr = -np.inf
        fold_sharpe_scores.append(sr)
        
        print(f"    Fold Sharpe: {sr:.4f}")
    
    # Average Sharpe ratio across folds
    avg_sharpe = np.mean(fold_sharpe_scores)
    cv_results_list.append({**params, "avg_sharpe": avg_sharpe})
    print(f"  Average Sharpe: {avg_sharpe:.4f}")
    
    if avg_sharpe > best_score:
        best_score = avg_sharpe
        best_params = params.copy()

cv_results_df = pd.DataFrame(cv_results_list)
print("\nCross-validation results:")
print(cv_results_df.sort_values("avg_sharpe", ascending=False))

print("\nBest parameters:")
print(pd.Series(best_params))
print(f"Best Average Sharpe: {best_score:.4f}")

# --- 5. Final model with best params ---
split = int(len(df) * 0.7)
X_train_final = X_unscaled.iloc[:split]
X_test_final = X_unscaled.iloc[split:]
y_train_final = y.iloc[:split]
y_test_final = y.iloc[split:]

# Scale the data
final_scaler = StandardScaler()
X_train_final_scaled = pd.DataFrame(
    final_scaler.fit_transform(X_train_final),
    index=X_train_final.index,
    columns=X_train_final.columns
)
X_test_final_scaled = pd.DataFrame(
    final_scaler.transform(X_test_final),
    index=X_test_final.index,
    columns=X_test_final.columns
)

# Train final model
print("\nTraining final model with best parameters...")
final_model = RandomForestClassifier(random_state=42, **best_params)
final_model.fit(X_train_final_scaled, y_train_final)

# Feature importance
importances = pd.Series(
    final_model.feature_importances_,
    index=X_train_final.columns
).sort_values(ascending=False)

print("\nTop 10 most important features:")
print(importances.head(10))

# Classification report
y_pred_final = final_model.predict(X_test_final_scaled)
print("\nClassification Report:")
print(classification_report(y_test_final, y_pred_final))

# Final backtest
final_df = df.iloc[split:].copy()
final_df["Strategy_returns"] = y_pred_final * final_df["Next_return"]
final_df["Cum_strategy"] = (1 + final_df["Strategy_returns"]).cumprod()
final_df["Cum_market"] = (1 + final_df["Next_return"]).cumprod()

# Calculate performance metrics
total_return_strategy = final_df["Cum_strategy"].iloc[-1] - 1
sharpe_ratio_strategy = sharpe(final_df["Strategy_returns"])
max_dd = (final_df["Cum_strategy"].cummax() - final_df["Cum_strategy"]).max()
win_rate = (final_df["Strategy_returns"] > 0).mean()

print("\nFinal Test Set Performance:")
print(f"Total Return:  {total_return_strategy:.2%}")
print(f"Sharpe Ratio:  {sharpe_ratio_strategy:.2f}")
print(f"Max Drawdown:  {max_dd:.2%}")
print(f"Win Rate:      {win_rate:.2%}")

# Plot results
plt.figure(figsize=(12, 6))
final_df["Cum_strategy"].plot(label="Strategy", color='blue')
final_df["Cum_market"].plot(label="Market", color='gray', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.legend()
plt.title(f"{ticker} Strategy vs Market Returns")
plt.xlabel("Date")
plt.ylabel("Cumulative Return (1 = 100%)")
plt.tight_layout()
plt.show()

print("\nAnalysis complete.")