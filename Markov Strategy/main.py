# 0. pip install yfinance scikit-learn pandas matplotlib

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# 1. Fetch data
ticker = "SPY"
df = yf.download(ticker, start="2015-01-01", end="2025-05-18")

# 2. Indicators
# 2.1 Simple Moving Averages
df["SMA10"] = df["Close"].rolling(10).mean()
df["SMA50"] = df["Close"].rolling(50).mean()

# 2.2 RSI(14)
delta = df["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
rs = avg_gain / avg_loss
df["RSI14"] = 100 - (100 / (1 + rs))

# 2.3 MACD & Signal
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["MACDsig"] = df["MACD"].ewm(span=9, adjust=False).mean()

# 2.4 Bollinger Bands (20, 2σ)
mid = df["Close"].rolling(20).mean()
std = df["Close"].rolling(20).std()
df["BB_up"] = mid + 2 * std
df["BB_dn"] = mid - 2 * std

df.dropna(inplace=True)

# 3. Label creation: next‐day return direction
df["Ret"] = df["Close"].pct_change().shift(-1)
df["Target"] = (df["Ret"] > 0).astype(int)
df.dropna(inplace=True)

# 4. Train/Test split (70/30 time series)
features = ["SMA10","SMA50","RSI14","MACD","MACDsig","BB_up","BB_dn"]
X = df[features]
y = df["Target"]
split = int(len(df) * 0.7)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# 5. Train model + grid search
base = RandomForestClassifier(random_state=42)
params = {"n_estimators":[100,200], "max_depth":[3,5]}
tscv = TimeSeriesSplit(n_splits=5)
grid = GridSearchCV(base, params, cv=tscv, scoring="accuracy")
grid.fit(X_train, y_train)
model = grid.best_estimator_
print("Best params:", grid.best_params_)

# 5.1 Classification report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 6. Backtest: go long if prediction=1, else flat
test = df.iloc[split:].copy()
test["Pred"] = model.predict(X_test)
test["StratR"] = test["Pred"] * test["Ret"]
test["CumStrat"] = (1 + test["StratR"]).cumprod()
test["CumMkt"] = (1 + test["Ret"]).cumprod()

# 7. Metrics
def sharpe(returns, periods=252):
    return np.sqrt(periods) * returns.mean() / returns.std()

total_ret    = test["CumStrat"].iloc[-1] - 1
sharpe_ratio = sharpe(test["StratR"])
max_dd       = (test["CumStrat"].cummax() - test["CumStrat"]).max()
win_rate     = (test["StratR"] > 0).mean()

print(f"Total Return:  {total_ret:.2%}")
print(f"Sharpe Ratio:  {sharpe_ratio:.2f}")
print(f"Max Drawdown:  {max_dd:.2%}")
print(f"Win Rate:      {win_rate:.2%}")

# 8. Plot cumulative returns
plt.figure()
test["CumStrat"].plot(label="Strategy")
test["CumMkt"].plot(label="Market")
plt.legend()
plt.title("Cumulative Returns")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.show()

# 9. (Optional) Re-run grid search with different features or 
#    wider hyperparameter ranges to optimize further.
