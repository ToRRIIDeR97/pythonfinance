# 0. Import required libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# Enable Metal GPU acceleration
print("TensorFlow version:", tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("Metal GPU acceleration enabled")
    except:
        print("Memory growth must be set before GPUs have been initialized")

# 1. Fetch data
ticker = "SPY"
df = yf.download(ticker, start="2015-01-01", end="2025-05-18")

# 2. Enhanced Technical Indicators
# Moving Averages and Volatility
df["SMA20"] = df["Close"].rolling(window=20).mean()
df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["STD20"] = df["Close"].rolling(window=20).std()
df["Upper_BB"] = df["SMA20"] + (df["STD20"] * 2)
df["Lower_BB"] = df["SMA20"] - (df["STD20"] * 2)
df["BB_Width"] = (df["Upper_BB"] - df["Lower_BB"]).div(df["SMA20"])
df["ATR"] = (df["High"] - df["Low"]).rolling(window=14).mean()

# Momentum
df["ROC"] = df["Close"].pct_change(10)
df["RSI14"] = df["Close"].diff()
gain = df["RSI14"].clip(lower=0)
loss = -df["RSI14"].clip(upper=0)
avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
rs = avg_gain / avg_loss
df["RSI14"] = 100 - (100 / (1 + rs))

# MACD
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["MACDsig"] = df["MACD"].ewm(span=9, adjust=False).mean()

# Volume-based
volume_sma = df["Volume"].rolling(20).mean()
df["Volume_Ratio"] = df["Volume"].div(volume_sma)
df["OBV"] = df["Volume"].where(df["Close"] > df["Close"].shift(1), -df["Volume"]).fillna(0).cumsum()

# Market Regime
df["Volatility"] = df["Close"].pct_change().rolling(window=20).std()
sma40 = df["Close"].rolling(window=40).mean()
df["Trend_Strength"] = (df["SMA20"] - sma40).abs().div(df["ATR"])

df.dropna(inplace=True)

# 3. Label creation
df["Ret"] = df["Close"].pct_change().shift(-1)
df["Target"] = (df["Ret"] > 0).astype(int)
df.dropna(inplace=True)

# 4. Feature preparation
features = [
    "SMA20", "EMA20",                # Core trend indicators
    "RSI14", "ROC",                  # Momentum
    "MACD", "MACDsig",              # Trend following
    "ATR", "BB_Width", "STD20",     # Volatility
    "Volume_Ratio", "OBV",          # Volume
    "Volatility", "Trend_Strength"  # Market regime
]

X = df[features]
y = df["Target"]

# Split data with larger test set
split = int(len(df) * 0.7)  # Changed from 0.8 to 0.7 to have more test data
X_train, X_test = X[:split], X[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Scale features for better training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define cyclic learning rate scheduler
def cyclic_learning_rate(epoch):
    initial_learning_rate = 0.001
    maximal_learning_rate = 0.01
    step_size = 2000
    cycle = np.floor(1 + epoch/step_size)
    x = np.abs(epoch/step_size - 2*cycle + 1)
    lr = initial_learning_rate + (maximal_learning_rate - initial_learning_rate) * np.maximum(0, (1-x))
    return lr

# Attention mechanism
def attention_layer(inputs, attention_size):
    # Create attention vector
    attention = tf.keras.layers.Dense(attention_size, activation='tanh')(inputs)
    attention = tf.keras.layers.Dense(1, use_bias=False)(attention)
    attention_weights = tf.keras.layers.Activation('softmax')(attention)
    
    # Apply attention weights to inputs
    context_vector = tf.keras.layers.Multiply()([inputs, attention_weights])
    return context_vector

# Model architecture with wide layers and SELU activation
def create_model(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    
    # Initial feature selection layer with L1 regularization
    x = tf.keras.layers.Dense(input_dim,
                            activation='linear',
                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01))(inputs)
    
    # First residual block
    block_1 = tf.keras.layers.Dense(256, activation='selu',
                                  kernel_initializer='lecun_normal')(x)
    block_1 = tf.keras.layers.BatchNormalization()(block_1)
    block_1 = tf.keras.layers.AlphaDropout(0.1)(block_1)
    block_1 = tf.keras.layers.Dense(256, activation='selu',
                                  kernel_initializer='lecun_normal')(block_1)
    block_1 = tf.keras.layers.BatchNormalization()(block_1)
    # Add residual connection
    x = tf.keras.layers.Add()([x, block_1])
    
    # Apply attention
    x = attention_layer(x, 256)
    
    # Second residual block
    block_2 = tf.keras.layers.Dense(128, activation='selu',
                                  kernel_initializer='lecun_normal')(x)
    block_2 = tf.keras.layers.BatchNormalization()(block_2)
    block_2 = tf.keras.layers.AlphaDropout(0.1)(block_2)
    block_2 = tf.keras.layers.Dense(128, activation='selu',
                                  kernel_initializer='lecun_normal')(block_2)
    block_2 = tf.keras.layers.BatchNormalization()(block_2)
    # Add residual connection with projection
    x = tf.keras.layers.Dense(128)(x)  # Project to match dimensions
    x = tf.keras.layers.Add()([x, block_2])
    
    # Final layers with strong regularization
    x = tf.keras.layers.Dense(64,
                            activation='selu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01),
                            kernel_initializer='lecun_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.AlphaDropout(0.1)(x)
    
    # Output layer with confidence-based scaling
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Use AMSGrad variant of Adam optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        amsgrad=True,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create and train model
model = create_model(len(features))

# Helper functions
def calculate_class_weights(y):
    """Calculate balanced class weights."""
    n_samples = len(y)
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    counts = np.bincount(y)
    weights = n_samples / (n_classes * counts)
    return dict(zip(unique_classes, weights))

def calculate_position_size(predictions, confidence_threshold=0.7):
    """Calculate position sizes based on prediction confidence."""
    # Convert predictions to confidence scores [0, 1]
    confidence = np.abs(predictions - 0.5) * 2
    
    # Initialize position sizes
    position_sizes = np.zeros_like(predictions)
    
    # Only take positions when confidence is above threshold
    confident_mask = confidence > confidence_threshold
    
    # Scale position size based on confidence
    position_sizes[confident_mask] = confidence[confident_mask]
    
    # Adjust direction based on prediction
    position_sizes[predictions < 0.5] *= -1
    
    return position_sizes.flatten()

# Calculate class weights to handle imbalanced data
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Custom callback for monitoring prediction confidence
class ConfidenceCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:  # Check every 100 epochs
            train_pred = self.model.predict(X_train_scaled)
            confidence = np.abs(train_pred - 0.5) * 2  # Scale confidence to [0, 1]
            avg_confidence = np.mean(confidence)
            print(f"\nEpoch {epoch} - Average prediction confidence: {avg_confidence:.4f}")

# Calculate class weights for balanced training
class_weight_dict = calculate_class_weights(y_train)

# Train the model with enhanced parameters
history = model.fit(
    X_train_scaled, 
    y_train,
    batch_size=128,
    epochs=5000,
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=200,
            restore_best_weights=True,
            min_delta=0.0001
        ),
        tf.keras.callbacks.LearningRateScheduler(cyclic_learning_rate),
        ConfidenceCallback()
    ],
    verbose=1
)

# 7. Evaluate the model
print("\nModel Evaluation:")
y_pred = model.predict(X_test_scaled)
y_pred_binary = (y_pred > 0.5).astype(int)
print(classification_report(y_test, y_pred_binary))

# Predict with confidence-based position sizing
test_predictions = model.predict(X_test_scaled)
position_sizes = calculate_position_size(test_predictions, confidence_threshold=0.7)

# Calculate returns with position sizing
df_test = df.iloc[split:].copy()
df_test['Predicted'] = test_predictions
df_test['Position_Size'] = position_sizes
df_test['Strategy_Returns'] = df_test['Ret'] * df_test['Position_Size']

# Enhanced performance metrics
def calculate_trading_metrics(returns, position_sizes):
    """Calculate comprehensive trading metrics."""
    # Basic metrics
    win_rate = (returns > 0).mean() * 100
    avg_return = returns.mean() * 100
    
    # Risk-adjusted metrics
    sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() != 0 else 0
    
    # Drawdown analysis
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # Position analysis
    avg_position_size = np.abs(position_sizes).mean()
    position_count = (np.abs(position_sizes) > 0).sum()
    utilization = position_count / len(position_sizes)
    
    return {
        'Win Rate (%)': win_rate,
        'Average Return (%)': avg_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown,
        'Average Position Size': avg_position_size,
        'Position Utilization (%)': utilization * 100
    }

# Calculate and display metrics
metrics = calculate_trading_metrics(df_test['Strategy_Returns'], df_test['Position_Size'])
print("\nTrading Performance Metrics with Position Sizing:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.2f}")

# 8. Enhanced visualization
plt.figure(figsize=(15, 5))

# Training metrics
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 9. Calculate and display additional metrics
test_predictions = model.predict(X_test_scaled)
threshold = 0.5
signals = (test_predictions > threshold).astype(int)
actual_returns = df["Ret"].iloc[split:].values
strategy_returns = actual_returns * signals.flatten()

# Add rolling window evaluation
def calculate_rolling_metrics(returns, window_size=63):  # ~3 months
    rolling_sharpe = returns.rolling(window=window_size).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() != 0 else 0
    )
    rolling_win_rate = returns.rolling(window=window_size).apply(
        lambda x: (x > 0).mean()
    )
    return rolling_sharpe, rolling_win_rate

# Calculate rolling metrics
strategy_returns_series = pd.Series(strategy_returns)
rolling_sharpe, rolling_win_rate = calculate_rolling_metrics(strategy_returns_series)

# Calculate drawdown
cumulative_returns = (1 + strategy_returns_series).cumprod()
rolling_max = cumulative_returns.expanding().max()
drawdown = (cumulative_returns - rolling_max) / rolling_max

print("\nStrategy Performance:")
print(f"Win Rate: {(strategy_returns > 0).mean():.2%}")
print(f"Average Return: {strategy_returns.mean():.4%}")
print(f"Sharpe Ratio: {strategy_returns.mean() / strategy_returns.std() * np.sqrt(252):.2f}")
print(f"Max Drawdown: {drawdown.min():.2%}")
print("\nRolling Metrics (3-month window):")
print(f"Average Rolling Sharpe: {rolling_sharpe.mean():.2f}")
print(f"Average Rolling Win Rate: {rolling_win_rate.mean():.2%}")
print(f"Test Period Length (days): {len(strategy_returns)}")

# Visualize rolling performance
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(cumulative_returns.index, cumulative_returns.values)
plt.title('Cumulative Returns')
plt.xlabel('Trade')
plt.ylabel('Returns')

plt.subplot(3, 1, 2)
plt.plot(rolling_sharpe.index, rolling_sharpe.values)
plt.title('Rolling Sharpe Ratio (3-month window)')
plt.xlabel('Trade')
plt.ylabel('Sharpe Ratio')

plt.subplot(3, 1, 3)
plt.plot(rolling_win_rate.index, rolling_win_rate.values)
plt.title('Rolling Win Rate (3-month window)')
plt.xlabel('Trade')
plt.ylabel('Win Rate')

plt.tight_layout()
plt.show()

# Calculate trading metrics
df_test = df.iloc[split:].copy()
df_test['Predicted'] = y_pred_binary
df_test['Strategy_Returns'] = df_test['Ret'] * df_test['Predicted']

# Performance metrics
win_rate = (df_test['Strategy_Returns'] > 0).mean() * 100
avg_return = df_test['Strategy_Returns'].mean() * 100

# Calculate Sharpe Ratio (assuming daily data)
sharpe_ratio = np.sqrt(252) * (df_test['Strategy_Returns'].mean() / df_test['Strategy_Returns'].std())

# Calculate Maximum Drawdown
cumulative_returns = (1 + df_test['Strategy_Returns']).cumprod()
rolling_max = cumulative_returns.expanding().max()
drawdown = (cumulative_returns - rolling_max) / rolling_max
max_drawdown = drawdown.min() * 100

# Calculate Rolling Metrics (30-day window)
rolling_returns = df_test['Strategy_Returns'].rolling(30).mean()
rolling_std = df_test['Strategy_Returns'].rolling(30).std()
rolling_sharpe = np.sqrt(252) * (rolling_returns / rolling_std)
rolling_win_rate = df_test['Strategy_Returns'].rolling(30).apply(lambda x: (x > 0).mean() * 100)

print("\nTrading Performance Metrics:")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Average Return: {avg_return:.4f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown:.2f}%")
print(f"Average Rolling Sharpe: {rolling_sharpe.mean():.2f}")
print(f"Average Rolling Win Rate: {rolling_win_rate.mean():.2f}%")

# Plotting results
plt.figure(figsize=(12, 6))
plt.plot(df_test.index, cumulative_returns)
plt.title('Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.grid(True)
plt.show()

# Plot rolling metrics
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

ax1.plot(df_test.index, rolling_sharpe)
ax1.set_title('30-Day Rolling Sharpe Ratio')
ax1.grid(True)

ax2.plot(df_test.index, rolling_win_rate)
ax2.set_title('30-Day Rolling Win Rate')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Position sizing function
def calculate_position_size(predictions, confidence_threshold=0.7):
    confidence = np.abs(predictions - 0.5) * 2  # Scale confidence to [0, 1]
    position_sizes = np.zeros_like(predictions)
    
    # Only take positions when confidence is above threshold
    confident_mask = confidence > confidence_threshold
    
    # Scale position size based on confidence
    position_sizes[confident_mask] = confidence[confident_mask]
    
    # Normalize position sizes to [-1, 1] range
    position_sizes[predictions < 0.5] *= -1
    
    return position_sizes
