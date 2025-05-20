# 0. Import required libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, AlphaDropout, Add, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split # For creating validation set
import keras_tuner
from tensorflow.keras import mixed_precision # For Mixed Precision Training

# Small epsilon to prevent division by zero
epsilon = 1e-10
AUTOTUNE = tf.data.AUTOTUNE # For tf.data.Dataset.prefetch

# --- Mixed Precision Training ---
try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print(f"Mixed precision policy set to: {mixed_precision.global_policy().name}")
except Exception as e:
    print(f"Could not set mixed precision policy: {e}. Continuing with default precision.")


# Custom Metric for Recall of Class 0
class RecallClass0(tf.keras.metrics.Metric):
    def __init__(self, name='recall_class_0', threshold=0.5, **kwargs):
        super(RecallClass0, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32) 
        y_pred_labels = tf.cast(y_pred > self.threshold, tf.float32)
        
        tn_conditions = tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred_labels, 0))
        self.true_negatives.assign_add(tf.reduce_sum(tf.cast(tn_conditions, self.dtype)))
        
        fp_conditions = tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred_labels, 1))
        self.false_positives.assign_add(tf.reduce_sum(tf.cast(fp_conditions, self.dtype)))

    def result(self):
        recall_0 = self.true_negatives / (self.true_negatives + self.false_positives + tf.keras.backend.epsilon())
        return recall_0

    def reset_state(self): 
        self.true_negatives.assign(0.)
        self.false_positives.assign(0.)

# GPU Check
try:
    print("TensorFlow version:", tf.__version__)
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"Metal GPU acceleration enabled. Found GPU(s): {physical_devices}")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}.")
    else:
        print("No GPU found by TensorFlow.")
except Exception as e:
    print(f"An error occurred during GPU configuration: {e}")


# 1. Fetch data
ticker = "SPY"
df = yf.download(ticker, start="2015-01-01", end="2024-05-18")

if df.empty: print(f"No data for {ticker}. Exiting."); exit()

# 2. Feature Engineering
print("Calculating features...")
df["SMA20"] = df["Close"].rolling(window=20).mean()
df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["STD20"] = df["Close"].rolling(window=20).std() 
df["Upper_BB"] = df["SMA20"] + (df["STD20"] * 2)
df["Lower_BB"] = df["SMA20"] - (df["STD20"] * 2)
df["BB_Width"] = (df["Upper_BB"] - df["Lower_BB"]).div(df["SMA20"] + epsilon)
tr1 = pd.DataFrame(df['High'] - df['Low'])
tr2 = pd.DataFrame(abs(df['High'] - df['Close'].shift(1)))
tr3 = pd.DataFrame(abs(df['Low'] - df['Close'].shift(1)))
tr_df = pd.concat([tr1, tr2, tr3], axis=1)
df['TR'] = tr_df.max(axis=1)
df['ATR'] = df['TR'].ewm(alpha=1/14, adjust=False, min_periods=14).mean()
df["ROC"] = df["Close"].pct_change(10)
delta = df["Close"].diff()
gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
rs = avg_gain / (avg_loss + epsilon)
df["RSI14"] = 100 - (100 / (1 + rs)); df["RSI14"] = df["RSI14"].fillna(50)
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["MACDsig"] = df["MACD"].ewm(span=9, adjust=False).mean()
volume_sma = df["Volume"].rolling(20).mean()
df["Volume_Ratio"] = df["Volume"].div(volume_sma + epsilon)
df["OBV"] = df["Volume"].where(df["Close"] > df["Close"].shift(1), -df["Volume"]).fillna(0).cumsum()
df["Daily_Ret"] = df["Close"].pct_change() 
df["Volatility"] = df["Daily_Ret"].rolling(window=20).std() 
sma40 = df["Close"].rolling(window=40).mean()
df["Vol_of_Vol"] = df["Volatility"].rolling(window=20).std()
df["Return_Skewness"] = df["Daily_Ret"].rolling(window=60).skew()

series_sma20 = df["SMA20"]; series_sma40 = sma40; series_atr = df["ATR"]
for s_name_str, s_obj_str_ref in [("SMA20", "series_sma20"), ("sma40", "series_sma40"), ("ATR", "series_atr")]:
    s_obj_val = locals()[s_obj_str_ref] 
    if isinstance(s_obj_val, pd.DataFrame):
        print(f"Debug: {s_name_str} was a DataFrame. Squeezing.")
        squeezed_s_obj = s_obj_val.squeeze()
        if isinstance(squeezed_s_obj, pd.Series):
            if s_name_str == "SMA20": series_sma20 = squeezed_s_obj
            elif s_name_str == "sma40": series_sma40 = squeezed_s_obj
            elif s_name_str == "ATR": series_atr = squeezed_s_obj
        elif s_obj_val.shape[1] > 0: 
            if s_name_str == "SMA20": series_sma20 = s_obj_val.iloc[:,0]
            elif s_name_str == "sma40": series_sma40 = s_obj_val.iloc[:,0]
            elif s_name_str == "ATR": series_atr = s_obj_val.iloc[:,0]
        else: 
            print(f"Error: {s_name_str} is an empty DataFrame after trying to squeeze. Setting component to NaN Series.")
            nan_series = pd.Series(np.nan, index=df.index)
            if s_name_str == "SMA20": series_sma20 = nan_series
            elif s_name_str == "sma40": series_sma40 = nan_series
            elif s_name_str == "ATR": series_atr = nan_series
if not (isinstance(series_sma20, pd.Series) and isinstance(series_sma40, pd.Series) and isinstance(series_atr, pd.Series)):
    df["Trend_Strength"] = np.nan
else:
    trend_strength_numerator = (series_sma20 - series_sma40).abs()
    trend_strength_denominator = (series_atr + epsilon)
    df["Trend_Strength"] = trend_strength_numerator / trend_strength_denominator

df.dropna(inplace=True)
if df.empty: print("DataFrame empty after features & dropna. Exiting."); exit()

df["Ret"] = df["Close"].pct_change().shift(-1)
df["Target"] = (df["Ret"] > 0).astype(int)
df.dropna(inplace=True)
if df.empty: print("DataFrame empty after labels & dropna. Exiting."); exit()

features_to_use = [f for f in [
    "SMA20", "EMA20", "RSI14", "ROC", "MACD", "MACDsig", "ATR", "BB_Width", 
    "STD20", "Volume_Ratio", "OBV", "Volatility", "Trend_Strength", 
    "Vol_of_Vol", "Return_Skewness"
] if f in df.columns]
if not features_to_use: print("No features available. Exiting."); exit()
print(f"Using features: {features_to_use}")

X = df[features_to_use]
y = df["Target"]

split_ratio = 0.7; split = int(len(df) * split_ratio)
if split < 50 or (len(df) - split) < 50: print(f"Warning: Small train/test size.")
if split < 1 or (len(df) - split) < 1: print("Train/test set zero size. Exiting."); exit()

X_train_pd, X_test_pd = X.iloc[:split], X.iloc[split:]
y_train_pd, y_test_pd = y.iloc[:split], y.iloc[split:]

if X_train_pd.empty or X_test_pd.empty: print("Train or test set empty. Exiting."); exit()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_pd)
X_test_scaled = scaler.transform(X_test_pd)
y_train = y_train_pd.values 
y_test = y_test_pd.values   

X_t, X_v, y_t, y_v = train_test_split(
    X_train_scaled, y_train, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_train if np.sum(y_train)>1 and np.sum(1-y_train)>1 else None
)

# --- tf.data.Dataset Pipeline with 4x Batch Sizes ---
TF_DATA_BATCH_SIZE = 128 * 4 * 4 # Was 512, now 2048
TUNER_SEARCH_BATCH_SIZE = 128 * 4 * 4 # Was 512, now 2048
FINAL_MODEL_FIT_BATCH_SIZE = 256 * 4 * 4 # Was 1024, now 4096
PREDICT_BATCH_SIZE = FINAL_MODEL_FIT_BATCH_SIZE # Keep predict batch size consistent

print(f"Using TF_DATA_BATCH_SIZE (for tf.data objects): {TF_DATA_BATCH_SIZE}")
print(f"Using TUNER_SEARCH_BATCH_SIZE: {TUNER_SEARCH_BATCH_SIZE}")
print(f"Using FINAL_MODEL_FIT_BATCH_SIZE: {FINAL_MODEL_FIT_BATCH_SIZE}")

train_dataset = tf.data.Dataset.from_tensor_slices((X_t, y_t))
train_dataset = train_dataset.shuffle(buffer_size=len(X_t)).batch(TF_DATA_BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_v, y_v))
val_dataset = val_dataset.batch(TF_DATA_BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

test_dataset_for_predict = tf.data.Dataset.from_tensor_slices((X_test_scaled, y_test))
test_dataset_for_predict = test_dataset_for_predict.batch(PREDICT_BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)


def build_model(hp):
    input_dim = X_train_scaled.shape[1] 
    inputs = Input(shape=(input_dim,))
    current_layer = Dense(
        input_dim, activation='linear', 
        kernel_regularizer=l1_l2(l1=hp.Float('l1_input', 1e-5, 1e-2, sampling='LOG', default=1e-4),
                                 l2=hp.Float('l2_input', 1e-5, 1e-2, sampling='LOG', default=1e-4))
    )(inputs)
    res1_units = hp.Int('res1_units', 64, 256, step=32, default=128)
    res1_dropout = hp.Float('res1_dropout', 0.1, 0.4, step=0.05, default=0.2)
    projected_input_for_res1 = current_layer
    if current_layer.shape[-1] != res1_units: projected_input_for_res1 = Dense(res1_units, name='proj_res1')(current_layer)
    res1 = Dense(res1_units, 'selu', kernel_initializer='lecun_normal')(current_layer)
    res1 = BatchNormalization()(res1); res1 = AlphaDropout(res1_dropout)(res1)
    res1 = Dense(res1_units, 'selu', kernel_initializer='lecun_normal')(res1)
    res1 = BatchNormalization()(res1); current_layer = Add()([projected_input_for_res1, res1])
    res2_units = hp.Int('res2_units', 32, 128, step=32, default=64)
    res2_dropout = hp.Float('res2_dropout', 0.1, 0.4, step=0.05, default=0.2)
    projected_input_for_res2 = Dense(res2_units, name='proj_res2')(current_layer)
    res2 = Dense(res2_units, 'selu', kernel_initializer='lecun_normal')(current_layer)
    res2 = BatchNormalization()(res2); res2 = AlphaDropout(res2_dropout)(res2)
    res2 = Dense(res2_units, 'selu', kernel_initializer='lecun_normal')(res2)
    res2 = BatchNormalization()(res2); current_layer = Add()([projected_input_for_res2, res2])
    final_units = hp.Int('final_units', 16, 64, step=16, default=32)
    final_dropout = hp.Float('final_dropout', 0.1, 0.4, step=0.05, default=0.2)
    current_layer = Dense(final_units, 'selu', kernel_regularizer=l2(hp.Float('l2_final', 1e-5, 1e-2, sampling='LOG', default=1e-4)), kernel_initializer='lecun_normal')(current_layer)
    current_layer = BatchNormalization()(current_layer); current_layer = AlphaDropout(final_dropout)(current_layer)
    outputs = Dense(1, activation='sigmoid', dtype='float32')(current_layer)
    model = Model(inputs=inputs, outputs=outputs)
    learning_rate = hp.Float("lr", 1e-4, 1e-2, sampling="LOG", default=1e-3)
    optimizer = Adam(learning_rate=learning_rate, amsgrad=True, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', 
                  metrics=['accuracy', RecallClass0(), tf.keras.metrics.Recall(name="recall_class_1")],
                  jit_compile=True) 
    return model

def calculate_class_weights(y_series_np):
    unique_classes, counts_unique = np.unique(y_series_np.astype(int), return_counts=True)
    if len(unique_classes) < 2 or np.any(counts_unique == 0):
        print(f"Warning: Class imbalance issue. Using default weights.")
        return {cls_val: 1.0 for cls_val in unique_classes} if len(unique_classes) > 0 else {0:1.0, 1:1.0}
    weights_values = len(y_series_np) / (len(unique_classes) * counts_unique)
    return dict(zip(unique_classes, weights_values))

class_weight_dict = calculate_class_weights(y_t) 
print(f"Class weights for training: {class_weight_dict}")

tuner = keras_tuner.Hyperband(
    build_model,
    objective=keras_tuner.Objective("val_recall_class_0", direction="max"),
    max_epochs=60, factor=3, hyperband_iterations=2,
    directory='keras_tuner_dir_tfdata_very_largebatch', 
    project_name='markov_tfdata_tuning_very_largebatch',
    overwrite=True
)

print("Starting hyperparameter search with tf.data and very large batch size...")
tuner_early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0001, restore_best_weights=True, verbose=1)

# KerasTuner's search will use the batch size defined in the train_dataset and val_dataset
tuner.search(train_dataset, epochs=100, 
             validation_data=val_dataset, 
             class_weight=class_weight_dict,
             callbacks=[tuner_early_stopping])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""Best HPs: Res1U:{best_hps.get('res1_units')},Res1D:{best_hps.get('res1_dropout')},LR:{best_hps.get('lr')}""")

model = tuner.hypermodel.build(best_hps) 
final_early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

# For model.fit, create new datasets with the FINAL_MODEL_FIT_BATCH_SIZE
# This assumes X_t, y_t, X_v, y_v are still available in their original (unbatched) form
train_dataset_final = tf.data.Dataset.from_tensor_slices((X_t, y_t))
train_dataset_final = train_dataset_final.shuffle(buffer_size=len(X_t)).batch(FINAL_MODEL_FIT_BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

val_dataset_final = tf.data.Dataset.from_tensor_slices((X_v, y_v))
val_dataset_final = val_dataset_final.batch(FINAL_MODEL_FIT_BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

history = model.fit(train_dataset_final, epochs=250, 
                    validation_data=val_dataset_final, 
                    class_weight=class_weight_dict,
                    callbacks=[final_early_stopping])

print("\nModel Evaluation with Best Hyperparameters:")
y_pred_probs_test = model.predict(X_test_scaled, batch_size=PREDICT_BATCH_SIZE, verbose=0)
y_pred_binary_test = (y_pred_probs_test > 0.5).astype(int)

print(classification_report(y_test, y_pred_binary_test, zero_division=0)) 
print(f"Accuracy on test set: {accuracy_score(y_test, y_pred_binary_test):.4f}")


def calculate_position_size(predictions_probs, confidence_threshold=0.6):
    confidence = np.abs(predictions_probs.flatten() - 0.5) * 2
    position_sizes = np.zeros_like(predictions_probs.flatten())
    confident_mask = confidence > confidence_threshold
    position_sizes[confident_mask] = confidence[confident_mask]
    position_sizes[predictions_probs.flatten() < 0.5] *= -1 
    return position_sizes

def calculate_trading_metrics(returns_series, name="Strategy"):
    if returns_series.empty or returns_series.isnull().all(): 
        return {f'{name} WR(%)': 0, f'{name} AvgRet(%)': 0, f'{name} Sharpe': np.nan, f'{name} MaxDD(%)': 0}
    win_rate = (returns_series > 0).mean() * 100
    avg_return = returns_series.mean() * 100
    std_dev = returns_series.std()
    sharpe = np.sqrt(252) * (returns_series.mean()/(std_dev+epsilon)) if (std_dev+epsilon)!=0 else np.nan
    cum_ret = (1 + returns_series).cumprod()
    run_max = cum_ret.expanding().max()
    drawdown = (cum_ret - run_max) / run_max
    max_dd = drawdown.min() * 100 if not drawdown.empty and not drawdown.isnull().all() else 0
    return {
        f'{name} WR(%)': win_rate, f'{name} AvgRet(%)': avg_return, 
        f'{name} Sharpe': sharpe, f'{name} MaxDD(%)': max_dd
    }

all_pos_sizing_metrics = []
df_test_analysis = df.iloc[split:].copy()
if 'Ret' not in df_test_analysis.columns:
    if 'Close' in df_test_analysis.columns:
        temp_ret = df_test_analysis['Close'].pct_change().shift(-1)
        if not temp_ret.empty: temp_ret.iloc[-1] = 0
        df_test_analysis['Ret'] = temp_ret
    else: df_test_analysis['Ret'] = 0.0
df_test_analysis['Ret'] = df_test_analysis['Ret'].astype(float)

for conf_thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    print(f"\n--- Evaluating with Confidence Threshold: {conf_thresh} ---")
    position_sizes_test = calculate_position_size(y_pred_probs_test, confidence_threshold=conf_thresh)
    df_test_analysis['Position_Size'] = position_sizes_test
    df_test_analysis['Strategy_Returns_PosSizing'] = df_test_analysis['Ret'] * df_test_analysis['Position_Size']
    metrics_pos_sizing = calculate_trading_metrics(df_test_analysis['Strategy_Returns_PosSizing'], name=f"PosSizing(Th={conf_thresh})")
    for metric, value in metrics_pos_sizing.items(): print(f"  {metric}: {value:.2f}")
    all_pos_sizing_metrics.append({'threshold': conf_thresh, **metrics_pos_sizing})

best_sharpe_val = -np.inf; best_conf_thresh_plot = 0.6
if all_pos_sizing_metrics:
    best_metric_entry = max(all_pos_sizing_metrics, key=lambda x: x.get(f'PosSizing(Th={x["threshold"]}) Sharpe', -np.inf))
    if best_metric_entry: 
        best_conf_thresh_plot = best_metric_entry['threshold']
        best_sharpe_val = best_metric_entry.get(f'PosSizing(Th={best_conf_thresh_plot}) Sharpe', -np.inf)
print(f"\nBest PosSizing ConfThresh (Sharpe): {best_conf_thresh_plot} (Sharpe: {best_sharpe_val:.2f})")

df_test_analysis['Predicted_Signal'] = y_pred_binary_test
df_test_analysis['Strategy_Returns_Binary'] = df_test_analysis['Ret'] * df_test_analysis['Predicted_Signal']
metrics_binary = calculate_trading_metrics(df_test_analysis['Strategy_Returns_Binary'], name="BinarySig")
print(f"\nTrading Performance (Binary Signal - Best Model):")
for metric, value in metrics_binary.items(): print(f"  {metric}: {value:.2f}")

plt.figure(figsize=(16, 12))
plt.subplot(2,2,1);
if hasattr(history, 'history') and 'loss' in history.history: plt.plot(history.history['loss'],label='Train'); plt.plot(history.history['val_loss'],label='Val'); plt.title('Best Model Loss'); plt.legend()
plt.subplot(2,2,2);
if hasattr(history, 'history') and 'accuracy' in history.history: plt.plot(history.history['accuracy'],label='Train'); plt.plot(history.history['val_accuracy'],label='Val'); plt.title('Best Model Acc'); plt.legend()

position_sizes_plot = calculate_position_size(y_pred_probs_test, confidence_threshold=best_conf_thresh_plot)
df_test_plot = df.iloc[split:].copy()
if 'Ret' not in df_test_plot.columns:
    if 'Close' in df_test_plot.columns:
        temp_ret_plot = df_test_plot['Close'].pct_change().shift(-1)
        if not temp_ret_plot.empty: temp_ret_plot.iloc[-1] = 0
        df_test_plot['Ret'] = temp_ret_plot
    else: df_test_plot['Ret'] = 0.0
df_test_plot['Ret'] = df_test_plot['Ret'].astype(float)
df_test_plot['Strategy_Returns'] = df_test_plot['Ret'] * position_sizes_plot
df_test_plot['Position_Size_Plot'] = position_sizes_plot 

plt.subplot(2,2,3);
if 'Strategy_Returns' in df_test_plot.columns and not df_test_plot['Strategy_Returns'].isnull().all(): (1+df_test_plot['Strategy_Returns']).cumprod().plot(label=f'PosSizing (Th={best_conf_thresh_plot})', ax=plt.gca())
if 'Ret' in df_test_plot.columns and not df_test_plot['Ret'].isnull().all(): (1+df_test_plot['Ret']).cumprod().plot(label='Buy & Hold', ax=plt.gca(), linestyle='--'); plt.title('Cum. Returns'); plt.legend()
plt.subplot(2,2,4);
if 'Position_Size_Plot' in df_test_plot.columns and not df_test_plot['Position_Size_Plot'].isnull().all(): df_test_plot['Position_Size_Plot'].plot(label='Position Size', ax=plt.gca(), alpha=0.7); plt.title('Position Size'); plt.legend()
plt.tight_layout(); plt.show()

print("\nScript finished. KerasTuner results in 'keras_tuner_dir_tfdata_very_largebatch'.")