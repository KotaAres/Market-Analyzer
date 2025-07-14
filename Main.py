import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# === Download data ===
data = yf.download("SPY", start="2020-01-01", end="2025-01-01")

# === Ensure 'Close' column is valid and numeric ===
if 'Close' not in data.columns:
    raise KeyError("Missing 'Close' column in downloaded data.")

data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data.dropna(subset=['Close'], inplace=True)

close = data['Close']

# === Technical indicators ===
data['SMA_20'] = SMAIndicator(close=close, window=20).sma_indicator()
data['RSI'] = RSIIndicator(close=close, window=14).rsi()

macd = MACD(close=close)
data['MACD'] = macd.macd()
data['MACD_signal'] = macd.macd_signal()

bb = BollingerBands(close=close, window=20, window_dev=2)
data['BB_upper'] = bb.bollinger_hband()
data['BB_lower'] = bb.bollinger_lband()

data.dropna(inplace=True)  # Drop rows with any NaNs from indicators

# === Signal Logic ===
data['buy_signal'] = (data['RSI'] < 30) & (data['Close'] < data['BB_lower'])
data['sell_signal'] = (data['RSI'] > 70) & (data['Close'] > data['BB_upper'])

# === Backtest ===
initial_balance = 10000
balance = initial_balance
position = 0
balance_history = []

for _, row in data.iterrows():
    price = float(row['Close'])
    if row['buy_signal'] and position == 0:
        position = balance / price
        balance = 0
    elif row['sell_signal'] and position > 0:
        balance = position * price
        position = 0
    total_equity = balance + position * price
    balance_history.append(total_equity)

data['equity_curve'] = balance_history

# === Output results ===
final_balance = balance_history[-1]
return_pct = 100 * (final_balance / initial_balance - 1)
print(f"Initial balance: ${initial_balance:.2f}")
print(f"Final balance:   ${final_balance:.2f}")
print(f"Return:          {return_pct:.2f}%")

# === Plotting ===
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['equity_curve'], label='Equity Curve')
plt.plot(data.index, data['Close'], alpha=0.3, label='Close Price')
plt.scatter(data.index[data['buy_signal']], data['Close'][data['buy_signal']],
            marker='^', color='green', label='Buy Signal')
plt.scatter(data.index[data['sell_signal']], data['Close'][data['sell_signal']],
            marker='v', color='red', label='Sell Signal')
plt.legend()
plt.title("Backtest with Buy/Sell Signals")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Machine Learning: Predict Up/Down Movement ===
data['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
features = ['SMA_20', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower']
ml_data = data.dropna(subset=features + ['target'])

X = ml_data[features]
y = ml_data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\nMachine Learning Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
