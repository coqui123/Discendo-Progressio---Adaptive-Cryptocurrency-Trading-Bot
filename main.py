import ccxt
import os
from datetime import datetime
import pandas as pd
import time
import yfinance as yf
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

TAKER_FEE = 0.001
MAKER_FEE = 0.0
RETRAIN_INTERVAL = 900  # 15 minutes in seconds

exchange = ccxt.mexc()

def fetch_crypto_price(symbol):
    file_path = f"{symbol}_price_history.txt"
    with open(file_path, 'r') as file:
        last_line = file.readlines()[-1]
        _, _, _, _, close_price = last_line.strip().split(',')
    return float(close_price)

def initialize_price_history(symbol, start_date="2020-01-01", interval="1d"):
    file_path = f"{symbol}_price_history.txt"
    if not os.path.exists(file_path):
        print(f"Fetching historical data for {symbol} from Yahoo Finance")
        ticker = yf.Ticker(f"{symbol}-USD")
        data = ticker.history(start=start_date, interval=interval)
        with open(file_path, 'w') as file:
            for index, row in data.iterrows():
                timestamp = index.strftime('%Y-%m-%d')
                file.write(f"{timestamp},{row['Open']},{row['High']},{row['Low']},{row['Close']}\n")
        print(f"Historical data for {symbol} has been saved to {file_path}")
    else:
        print(f"Price history file for {symbol} already exists")

def read_balances_from_file():
    balances = {}
    usd_balance = None
    if os.path.exists("balances.txt"):
        with open("balances.txt", 'r') as file:
            for line in file:
                currency, amount = line.strip().split(',')
                if currency == "USD":
                    usd_balance = float(amount)
                else:
                    balances[currency] = float(amount)
    if usd_balance is None:
        usd_balance = 2000.0
    return usd_balance, balances

def write_balances_to_file(usd_balance, balances):
    with open("balances.txt", 'w') as file:
        file.write(f"USD,{usd_balance}\n")
        for currency, amount in balances.items():
            file.write(f"{currency},{amount}\n")

def record_trade_history(symbol, amount, price, trade_type):
    file_path = f"{symbol}_trade_history.txt"
    with open(file_path, 'a') as file:
        file.write(f"{datetime.utcnow().isoformat()},{trade_type},{amount},{price},{amount * price}\n")

def calculate_indicators(prices):
    prices['sma_20'] = prices['price'].rolling(window=20).mean()
    prices['sma_50'] = prices['price'].rolling(window=50).mean()
    prices['rsi'] = calculate_rsi(prices['price'], 14)
    prices['macd'], prices['signal'], _ = calculate_macd(prices['price'])
    prices['atr'] = calculate_atr(prices[['high', 'low', 'close']], 14)
    prices['bollinger_upper'], prices['bollinger_lower'] = calculate_bollinger_bands(prices['price'], 20, 2)
    prices.dropna(inplace=True)
    return prices

def calculate_rsi(prices, period):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal, macd - signal

def calculate_atr(prices, period):
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift())
    low_close = np.abs(prices['low'] - prices['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=period).mean()

def calculate_bollinger_bands(prices, period, std_dev):
    sma = prices.rolling(window=period).mean()
    rolling_std = prices.rolling(window=period).std()
    upper_band = sma + (rolling_std * std_dev)
    lower_band = sma - (rolling_std * std_dev)
    return upper_band, lower_band

def prepare_data(prices):
    prices['next_price'] = prices['price'].shift(-1)
    prices['target'] = np.where(prices['next_price'] > prices['price'], 1, 0)
    features = ['sma_20', 'sma_50', 'rsi', 'macd', 'signal', 'atr', 'bollinger_upper', 'bollinger_lower']
    X = prices[features]
    y = prices['target']
    return X, y

def train_ensemble_model(X, y):
    if len(X) == 0 or len(y) == 0:
        print("Not enough data to train the ensemble model.")
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = XGBClassifier(n_estimators=100, random_state=42)
    lgbm_model = LGBMClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    lgbm_model.fit(X_train, y_train)
    ensemble_predictions = (rf_model.predict(X_test) + xgb_model.predict(X_test) + lgbm_model.predict(X_test)) / 3
    ensemble_predictions = np.round(ensemble_predictions)
    accuracy = accuracy_score(y_test, ensemble_predictions)
    precision = precision_score(y_test, ensemble_predictions)
    recall = recall_score(y_test, ensemble_predictions)
    f1 = f1_score(y_test, ensemble_predictions)
    print(f"Ensemble Model Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    return rf_model, xgb_model, lgbm_model

def make_trading_decision(rf_model, xgb_model, lgbm_model, latest_data):
    rf_features = ['sma_20', 'sma_50', 'rsi', 'macd', 'signal', 'atr', 'bollinger_upper', 'bollinger_lower']
    X_latest = latest_data[rf_features]
    rf_prediction = rf_model.predict(X_latest)
    xgb_prediction = xgb_model.predict(X_latest)
    lgbm_prediction = lgbm_model.predict(X_latest)
    ensemble_prediction = (rf_prediction + xgb_prediction + lgbm_prediction) / 3
    return 1 if ensemble_prediction > 0.5 else 0

def retrain_models(prices_data):
    X, y = prepare_data(prices_data)
    rf_model, xgb_model, lgbm_model = train_ensemble_model(X, y)
    print(f"Retrained models at {datetime.now()}")
    return rf_model, xgb_model, lgbm_model

def should_retrain(last_retrain_time, retrain_interval=RETRAIN_INTERVAL):
    return time.time() - last_retrain_time > retrain_interval

def save_models(rf_model, xgb_model, lgbm_model, prefix=''):
    joblib.dump(rf_model, f'{prefix}rf_model.joblib')
    joblib.dump(xgb_model, f'{prefix}xgb_model.joblib')
    joblib.dump(lgbm_model, f'{prefix}lgbm_model.joblib')

def load_models(prefix=''):
    rf_model = joblib.load(f'{prefix}rf_model.joblib')
    xgb_model = joblib.load(f'{prefix}xgb_model.joblib')
    lgbm_model = joblib.load(f'{prefix}lgbm_model.joblib')
    return rf_model, xgb_model, lgbm_model

def trade_and_hedge(cryptos, usd_balance, balances, rf_model, xgb_model, lgbm_model):
    prices = {crypto: fetch_crypto_price(crypto) for crypto in cryptos}
    for crypto, price in prices.items():
        crypto_name = crypto.replace("USDT", "")
        file_path = f"{crypto}_price_history.txt"
        if not os.path.exists(file_path):
            print(f"Price history file not found for {crypto}")
            continue
        prices_data = pd.read_csv(file_path, header=None, names=['timestamp', 'price', 'high', 'low', 'close'])
        prices_data['price'] = prices_data['price'].astype(float)
        prices_data = calculate_indicators(prices_data)
        X, y = prepare_data(prices_data)
        if len(X) < 1:
            print(f"Not enough data to make a prediction for {crypto}")
            continue
        trade_decision = make_trading_decision(rf_model, xgb_model, lgbm_model, prices_data.tail(1))
        min_trade_amount = 5.0
        trade_amount_usd = 500000.0
        if trade_amount_usd < min_trade_amount:
            continue
        if trade_decision == 0:
            crypto_balance = balances.get(crypto_name, 0.0)
            if crypto_balance * price >= trade_amount_usd and crypto_balance > 0.0:
                crypto_amount = trade_amount_usd / price
                fee = trade_amount_usd * TAKER_FEE
                net_trade_amount_usd = trade_amount_usd - fee
                balances[crypto_name] = crypto_balance - crypto_amount
                usd_balance += net_trade_amount_usd
                print(f"Short sold {crypto_amount:.10f} of {crypto} for ${net_trade_amount_usd:.2f} (Fee: ${fee:.2f})")
                record_trade_history(crypto, -crypto_amount, price, "sell")
        else:
            if usd_balance >= trade_amount_usd:
                crypto_amount = trade_amount_usd / price
                fee = trade_amount_usd * TAKER_FEE
                net_trade_amount_usd = trade_amount_usd - fee
                balances[crypto_name] = balances.get(crypto_name, 0.0) + crypto_amount
                usd_balance -= net_trade_amount_usd
                print(f"Bought {crypto_amount:.10f} of {crypto} for ${net_trade_amount_usd:.2f} (Fee: ${fee:.2f})")
                record_trade_history(crypto, crypto_amount, price, "buy")
    return usd_balance, balances

def fetch_latest_data(symbol):
    file_path = f"{symbol}_price_history.txt"
    with open(file_path, 'r') as file:
        lines = file.readlines()
    last_line = lines[-1]
    timestamp, open_price, high, low, close = last_line.strip().split(',')
    new_data = pd.DataFrame({
        'timestamp': [timestamp],
        'price': [float(close)],
        'high': [float(high)],
        'low': [float(low)],
        'close': [float(close)]
    })
    return calculate_indicators(new_data)

def main():
    usd_balance, balances = read_balances_from_file()
    cryptos = ["XTZ"]  # Add more cryptocurrencies as needed

    for crypto in cryptos:
        try:
            initialize_price_history(crypto)
        except Exception as e:
            print(f"Error initializing price history for {crypto}: {e}")

    prices_data = pd.DataFrame()
    for crypto in cryptos:
        file_path = f"{crypto}_price_history.txt"
        if not os.path.exists(file_path):
            print(f"Price history file not found for {crypto}")
            continue
        crypto_prices = pd.read_csv(file_path, header=None, names=['timestamp', 'price', 'high', 'low', 'close'])
        crypto_prices['price'] = crypto_prices['price'].astype(float)
        crypto_prices['high'] = crypto_prices['high'].astype(float)
        crypto_prices['low'] = crypto_prices['low'].astype(float)
        crypto_prices['close'] = crypto_prices['close'].astype(float)
        crypto_prices = calculate_indicators(crypto_prices)
        prices_data = pd.concat([prices_data, crypto_prices])

    if len(prices_data) == 0:
        print("Not enough data to train the models. Please ensure price history files are populated.")
        return

    try:
        rf_model, xgb_model, lgbm_model = load_models()
        print("Loaded existing models")
    except:
        print("Training new models")
        rf_model, xgb_model, lgbm_model = retrain_models(prices_data)
        save_models(rf_model, xgb_model, lgbm_model)

    last_retrain_time = time.time()
    iteration = 0

    while True:
        iteration += 1
        print(f"\nIteration {iteration} - {datetime.now()}")

        if should_retrain(last_retrain_time):
            print("Retraining models")
            rf_model, xgb_model, lgbm_model = retrain_models(prices_data)
            save_models(rf_model, xgb_model, lgbm_model)
            last_retrain_time = time.time()

        usd_balance, balances = trade_and_hedge(cryptos, usd_balance, balances, rf_model, xgb_model, lgbm_model)
        write_balances_to_file(usd_balance, balances)

        current_portfolio_value = usd_balance
        for crypto, amount in balances.items():
            try:
                price = fetch_crypto_price(f"{crypto}USDT")
                current_portfolio_value += amount * price
            except Exception as e:
                print(f"Error fetching price for {crypto}: {e}")

        print(f"Total Portfolio Value in USD: {current_portfolio_value:.2f}")

        # Update prices_data with new data
        for crypto in cryptos:
            new_data = fetch_latest_data(crypto)
            prices_data = pd.concat([prices_data, new_data]).tail(10000)  # Keep last 10000 rows

        print(f"Next model update in {(RETRAIN_INTERVAL - (time.time() - last_retrain_time)) / 60:.2f} minutes")
        time.sleep(60)  # Wait for 60 seconds before the next iteration

if __name__ == "__main__":
    main()


