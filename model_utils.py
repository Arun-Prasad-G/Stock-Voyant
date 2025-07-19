# model_utils.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def load_and_train_model(file_path):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
    data.dropna(subset=['Date'], inplace=True)
    data.sort_values('Date', inplace=True)
    data.set_index('Date', inplace=True)

    for col in ['OPEN', 'HIGH', 'LOW']:
        data[col] = data[col].astype(str).str.replace(',', '', regex=True).astype(float)

    data['return'] = data['HIGH'].pct_change()
    data['volatility'] = data['return'].rolling(window=5).std()
    data['MA_5'] = data['HIGH'].rolling(window=5).mean()
    data['MA_10'] = data['HIGH'].rolling(window=10).mean()
    data.fillna(method='bfill', inplace=True)

    features = ['OPEN', 'LOW', 'MA_5', 'MA_10']
    target_price = 'HIGH'
    target_volatility = 'volatility'

    feature_scaler = MinMaxScaler()
    price_scaler = MinMaxScaler()
    vol_scaler = MinMaxScaler()

    X_scaled = feature_scaler.fit_transform(data[features])
    y_price_scaled = price_scaler.fit_transform(data[[target_price]])
    y_vol_scaled = vol_scaler.fit_transform(data[[target_volatility]])

    sequence_length = 10
    X_seq, y_price_seq, y_vol_seq = create_sequences(X_scaled, y_price_scaled, y_vol_scaled, sequence_length)

    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train = np.hstack((y_price_seq[:split], y_vol_seq[:split]))
    y_test = np.hstack((y_price_seq[split:], y_vol_seq[split:]))

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(32),
        Dense(32, activation='relu'),
        Dense(2)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=80, validation_data=(X_test, y_test), verbose=0)

    rf_price = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_vol = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_price.fit(data[features], data[target_price])
    rf_vol.fit(data[features], data[target_volatility])

    return {
        'model': model,
        'rf_price': rf_price,
        'rf_vol': rf_vol,
        'scalers': (feature_scaler, price_scaler, vol_scaler),
        'X_scaled': X_scaled,
        'features': features,
        'data': data,
        'sequence_length': sequence_length
    }


def create_sequences(X, y1, y2, seq_len):
    Xs, y_prices, y_vols = [], [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        y_prices.append(y1[i])
        y_vols.append(y2[i])
    return np.array(Xs), np.array(y_prices), np.array(y_vols)


def predict_future(model_data, user_start_date, num_days):
    from datetime import timedelta

    future_dates = pd.date_range(start=user_start_date, periods=num_days, freq='D')
    last_seq = model_data['X_scaled'][-model_data['sequence_length']:, :]
    model = model_data['model']
    rf_price = model_data['rf_price']
    rf_vol = model_data['rf_vol']
    feature_scaler, price_scaler, vol_scaler = model_data['scalers']
    data = model_data['data']
    features = model_data['features']

    predictions = []

    for _ in range(num_days):
        lstm_input = last_seq.reshape(1, model_data['sequence_length'], len(features))
        lstm_pred = model.predict(lstm_input, verbose=0)[0]
        price_pred = price_scaler.inverse_transform([[lstm_pred[0]]])[0][0]
        vol_pred = vol_scaler.inverse_transform([[lstm_pred[1]]])[0][0]

        last_features = data[features].iloc[-1:].values
        rf_price_pred = rf_price.predict(last_features)[0]
        rf_vol_pred = rf_vol.predict(last_features)[0]

        final_price = 0.7 * price_pred + 0.3 * rf_price_pred
        final_vol = 0.7 * vol_pred + 0.3 * rf_vol_pred

        predictions.append((round(final_price, 2), round(final_vol, 4)))

        new_scaled = feature_scaler.transform([[final_price, final_price - 20, final_price, final_price]])
        last_seq = np.vstack((last_seq[1:], new_scaled))

    return future_dates, predictions