
# 📈 Stock-Voyant

AI-Based Stock Price & Volatility Predictor

This project is a hybrid machine learning and deep learning web application designed to **predict future stock prices and volatility**. It uses historical stock data to train a **hybrid model** that combines LSTM (Long Short-Term Memory) neural networks and Random Forest regressors for more accurate forecasting.

---

## 🚀 Features

- Upload your own **stock price CSV** file.
- Predict future **stock prices** and **volatility** for a specified number of days.
- Combines deep learning (LSTM) and traditional machine learning (Random Forest).
- Scaled, sequence-based time series modeling.
- Interactive, user-friendly **web interface** built with Python full stack.

---

## 🧠 Technologies Used

- **Python**
- **Pandas**, **NumPy** – Data manipulation
- **TensorFlow / Keras** – LSTM deep learning model
- **Scikit-learn** – Random Forest and data scaling
- **Flask** – Web framework for handling file uploads and predictions
- **HTML/CSS**-Front end part 

---

## 📊 Data Source

> ✅ **Only stock data downloaded from the [NSE India website](https://www.nseindia.com/)** is supported.

Please ensure your file follows the NSE format with the following columns:

- `Date` (in dd-mm-yyyy or yyyy-mm-dd format)
- `OPEN`
- `HIGH`
- `LOW`

