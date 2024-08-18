# Discendo Progressio

*Latin for "Learning Progress"*

## Adaptive Cryptocurrency Trading Bot

Discendo Progressio is an advanced cryptocurrency trading bot that utilizes machine learning algorithms to make informed trading decisions. The bot employs an ensemble of Random Forest, XGBoost, and LightGBM models to analyze market trends and execute trades automatically.

### Features

- **Multi-Model Ensemble**: Combines predictions from Random Forest, XGBoost, and LightGBM for robust decision-making.
- **Automatic Retraining**: Models are periodically retrained to adapt to changing market conditions.
- **Technical Indicators**: Utilizes various technical indicators including SMA, RSI, MACD, ATR, and Bollinger Bands.
- **Historical Data Management**: Fetches and maintains historical price data for analysis.
- **Trade Logging**: Records all trades for future analysis and auditing.
- **Balance Management**: Keeps track of USD and cryptocurrency balances.
- **Configurable Parameters**: Easily adjust trading amounts, fees, and retraining intervals.

### Prerequisites

- Python 3.7+
- ccxt
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- yfinance

