### 📊 BharatMarketAI

#### BharatMarketAI is an end-to-end AI-powered Indian stock market research platform that combines machine learning, walk-forward backtesting, no-lookahead trading logic, and explainable signals.

##### The project is designed as a research & portfolio demo, not financial advice.

#### 🚀 Key Highlights

* ✅ Indian stock market focus (NIFTY stocks – Yahoo Finance)

* ✅ Walk-forward model training (realistic evaluation)

* ✅ Strict no-lookahead bias execution

* ✅ LightGBM-based ML model

* ✅ Top-K portfolio backtesting

* ✅ Streamlit interactive dashboard

* ✅ Explainable BUY / HOLD / SELL advisor

* ✅ News & geopolitical feature support (optional module)
-----

### 🧠 System Architecture
* Market Data (OHLCV)
        ↓
* Feature Engineering (Technical + Volatility + Trend)
        ↓
* Walk-Forward ML Training (LightGBM)
        ↓
* Out-of-Sample Predictions (OOS)
        ↓
* No-Lookahead Backtesting
        ↓
* Portfolio Strategy (Top-K)
        ↓
* Streamlit Dashboard + Advisor

#### 📌 Why this project is different

Most “AI trading” projects fail because of:

random train/test splits ❌

lookahead bias ❌

unrealistic execution prices ❌

BharatMarketAI explicitly avoids these problems by:

using rolling walk-forward windows

generating pure out-of-sample predictions

entering trades at next-day open

separating research backtests from UI logic
-----

### 🧪 Backtesting Methodology (Important)

Signal date: day t (model sees data up to close of t)

Entry: next trading day open (t+1)

Exit:

fixed holding horizon (e.g. 5 days) OR

ATR-based stop OR

model confidence drop

Portfolio: Top-K signals, equal-weighted

Costs: round-trip transaction cost applied

Metrics: CAGR, Sharpe, Max Drawdown, Total Return

This makes the results realistic and reproducible.
-----

### 📊 Dashboard Features
##### 🏠 Overview

Equity curves (OOS / no-lookahead)

Monthly returns heatmap

Drawdown & performance metrics

Symbol contribution analysis
------

#### 🎯 Signals

Latest ML probabilities

Sort & filter by confidence

Downloadable signal table
-----

#### 🧪 Backtest Lab

Interactive strategy parameters

Threshold tuning

Top-K selection

Live equity curve visualization
------

#### 🧠 Advisor

Single-stock analysis

BUY / HOLD / SELL decision

Confidence score (p_up)

Technical explanation (RSI, Trend, ATR)

Optional news & geopolitics context
------

#### 📰 News (Optional)

Daily aggregated sentiment

Geopolitical risk indicators

Recent headlines
------

### 🛠️ Tech Stack

Language: Python 3.11

ML: LightGBM, scikit-learn

Data: Pandas, NumPy, yfinance

Indicators: ta-lib (technical analysis)

UI: Streamlit

Backtesting: Custom engine (no-lookahead)

Deployment: Streamlit Cloud / Local

📂 Project Structure
BharatMarketAI/
├── app.py                  # Streamlit UI
├── src/
│   ├── data/               # Market & news data loaders
│   ├── features/           # Feature engineering
│   ├── models/             # ML training & walk-forward logic
│   ├── backtest/           # No-lookahead backtesting engine
│   └── ui/                 # Advisor logic
├── reports/                # Saved backtests & predictions
├── data_cache/             # Cached market/news data
├── requirements.txt
├── runtime.txt
└── README.md

▶️ How to Run Locally
# create virtual environment
python -m venv venv
source venv/Scripts/activate   # Windows
# source venv/bin/activate     # Linux/Mac

# install dependencies
pip install -r requirements.txt

# generate market data
python -m src.data.market

# build features
python -m src.features.build_features

# walk-forward training
python -m src.models.walk_forward

# run no-lookahead top-k backtest
python -m src.backtest.oos_backtest_topk_nolookahead_trades

# launch dashboard
streamlit run app.py

### ☁️ Deployment

This project is deployment-ready.

Recommended:

Streamlit Community Cloud

Steps:

Push repo to GitHub

Connect Streamlit Cloud

Set entry file:

app.py


### Deploy 🚀

#### ⚠️ Disclaimer

**This project is for educational and research purposes only.
It does not constitute financial advice.
Stock market investments involve risk.
Past performance does not guarantee future results.**

#### 👤 Author

##### Avdhut Shinde
##### AI / ML Engineer
##### Focus: Applied ML, Trading Systems, Data Science

##### ⭐ If you like this project
-----
Star ⭐ the repo

Fork 🍴 and experiment
