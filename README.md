# Nikkei 225 Closing Price Prediction

A repository for predicting the Nikkei 225 closing price using macro/micro fundamental indicators and technical analysis.

---

## Analysis Overview

### Fundamental Analysis

**Macro Indicators**
- Nikkei 225 Futures
- FX rates (USD/JPY, EUR/JPY, etc.)
- Commodities (crude oil, gold, etc.)
- US indices (S&P 500, NASDAQ, Dow Jones)
- Bond yields (US Treasuries, JGB)

**Micro Indicators (major Nikkei 225 constituents)**
- SoftBank Group (9984)
- Fast Retailing (9983)
- Other high-weight Nikkei 225 stocks

### Technical Analysis

- Moving averages (SMA / EMA)
- Bollinger Bands
- RSI / MACD
- Volume analysis

### Statistical & Correlation Analysis

| Method | Purpose |
|--------|---------|
| Pearson correlation | Linear relationship between each indicator and closing price |
| Lag correlation | Identifying leading indicators (e.g. US market impact on next-day Nikkei) |
| Multiple regression (OLS) | Estimating effect size of multiple indicators simultaneously |
| VIF analysis | Diagnosing multicollinearity among predictors |
| Granger causality test | Testing whether indicators statistically predict Nikkei 225 |
| VAR model | Modeling interdependencies among multiple time series |
| PCA | Dimensionality reduction and feature selection |

### Prediction Models

| Model | Notes |
|-------|-------|
| ARIMA / SARIMA | Autoregressive and seasonal time series modeling |
| Ridge / Lasso regression | Linear baseline with regularization-based feature selection |
| Random Forest / XGBoost / LightGBM | Non-linear relationships and feature importance |
| LSTM | Deep learning for sequential patterns |

---

## Directory Structure

```
Nikkei_Analysis/
├── data/                      # Managed via Google Drive (not in git)
│   ├── raw/                   # Raw data as fetched (read-only)
│   └── processed/             # Cleaned and feature-engineered data
├── notebooks/                 # Exploratory analysis (.ipynb)
│   ├── 01_data_exploration.ipynb
│   ├── 02_correlation.ipynb
│   ├── 03_technical.ipynb
│   └── 04_modeling.ipynb
├── src/                       # Reusable modules
│   ├── data/                  # Data fetching and loading
│   ├── features/              # Feature engineering (technical indicators, etc.)
│   ├── models/                # Model definitions, training, inference
│   └── visualization/         # Plotting utilities
├── output/                    # Analysis outputs
│   ├── figures/               # Charts and plots
│   ├── models/                # Saved model files (not in git)
│   └── reports/               # Evaluation reports and prediction CSVs
├── requirements.txt
└── README.md
```

---

## Workflow & Operations

### Development Flow

```
Local (edit notebooks / src)
    └─► git push to GitHub
            └─► Google Colab (pull & run)
```

### Running on Google Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. `File` → `Open notebook` → `GitHub` tab
3. Enter `https://github.com/Takumi-Itokawa-Finance/Nikkei_Analysis`
4. Select the target notebook and open it
5. Run **Cell 0** (environment setup) — this will:
   - `git clone` the repository to `/content/Nikkei_Analysis/`
   - Mount Google Drive and link `data/` for persistence
   - Install required libraries

### Data Persistence

`data/` is excluded from git and stored in Google Drive:

```
MyDrive/
└── Nikkei_Analysis/
    └── data/
        ├── raw/
        └── processed/
```

The setup cell in each notebook creates this structure automatically on first run.

### Updating Code

```bash
# After editing locally
git add .
git commit -m "your message"
git push origin main
```

In Colab, re-run Cell 0 (`git clone`) or use `!git pull` to get the latest changes.

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data fetching | `yfinance`, `pandas-datareader` |
| Data processing | `pandas`, `numpy` |
| Technical indicators | `pandas-ta` |
| Statistical analysis | `statsmodels`, `scipy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Machine learning | `scikit-learn`, `xgboost`, `lightgbm` |
| Time series models | `statsmodels` (ARIMA / VAR) |
| Deep learning | `tensorflow` / `pytorch` |
| Feature importance | `shap` |

---

## Setup (local)

```bash
git clone https://github.com/Takumi-Itokawa-Finance/Nikkei_Analysis.git
cd Nikkei_Analysis
pip install -r requirements.txt
```

---

## Analysis Pipeline

```
Data fetching
    └─► Preprocessing & feature engineering
            └─► Correlation & regression analysis (OLS / Granger / VAR)
                    └─► Model training (linear / boosting / LSTM)
                            └─► Evaluation & backtesting → output/
```

---

## Evaluation Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- Directional accuracy (correct next-day up/down prediction rate)
