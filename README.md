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

In Colab, re-run **Step 2** (which runs `git pull`) to pick up the latest changes.
Note: the notebook cells themselves are cached by Colab — reopen the notebook from GitHub
if cell-level changes need to take effect.

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data fetching | `yfinance`, `stooq` (direct CSV for JGB) |
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

## How to Interpret Results

### 02 — Correlation & Statistical Analysis

#### Pearson Correlation (log returns)
| Range | Interpretation |
|-------|----------------|
| \|r\| > 0.7 | Strong linear relationship |
| 0.3 < \|r\| < 0.7 | Moderate — worth investigating |
| \|r\| < 0.3 | Weak — likely not useful alone |

> Computed on **log returns**, not price levels, to avoid spurious correlation from shared trends.

#### Lag Correlation (CCF)
- Peak at **negative lag** (e.g. lag −1): the indicator leads Nikkei → potentially actionable for prediction
- Peak at **positive lag**: the indicator lags Nikkei → reactive, not useful for forecasting
- Practical focus: lag −1 and −2 (data available before next Japan open)

#### Rolling Correlation (60-day window)
- Stable over time → reliable feature
- Correlation flipping sign during crises → use with caution; consider regime-aware models

#### Mutual Information
- 0 = no dependency (linear or non-linear)
- Higher = stronger dependency
- If MI >> |Pearson r|: non-linear relationship exists that Ridge will miss (use XGBoost)

#### OLS Regression + VIF
| Metric | Threshold | Action |
|--------|-----------|--------|
| p-value | < 0.05 | Statistically significant coefficient |
| R² | higher = better | Fraction of variance explained |
| VIF | > 10 | Severe multicollinearity — drop one of the correlated features |
| VIF | 5–10 | Moderate concern — monitor |

#### Granger Causality
| p-value | Interpretation |
|---------|----------------|
| < 0.05 | Past values of X improve prediction of Nikkei (beyond Nikkei's own past) |
| < 0.01 | Strong predictive evidence |
| ≥ 0.05 | No significant Granger causality |

> Focus on **lag 1** results — these are most actionable for next-day prediction.

#### Cointegration
- Engle–Granger p < 0.05: the pair shares a long-run equilibrium → mean-reversion strategies may apply
- Johansen rank > 0: at least one cointegrating relationship among the multivariate set

#### VAR Impulse Response
- Positive IRF at lag 1: a shock in X leads to a same-direction move in Nikkei next day
- Sign and magnitude decay after a few lags → short-lived vs persistent effects

#### PCA
- **PC1** typically captures market beta (all assets rising/falling together)
- **PC2** often captures risk-off dynamics (equities down, bonds/JPY up)
- Cumulative explained variance > 80% with few PCs → feature set is highly redundant

---

### 03 — Technical Indicators

#### RSI (Relative Strength Index, 0–100)
| Level | Signal |
|-------|--------|
| > 70 | Overbought — potential downward reversal |
| 50–70 | Bullish momentum |
| 30–50 | Bearish momentum |
| < 30 | Oversold — potential upward reversal |

#### Bollinger Bands
| Metric | Interpretation |
|--------|----------------|
| %B > 1.0 | Price above upper band — stretched to the upside |
| %B = 0.5 | Price at mid band (20-day SMA) |
| %B < 0.0 | Price below lower band — stretched to the downside |
| BB Width increasing | Volatility expanding (often precedes a large move) |
| BB Width contracting | Volatility compressing (squeeze — breakout may follow) |

#### MACD Histogram
| Value | Interpretation |
|-------|----------------|
| Positive and growing | Bullish momentum strengthening |
| Positive and shrinking | Bullish momentum fading |
| Negative and shrinking (toward 0) | Bearish momentum fading — potential reversal |
| Negative and growing (more negative) | Bearish momentum strengthening |

#### Stochastic Oscillator (%K, 0–100)
| Level | Signal |
|-------|--------|
| %K > 80 | Overbought |
| %K < 20 | Oversold |
| %K crosses above %D | Bullish signal |
| %K crosses below %D | Bearish signal |

#### ATR % (ATR / Close × 100)
- Low ATR% (< 0.5%): quiet, low-volatility regime
- High ATR% (> 1.5%): elevated volatility — model predictions are less reliable; position sizing should be reduced

#### Volume Ratio (Volume / 20-day avg)
| Ratio | Interpretation |
|-------|----------------|
| > 1.5 | High conviction — move more likely to continue |
| 0.8–1.2 | Normal |
| < 0.5 | Low participation — move may lack follow-through |

---

### 04 — Model Evaluation

#### RMSE
- Measured in log-return units (e.g. 0.008 ≈ 0.8% daily error)
- Compare against the **naive baseline RMSE** (always predicting 0): if your model's RMSE is similar, it adds no value
- Lower is better, but RMSE alone does not indicate profitability

#### Directional Accuracy
| Value | Interpretation |
|-------|----------------|
| 50% | No better than random |
| 52–54% | Modest edge — potentially profitable with low transaction costs |
| > 55% | Strong signal |

> In practice, 52%+ sustained over the test period is considered meaningful for daily equity prediction.

#### Sharpe Ratio (annualised, long/short signal strategy)
| Value | Interpretation |
|-------|----------------|
| < 0 | Strategy loses money — model's direction calls are counterproductive |
| 0–0.5 | Marginally positive |
| 0.5–1.0 | Acceptable |
| > 1.0 | Good risk-adjusted return |
| > 2.0 | Excellent (uncommon for pure technical signals) |

> These Sharpe figures assume **no transaction costs**. Add at least 0.05–0.1% per round-trip for a realistic estimate.

---

## Evaluation Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| RMSE | √mean((y − ŷ)²) | Scale: log-return units |
| Directional accuracy | mean(sign(ŷ) == sign(y)) | 50% = random baseline |
| Sharpe ratio | mean(r) / std(r) × √252 | Annualised; r = signal × actual return |
