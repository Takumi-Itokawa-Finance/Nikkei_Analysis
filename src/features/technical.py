"""
Technical indicator computation for Nikkei 225.

compute_indicators(df) : add all indicators to the OHLCV DataFrame
build_features(df)     : return model-ready feature matrix (no look-ahead)
"""

import numpy as np
import pandas as pd
import pandas_ta as ta


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _col(df: pd.DataFrame, prefix: str) -> str:
    """Return the first column whose name starts with *prefix*."""
    match = [c for c in df.columns if c.startswith(prefix)]
    if not match:
        raise KeyError(f"No column starting with '{prefix}' in {df.columns.tolist()}")
    return match[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators and append them to *df*.

    Parameters
    ----------
    df : DataFrame with columns [Open, High, Low, Close, Volume]

    Returns
    -------
    DataFrame with original columns + indicator columns.
    """
    df = df.copy()

    # -- Moving averages ------------------------------------------------------
    for p in [5, 10, 20, 50, 75, 200]:
        df[f'SMA_{p}'] = ta.sma(df['Close'], length=p)
    for p in [5, 20, 50]:
        df[f'EMA_{p}'] = ta.ema(df['Close'], length=p)

    # -- Bollinger Bands -------------------------------------------------------
    bb = ta.bbands(df['Close'], length=20, std=2)
    df['BB_lower'] = bb[_col(bb, 'BBL')]
    df['BB_mid']   = bb[_col(bb, 'BBM')]
    df['BB_upper'] = bb[_col(bb, 'BBU')]
    df['BB_width'] = bb[_col(bb, 'BBB')]
    df['BB_pct']   = bb[_col(bb, 'BBP')]

    # -- RSI -------------------------------------------------------------------
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    df['RSI_9']  = ta.rsi(df['Close'], length=9)

    # -- MACD ------------------------------------------------------------------
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD']        = macd[_col(macd, 'MACD_')]
    df['MACD_signal'] = macd[_col(macd, 'MACDs_')]
    df['MACD_hist']   = macd[_col(macd, 'MACDh_')]

    # -- Volume ----------------------------------------------------------------
    df['Volume_SMA20'] = ta.sma(df['Volume'], length=20)
    df['Volume_ratio'] = df['Volume'] / df['Volume_SMA20']
    df['OBV']          = ta.obv(df['Close'], df['Volume'])

    # -- ATR -------------------------------------------------------------------
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ATR_pct'] = df['ATR_14'] / df['Close'] * 100

    # -- Stochastic ------------------------------------------------------------
    stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
    df['Stoch_K'] = stoch[_col(stoch, 'STOCHk_')]
    df['Stoch_D'] = stoch[_col(stoch, 'STOCHd_')]

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build model-ready feature matrix from a DataFrame produced by
    compute_indicators().  No look-ahead bias: all features are derived
    from time-t data; target is time t+1 log return.

    Returns
    -------
    DataFrame with feature columns + 'target_return' + 'target_close'.
    Rows with NaN (indicator warm-up) are dropped.
    """
    feat = pd.DataFrame(index=df.index)

    # Target
    feat['target_return'] = np.log(df['Close'] / df['Close'].shift(1)).shift(-1)
    feat['target_close']  = df['Close'].shift(-1)

    # Trend: price relative to moving averages
    for p in [5, 10, 20, 50, 75, 200]:
        feat[f'close_vs_sma{p}'] = df['Close'] / df[f'SMA_{p}'] - 1
    for p in [5, 20, 50]:
        feat[f'close_vs_ema{p}'] = df['Close'] / df[f'EMA_{p}'] - 1

    # MA crossovers
    feat['sma5_vs_sma20']   = df['SMA_5']  / df['SMA_20']  - 1
    feat['sma20_vs_sma50']  = df['SMA_20'] / df['SMA_50']  - 1
    feat['sma50_vs_sma200'] = df['SMA_50'] / df['SMA_200'] - 1

    # Volatility
    feat['BB_pct']   = df['BB_pct']
    feat['BB_width'] = df['BB_width']
    feat['ATR_pct']  = df['ATR_pct']

    # Momentum
    feat['RSI_14']    = df['RSI_14']
    feat['RSI_9']     = df['RSI_9']
    feat['MACD_hist'] = df['MACD_hist']
    feat['Stoch_K']   = df['Stoch_K']
    feat['Stoch_D']   = df['Stoch_D']
    feat['Stoch_KD']  = df['Stoch_K'] - df['Stoch_D']

    # Past returns
    for d in [1, 2, 3, 5, 10]:
        feat[f'return_{d}d'] = np.log(df['Close'] / df['Close'].shift(d))

    # Volume
    feat['volume_ratio'] = df['Volume_ratio']
    feat['obv_change']   = df['OBV'].pct_change()

    return feat.dropna()
