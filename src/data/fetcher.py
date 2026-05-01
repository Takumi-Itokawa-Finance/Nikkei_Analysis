"""
Data fetching module for Nikkei 225 prediction project.

All functions return a DataFrame of Close prices indexed by date.
Use fetch_all() to get the full merged feature matrix.
"""

from datetime import datetime, timedelta

import pandas as pd
import pandas_datareader.data as web
import yfinance as yf


# ---------------------------------------------------------------------------
# Ticker definitions
# ---------------------------------------------------------------------------

TICKERS_INDEX = {
    "Nikkei225_Futures": "NKD=F",
    "SP500":             "^GSPC",
    "NASDAQ":            "^IXIC",
    "DowJones":          "^DJI",
    "VIX":               "^VIX",
}

TICKERS_FX = {
    "USD_JPY": "JPY=X",
    "EUR_JPY": "EURJPY=X",
    "GBP_JPY": "GBPJPY=X",
    "CNY_JPY": "CNYJPY=X",
}

TICKERS_COMMODITY = {
    "WTI_Crude": "CL=F",
    "Gold":      "GC=F",
    "Silver":    "SI=F",
    "Copper":    "HG=F",
}

TICKERS_US_BOND = {
    "US_2Y":  "^IRX",
    "US_10Y": "^TNX",
    "US_30Y": "^TYX",
}

TICKERS_MICRO = {
    "FastRetailing":  "9983.T",
    "SoftBank":       "9984.T",
    "Tokyo_Electron": "8035.T",
    "KDDI":           "9433.T",
    "ShinEtsu_Chem":  "4063.T",
}

# stooq ticker for JGB 10Y yield
TICKER_JGB_10Y = "10lj.b"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _yf_close(tickers: dict, period: str, interval: str) -> pd.DataFrame:
    """Download Close prices for a dict of {name: ticker} via yfinance."""
    frames = {}
    for name, ticker in tickers.items():
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if len(df) > 0:
            frames[name] = df["Close"].squeeze()
        else:
            print(f"[WARN] No data returned for {name} ({ticker})")
    return pd.DataFrame(frames)


def _date_range(period: str):
    """Convert yfinance-style period string to (start, end) datetimes."""
    end = datetime.today()
    days = {"1y": 365, "2y": 730, "3y": 1095, "5y": 1825, "10y": 3650}
    delta = days.get(period, 730)
    return end - timedelta(days=delta), end


# ---------------------------------------------------------------------------
# Public fetch functions
# ---------------------------------------------------------------------------

def fetch_nikkei(period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch Nikkei 225 OHLCV data."""
    df = yf.download("^N225", period=period, interval=interval, progress=False)
    if len(df) == 0:
        raise ValueError("Failed to fetch Nikkei 225 data.")
    return df


def fetch_macro_indices(period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch equity index and futures Close prices."""
    return _yf_close(TICKERS_INDEX, period, interval)


def fetch_fx(period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch FX rate Close prices (all vs JPY)."""
    return _yf_close(TICKERS_FX, period, interval)


def fetch_commodities(period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch commodity futures Close prices."""
    return _yf_close(TICKERS_COMMODITY, period, interval)


def fetch_us_bonds(period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch US Treasury yield Close prices."""
    return _yf_close(TICKERS_US_BOND, period, interval)


def fetch_jgb(period: str = "2y") -> pd.DataFrame:
    """
    Fetch JGB 10Y yield from stooq via pandas-datareader.
    Returns a single-column DataFrame named 'JGB_10Y'.
    """
    start, end = _date_range(period)
    df = web.DataReader(TICKER_JGB_10Y, "stooq", start=start, end=end)
    df = df.sort_index()
    if len(df) == 0:
        print("[WARN] No JGB data returned from stooq.")
        return pd.DataFrame()
    return df[["Close"]].rename(columns={"Close": "JGB_10Y"})


def fetch_micro(period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch Close prices of major Nikkei 225 constituent stocks."""
    return _yf_close(TICKERS_MICRO, period, interval)


def fetch_all(period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch and merge all Close price series into one DataFrame.

    Returns
    -------
    pd.DataFrame
        Index: trading dates
        Columns: Nikkei225 + all macro/micro indicators
    """
    nikkei = fetch_nikkei(period, interval)
    close_df = pd.DataFrame({"Nikkei225": nikkei["Close"].squeeze()})

    for fetcher in [
        lambda: fetch_macro_indices(period, interval),
        lambda: fetch_fx(period, interval),
        lambda: fetch_commodities(period, interval),
        lambda: fetch_us_bonds(period, interval),
        lambda: fetch_micro(period, interval),
    ]:
        df = fetcher()
        if len(df) > 0:
            close_df = close_df.join(df, how="left")

    # JGB uses date range instead of period string
    df_jgb = fetch_jgb(period)
    if len(df_jgb) > 0:
        close_df = close_df.join(df_jgb, how="left")

    return close_df
