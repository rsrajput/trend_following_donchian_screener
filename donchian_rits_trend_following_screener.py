#!/usr/bin/env python3
"""
donchian_rits_trend_following_screener.py

Trend-following screener implementing:
- Trend: SMA50 > SMA200, close > SMA50
- Momentum: MACD histogram > 0, Awesome Oscillator (AO) > 0
- Breakout: close > prior 50-day high (Donchian breakout)
- Exit/Stop: 4-day Donchian low
- ADX filter: ADX(14) >= 20 (optional, default ON)
- Volume confirmation (optional)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import ADXIndicator
import argparse
import os
import warnings

# Silence future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------
# Utility helpers
# ----------------------------
def sma(series: pd.Series, window: int):
    return series.rolling(window=window, min_periods=1).mean()

def ema(series: pd.Series, window: int):
    return series.ewm(span=window, adjust=False).mean()

def macd_hist(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - macd_signal
    return macd, macd_signal, hist

def awesome_oscillator(high: pd.Series, low: pd.Series, short=5, long=34):
    median_price = (high + low) / 2
    ao = median_price.rolling(window=short).mean() - median_price.rolling(window=long).mean()
    return ao

def donchian_high(series_high: pd.Series, window: int):
    return series_high.rolling(window=window, min_periods=1).max()

def donchian_low(series_low: pd.Series, window: int):
    return series_low.rolling(window=window, min_periods=1).min()

def as_float(x):
    """Convert any pandas/numpy object to plain float safely."""
    if isinstance(x, (pd.Series, pd.Index, np.ndarray, list, tuple)):
        if len(x) == 0:
            return np.nan
        x = x[0]
    try:
        return float(x)
    except Exception:
        return np.nan

# ----------------------------
# Indicator computation
# ----------------------------
def compute_indicators(df: pd.DataFrame):
    df = df.copy()
    df["SMA20"] = sma(df["Close"], 20)
    df["SMA50"] = sma(df["Close"], 50)
    df["SMA200"] = sma(df["Close"], 200)

    macd, macd_signal, macd_hist_series = macd_hist(df["Close"])
    df["MACD_HIST"] = macd_hist_series
    df["AO"] = awesome_oscillator(df["High"], df["Low"])
    df["DONCHIAN_HIGH_50"] = donchian_high(df["High"], 50)
    df["DONCHIAN_LOW_4"] = donchian_low(df["Low"], 4)

    try:
        adx_ind = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14, fillna=True)
        df["ADX_14"] = adx_ind.adx()
    except Exception:
        df["ADX_14"] = np.nan

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(window=14, min_periods=1).mean()

    return df

# ----------------------------
# Signal logic
# ----------------------------
def is_trending_long(df: pd.DataFrame, adx_threshold=20, require_volume=False, volume_mult=1.5):
    if df is None or len(df) < 60:
        return False, {"error": "insufficient_data"}

    today = df.iloc[-1]
    prev_50_high = float(df["DONCHIAN_HIGH_50"].iloc[-2]) if len(df) > 1 else np.nan

    close = as_float(today["Close"])
    sma50 = as_float(today["SMA50"])
    sma200 = as_float(today["SMA200"])
    macd_hist_val = as_float(today["MACD_HIST"])
    ao_val = as_float(today["AO"])
    adx_val = as_float(today.get("ADX_14", np.nan))

    cond_trend = (sma50 > sma200) and (close > sma50)
    cond_momentum = (macd_hist_val > 0) and (ao_val > 0)
    cond_breakout = close > prev_50_high if not np.isnan(prev_50_high) else False
    cond_adx = (adx_val >= adx_threshold) if not np.isnan(adx_val) else True

    cond_volume = True
    if require_volume and "Volume" in df.columns:
        avg_vol_20 = float(df["Volume"].tail(20).mean())
        cond_volume = today["Volume"] >= (avg_vol_20 * volume_mult)

    ok = all([cond_trend, cond_momentum, cond_breakout, cond_adx, cond_volume])
    reasons = {
        "trend": cond_trend,
        "momentum": cond_momentum,
        "breakout": cond_breakout,
        "adx": cond_adx,
        "volume": cond_volume,
    }

    return ok, reasons

# ----------------------------
# Screener logic
# ----------------------------
def screen_tickers(tickers, period="2y", interval="1d", adx_threshold=20, require_volume=False):
    results = []
    trending_tickers = []

    for t in tickers:
        try:
            df = yf.download(t, period=period, interval=interval, progress=False, auto_adjust=False, threads=False)
            if df.empty:
                print(f"[{t}] no data, skipping.")
                continue

            df = compute_indicators(df)
            ok, reasons = is_trending_long(df, adx_threshold=adx_threshold, require_volume=require_volume)

            if ok:
                print(f"{t}: ✅ Trending long!")
                trending_tickers.append(t)
                latest = df.iloc[-1]
                prev_50_high = df["DONCHIAN_HIGH_50"].shift(1).iloc[-1]
                stop_4d = df["DONCHIAN_LOW_4"].iloc[-1]
                atr = df["ATR_14"].iloc[-1]

                results.append({
                    "ticker": t,
                    "date": df.index[-1].strftime("%Y-%m-%d"),
                    "close": as_float(latest["Close"]),
                    "prev_50_high": as_float(prev_50_high),
                    "stop_4d": as_float(stop_4d),
                    "SMA50": as_float(latest["SMA50"]),
                    "SMA200": as_float(latest["SMA200"]),
                    "MACD_HIST": as_float(latest["MACD_HIST"]),
                    "AO": as_float(latest["AO"]),
                    "ADX_14": as_float(latest["ADX_14"]),
                    "ATR_14": as_float(atr),
                })
            else:
                fails = [k for k, v in reasons.items() if not v]
                print(f"{t}: ❌ Not trending. Fails on {fails}")

        except Exception as e:
            print(f"[{t}] error: {e}")

    df_results = pd.DataFrame(results)
    return df_results, trending_tickers

# ----------------------------
# CLI entry point
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Trend-following screener")
    parser.add_argument("--tickers-file", help="File with tickers (one per line).", required=False)
    parser.add_argument("--period", default="2y", help="yfinance period (2y, 1y, 6mo, etc.)")
    parser.add_argument("--adx", type=float, default=20.0, help="ADX threshold (default 20)")
    parser.add_argument("--volume", action="store_true", help="Require volume confirmation")
    parser.add_argument("--out", default="screener_results.csv", help="CSV output filename")
    args = parser.parse_args()

    tickers = []
    if args.tickers_file:
        if not os.path.exists(args.tickers_file):
            print("Tickers file not found.")
            return
        with open(args.tickers_file, "r") as fh:
            for line in fh:
                s = line.strip()
                if s:
                    tickers.append(s)

    if not tickers:
        tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

    print(f"\nScreening {len(tickers)} tickers for breakout/trend (period={args.period}) ...\n")

    df_res, trending = screen_tickers(tickers, period=args.period, adx_threshold=args.adx, require_volume=args.volume)

    if df_res.empty:
        print("\nNo matches found.")
    else:
        df_res = df_res.sort_values(by=["date", "ticker"]).reset_index(drop=True)
        df_res.to_csv(args.out, index=False)
        print(f"\nFound {len(df_res)} matches. Saved to {args.out}")
        print(df_res.to_string(index=False))

    # ✅ Summary section
    print("\n" + "="*60)
    print("📈 SUMMARY: TRENDING LONG STOCKS")
    print("="*60)
    if trending:
        for sym in trending:
            print(f"✅ {sym}")
    else:
        print("No stocks are currently in a confirmed uptrend.")
    print("="*60)

if __name__ == "__main__":
    main()
