# Rits Donchian Trend Following Screener

A **Python-based stock trend screener** that helps identify **strongly trending stocks** while avoiding **choppy or sideways markets**.  
The system uses **Donchian Channels, Moving Averages (20/50/200), MACD Histogram**, and **Awesome Oscillator** to detect high-probability uptrends.

---

## Features

- **Trend Detection Logic**
  - Uses Donchian Channels (50-day breakout + 4-day pullback exit).
  - Confirms trend using SMA(20), SMA(50), and SMA(200).
  - Confirms momentum with MACD Histogram and Awesome Oscillator.
  - Filters out choppy / sideways stocks automatically.

- **Flexible Input Options**
  - Read tickers from a text file (e.g., `tickers.txt`).
  - Or provide tickers directly as command-line arguments.

- **Exchange Support**
  - Works for both **NSE** (`.NS`) and **BSE** (`.BO`) stock symbols via Yahoo Finance.

- **Clean Summary Output**
  - Prints detailed status of each stock (Trending / Not Trending).
  - Displays a **final summary list** of all trending stocks at the end.

---

## Installation

Clone or download the repository, then install the dependencies:

```bash
pip install -r requirements.txt


If you don’t have a requirements.txt file, install manually:

```bash
pip install yfinance pandas numpy ta


# Usage
## Option 1: Using a tickers file

Create a text file (e.g., tickers.txt) containing one ticker per line:

RELIANCE.NS
TCS.NS
INFY.NS
HDFCBANK.NS
SBIN.BO

Run the screener:

python trend_screener.py --tickers-file tickers.txt

## Option 2: Using command-line tickers directly

python trend_screener.py --tickers RELIANCE.NS TCS.NS INFY.NS


# Example Output
[RELIANCE.NS] ✅ Trending long!
[TCS.NS] ⛔ Not trending
[INFY.NS] ⛔ Not trending
[HDFCBANK.NS] ✅ Trending long!

======== 📈 SUMMARY ========
Trending Long:
- RELIANCE.NS
- HDFCBANK.NS
============================

Total trending: 2 out of 4


🧠 How It Works

Data Fetching:
Fetches OHLC data using yfinance for each ticker.

Indicators Calculation:

Donchian Channels: 50-day high/low breakout and 4-day pullback.

Moving Averages: SMA(20), SMA(50), SMA(200) trend alignment.

MACD Histogram: Momentum confirmation.

Awesome Oscillator: Cross-verification of momentum.

Signal Logic:
A stock is considered "Trending Long" if:

Price is above SMA(50) and SMA(200).

SMA(50) > SMA(200).

MACD Histogram > 0 and AO > 0.

Price is breaking above the Donchian upper band.

💡 Tips

For daily analysis, schedule this script using Windows Task Scheduler or a simple .bat file.

Try scanning Nifty 50 or Midcap stocks for better liquidity.

Combine results with fundamental screening for higher conviction entries.

🧾 License

MIT License © 2025
You are free to use, modify, and distribute this script for personal or commercial use.

🙌 Credits

Developed by Rituraj, inspired by the Turtle Trading philosophy and modern trend-following principles.
