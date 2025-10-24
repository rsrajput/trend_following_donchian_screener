# trend_following_donchian_screener

Requirements (install once):
pip install yfinance pandas numpy ta


To use this python program:
1. (using a text file for stock symbols)

  python trend_screener.py --tickers-file tickers.txt

  tickers.txt can be any text file containing tickers like RELIANCE.NS

2. directly from commandline arguments:

  python trend_screener.py --tickers RELIANCE.NS TCS.NS INFY.NS