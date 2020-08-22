from datetime import datetime, timedelta

EQUITY_LOOKBACK = 5
ETF_LOOKBACK = 90
PREDICTION_WINDOW = 20
YEARS_LOOKBACK = 8

ETF = 'SPY'
CURRENT_MONTH = datetime.now().month
CURRENT_DAY = datetime.now().day
CURRENT_YEAR = datetime.now().year
CURRENT_DATETIME = datetime(CURRENT_YEAR, CURRENT_MONTH, CURRENT_DAY)
START_DATETIME = CURRENT_DATETIME - timedelta(YEARS_LOOKBACK * 365)

SCORING = 'average_precision'
MODEL_VARIABLES = ['Slow_Stochastic_%K', 'Fast_Stochastic_%D', 'Williams_%R', 'Raw_Price_Difference', 'Price_ROC',
                   'RSI', 'ATR', 'Average_Price_Volatility', 'Disparity_Index', 'MACD', 'On_Balance_Volume', 'Label']
