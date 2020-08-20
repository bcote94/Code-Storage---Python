from datetime import datetime

ETF = 'SPY'
CURRENT_MONTH = datetime.now().month
CURRENT_DAY = datetime.now().day
CURRENT_YEAR = datetime.now().year
CURRENT_DATETIME = datetime(CURRENT_YEAR, CURRENT_MONTH, CURRENT_DAY)

EQUITY_LOOKBACK = 5
ETF_LOOKBACK = 90
PREDICTION_WINDOW = 20

SCORING = 'average_precision'