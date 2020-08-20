import pandas_datareader.data as web
from utils.constants import *


def read_yahoo_data(ticker, start_date):
    data = web.get_data_yahoo([ticker],
                              start=datetime(start_date[0], start_date[1], start_date[2]),
                              end=CURRENT_DATETIME)
    return data
