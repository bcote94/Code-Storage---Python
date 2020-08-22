import pandas_datareader.data as web
from utils import logger
from utils.constants import *

LOGGER = logger.setup_logger(__name__)


def read_yahoo_data(ticker):
    LOGGER.debug("Gathering daily stock data for {} from {} to {}".format(ticker, START_DATETIME, CURRENT_DATETIME))
    return web.get_data_yahoo([ticker], start=START_DATETIME, end=CURRENT_DATETIME).drop(['Adj Close'], axis=1)
