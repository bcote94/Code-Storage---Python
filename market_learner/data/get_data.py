import pandas_datareader.data as web
from utils import logger
from utils.constants import *

LOGGER = logger.setup_logger(__name__)


def read_yahoo_data(ticker, start_date):
    if type(start_date) != list or len(start_date) != 3:
        raise Exception("start_date must be passed in as a list in the format [year, month, day]")
    if not all(isinstance(val, int) for val in start_date):
        raise Exception("year/month/date must all be integer valued.")

    start_datetime = datetime(start_date[0], start_date[1], start_date[2])
    if start_datetime.date() > CURRENT_DATETIME:
        raise Exception("Start Date {} must be prior to the Current Time {}.".format(start_datetime, CURRENT_DATETIME))

    LOGGER.debug("Gathering daily stock data for {} from {} to {}".format(ticker, start_datetime, CURRENT_DATETIME))
    return web.get_data_yahoo([ticker], start=start_datetime, end=CURRENT_DATETIME).drop(['Adj Close'], axis=1)
