from utils import logger
from utils.decorator import timing
from utils.constants import *
from data import get_data, feature_engineering
from models import predict, train

LOGGER = logger.setup_logger(__name__)


@timing
def run(stock, start_date):
    equity_data = get_data.read_yahoo_data(stock, start_date)
    etf_data = get_data.read_yahoo_data(ETF, start_date)
    print(equity_data.loc['2020-08-20'])
    feature_engineering.run(equity_data)
    return equity_data, etf_data


if __name__ == '__main__':
    stock = 'TSLA'
    start_date = [2020, 1, 6]
    run(stock, start_date)