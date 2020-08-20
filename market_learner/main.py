from utils import logger
from utils.decorator import timing
from utils.constants import *
from market_learner.data_prep import get_data

LOGGER = logger.setup_logger(__name__)
START_DATE = [2020, 1, 6]
STOCK = 'TSLA'


@timing
def main():
    stock_data = get_data.read_yahoo_data(STOCK, START_DATE)
    etf_data = get_data.read_yahoo_data(ETF, START_DATE)
    print(etf_data)
    return stock_data, etf_data


if __name__ == "__main__":
    main()
