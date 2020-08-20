from datetime import datetime
import pandas_datareader.data as web
from utils import logger
from utils.decorator import timing
from utils.constants import *
from model.fit_predict import testing

LOGGER = logger.setup_logger(__name__)


@timing
def main():
    data = web.get_data_yahoo(['TSLA'], start=datetime(2020, 1, 6), end=CURRENT_DATETIME)
    spy = web.get_data_yahoo([ETF], start=datetime(2020, 1, 6), end=CURRENT_DATETIME)
    return data, spy


if __name__ == "__main__":
    main()