import sys
from utils import logger
from utils.decorator import timing
from utils.constants import *
from data import data_reader, feature_engineering
from models import predict, train

LOGGER = logger.setup_logger(__name__)


@timing
def run(stock):
    equity_data = data_reader.read_yahoo_data(stock)
    etf_data = data_reader.read_yahoo_data(ETF)
    equity_enriched = feature_engineering.FeatureEngineering(lookback=EQUITY_LOOKBACK,
                                                             window=PREDICTION_WINDOW,
                                                             length=len(equity_data)).run(equity_data)

    etf_enriched = feature_engineering.FeatureEngineering(lookback=ETF_LOOKBACK,
                                                          window=PREDICTION_WINDOW,
                                                          length=len(etf_data)).run(etf_data)

    #TODO: Then combine them somehow
    return


if __name__ == '__main__':
    run(sys.argv[1])