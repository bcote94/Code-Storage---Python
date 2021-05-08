import sys
from utils import logger
from utils.decorator import timing
from utils.constants import *
from data import data_reader, feature_engineering, transformation
from models import predict

LOGGER = logger.setup_logger(__name__)


@timing
def run(equity, etf='SPY'):
    equity_data = data_reader.read_yahoo_data(ticker=equity)
    etf_data = data_reader.read_yahoo_data(ticker=etf)
    equity_enriched = feature_engineering.FeatureEngineering(lookback=EQUITY_LOOKBACK,
                                                             window=PREDICTION_WINDOW,
                                                             length=len(equity_data)).run(equity_data)

    etf_enriched = feature_engineering.FeatureEngineering(lookback=ETF_LOOKBACK,
                                                          window=PREDICTION_WINDOW,
                                                          length=len(etf_data)).run(etf_data)

    data = transformation.merge(etf=etf_enriched, equity=equity_enriched)
    train_raw, test_raw, train_y, test_y = transformation.train_test_split(data=data, split_per=.8)
    train, test = transformation.scale(train_raw), transformation.scale(test_raw)
    estimator = predict.fit_predict(train=train, train_y=train_y, equity=equity)

    test_y = test_y[0:-PREDICTION_WINDOW]
    pred = estimator.predict(test[0:-PREDICTION_WINDOW])
    predict._tot_performance(pred, test_y)

    #TODO: Now refit on all the normalized full set of data, sans the latest day

    t = transformation.scale(data)
    return equity_data, pred


if __name__ == '__main__':
    run(sys.argv[1])
