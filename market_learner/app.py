import sys
import numpy as np
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
    data_scaled = transformation.scale(data.drop('label', axis=1))
    data_scaled['label'] = data['label']

    train, test, train_y, test_y = transformation.train_test_split(data=data_scaled, split_per=.8)
    params, estimator = predict.fit_predict(train=train, train_y=train_y, equity=equity)

    test_y = test_y[0:-PREDICTION_WINDOW]
    test_pred = estimator.predict(test[0:-PREDICTION_WINDOW])
    predict.tot_performance(test_pred, test_y)

    best_estimator = predict.static_fit(data_scaled.drop('label', axis=1), data_scaled['label'], **params)
    pred = best_estimator.predict(np.array(data_scaled.drop('label', axis=1).iloc[-1]).reshape(1, -1))
    print(pred)
    return 1


if __name__ == '__main__':
    ticker = 'AAPL'
    run(ticker)
