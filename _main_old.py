from dateutil.relativedelta import relativedelta
from ML.Predict import Stock_Predict
import datetime
import json
import os
import sys

STOCK = sys.argv[1]
if len(sys.argv) == 3:
    HYPER_PARAM_PATH = sys.argv[2]
else:
    HYPER_PARAM_PATH = '/home/data/hyper_parameters.json'


def main():
    cross_validation_params = {'n_iter': 50, 'cv': 5}
    if os.path.isfile(HYPER_PARAM_PATH):
        with open(HYPER_PARAM_PATH) as hyper_params:
            hyper_parameters = json.loads(hyper_params.read())
    else:
        hyper_parameters = None

    params = {'start_date': (datetime.datetime.today() - relativedelta(years=4)).strftime('%d-%m-%Y'),
              'end_date': datetime.datetime.today().strftime('%d-%m-%Y'),
              'cross_validate': True,
              'idx_days_back': 90,
              'stock_days_back': 5,
              'pred_window': 20,
              'index_ticker': 'SPY',
              'stock': STOCK,
              'hyper_parameters': hyper_parameters,  # Requires a dict argument iff cross_validate is true
              'scoring': 'average_precision',
              'cross_validation_params': cross_validation_params,
              'use_reader': False}

    Stock_Predict(params).predict()


if __name__ == '__main__':
    main()