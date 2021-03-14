import numpy as np
import pandas as pd
from utils import logger
from utils.constants import *
from sklearn import preprocessing

LOGGER = logger.setup_logger(__name__)


def merge(etf, equity):
    df = pd.merge(etf, equity, left_index=True, right_index=True, how='outer', suffixes=['_etf', '_equity']) \
        .replace([np.inf, -np.inf], np.nan) \
        .drop('label_etf', axis=1) \
        .rename({'label_equity': 'label'}, axis=1)
    return df


def train_test_split(data):
    train, test = np.split(data, [int(.90 * len(data))])
    train_y, test_y = train['label'], test['label']
    train, test = train.drop('label', axis=1), test.drop('label', axis=1)
    return train, test, train_y, test_y


def scale(data):
    scaler = preprocessing.StandardScaler()
    scaled = scaler.fit_transform(data)
    return pd.DataFrame(scaled, columns=data.columns, index=data.index)
