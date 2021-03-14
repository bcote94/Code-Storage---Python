import numpy as np
import pandas as pd
from utils import logger
from utils.constants import *
from sklearn import preprocessing

LOGGER = logger.setup_logger(__name__)


def _train_test_split(data):
    train = data[data.Output_y != -1].drop('Output_y', axis=1)
    train_y = data[data.Output_y != -1]['Output_y']
    test = data[data.Output_y == 1].drop('Output_y', axis=1)

    modelDfs = []
    c = preprocessing.StandardScaler()
    for df in train, test:
        scaled = c.fit_transform(df)
        df_scaled = pd.DataFrame(scaled, columns=df.columns, index=df.index)
        modelDfs.append(df_scaled)
    modelDfs.append(train_y)
    return modelDfs
