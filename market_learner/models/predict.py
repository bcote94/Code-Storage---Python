import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as cmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from utils import logger
from utils.constants import *

LOGGER = logger.setup_logger(__name__)


def fit_predict(train, train_y, equity):
    os.path.expanduser(MODEL_PATH)
    path = "{}{}_model.json".format(MODEL_PATH, equity)
    estimator = _train(train=train, train_y=train_y, path=path)
    return estimator


def static_fit(train, train_y, **params):
    tree_model = RandomForestClassifier(n_estimators=params.get('n_estimators'),
                                        criterion="gini",
                                        max_depth=params.get('max_depth'),
                                        min_samples_split=params.get('min_samples_split'),
                                        min_samples_leaf=params.get('min_samples_leaf'),
                                        max_features=params.get('max_features'),
                                        bootstrap=True,
                                        verbose=1)

    estimator = tree_model.fit(train, train_y)
    return estimator


def _cv_fit(train, train_y):
    rf_random = RandomizedSearchCV(estimator=RandomForestClassifier(),
                                   param_distributions=_grid_constructor(),
                                   n_iter=CV_PARAMS.get('n_iter'),
                                   scoring=CV_PARAMS.get('scoring'),
                                   cv=CV_PARAMS.get('cv'),
                                   verbose=1,
                                   n_jobs=-1)
    rf_random.fit(train, train_y)
    params = rf_random.best_params_
    estimator = rf_random.best_estimator_
    _cv_performance(estimator, train, train_y)
    return params, estimator


# noinspection PyTypeChecker
def _grid_constructor():
    random_grid = {'n_estimators': [int(x) for x in np.linspace(start=10, stop=500, num=20)],
                   'max_features': ['sqrt', 'log2', None],
                   'max_depth': [int(x) for x in np.linspace(1, 10, num=1)],
                   'min_samples_split': [2, 5, 10, 15],
                   'min_samples_leaf': [1, 2, 5],
                   }
    return random_grid


def _cv_performance(estimator, train, train_y):
    scores = cross_val_score(estimator,
                             train,
                             train_y,
                             cv=CV_PARAMS.get('cv'),
                             scoring=CV_PARAMS.get('scoring'))
    for i, score in enumerate(scores):
        LOGGER.debug("Validation Set {} score: {}".format(i, score))
    return 1


def _tot_performance(pred, test_y):
    truth_array = []
    for i in range(len(test_y)):
        truth_array.append(1. if pred[i] == test_y[i] else 0.)
    print('The Total Predictive Accuracy Is:', 100 * sum(truth_array) / len(test_y))

    # Prints confusion matrix
    cm = cmat(test_y, pred)
    spe = cm[0, 0] / sum(cm[0,])
    sen = cm[1, 1] / sum(cm[1,])
    print("Sensitivity is", 100 * spe, "% and Specificity is", 100 * sen, "%")


def _train(train, train_y, path):
    if not os.path.exists(path):
        LOGGER.debug("Building Model Parameters...")
        params, estimator = _cv_fit(train, train_y)
        LOGGER.debug("Parameters generated: {}".format(params))
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False)
    else:
        LOGGER.debug("Importing Model Parameters...")
        with open(path, 'r') as file:
            params = json.load(file)
        LOGGER.debug("Parameters Loaded: {}".format(params))
        estimator = static_fit(train, train_y, **params)
    return estimator
