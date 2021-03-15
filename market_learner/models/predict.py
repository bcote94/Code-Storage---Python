import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from utils import logger
from utils.constants import *

LOGGER = logger.setup_logger(__name__)


def fit_predict(train, train_y, test, **params):
    tree_model = RandomForestClassifier(n_estimators=params.get('n_estimators'),
                                        criterion="gini",
                                        max_depth=params.get('max_depth'),
                                        min_samples_split=params.get('min_samples_split'),
                                        min_samples_leaf=params.get('min_samples_leaf'),
                                        max_features=params.get('max_features'),
                                        bootstrap=True,
                                        verbose=1)

    estimator = tree_model.fit(train, train_y)
    _performance(estimator, train, train_y)
    return estimator.predict(test)


def cv_fit(train, train_y):
    rf_random = RandomizedSearchCV(estimator=RandomForestClassifier(),
                                   param_distributions=_grid_constructor(),
                                   n_iter=CV_PARAMS.get('n_iter'),
                                   scoring=CV_PARAMS.get('scoring'),
                                   cv=CV_PARAMS.get('cv'),
                                   verbose=1,
                                   n_jobs=-1)
    rf_random.fit(train, train_y)
    optimized_params = rf_random.best_params_
    LOGGER.debug(optimized_params)
    estimator = rf_random.best_estimator_
    _performance(estimator, train, train_y)
    return optimized_params, estimator


def _performance(estimator, train, train_y):
    scores = cross_val_score(estimator,
                             train,
                             train_y,
                             cv=CV_PARAMS.get('cv'),
                             scoring=CV_PARAMS.get('scoring'))
    for i, score in enumerate(scores):
        LOGGER.debug("Validation Set {} score: {}".format(i, score))
    return 1


# noinspection PyTypeChecker
def _grid_constructor():
    random_grid = {'n_estimators': [int(x) for x in np.linspace(start=10, stop=500, num=20)],
                   'max_features': ['sqrt', 'log2', None],
                   'max_depth': [int(x) for x in np.linspace(1, 10, num=1)],
                   'min_samples_split': [2, 5, 10, 15],
                   'min_samples_leaf': [1, 2, 5],
                   }
    return random_grid

def ConfusionMat(pred, ytest):
    from sklearn.metrics import confusion_matrix

    truthArray = []
    for i in range(len(ytest)):
            # print "prediction: %d\tactual: %d" % (predictions[i], Y_test[i])
            truthArray.append(1. if pred[i] == ytest[i] else 0.)

        # print len(truthArray)
    print('The Total Predictive Accuracy Is:',100*sum(truthArray)/len(ytest))

    #Prints confusion matrix
    cm = confusion_matrix(ytest,pred)
    spe = cm[0,0]/sum(cm[0,])
    sen = cm[1,1]/sum(cm[1,])
    print("Sensitivity is",100*spe,"% and Specificity is",100*sen,"%")