# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 19:46:46 2019

@author: Brian Cote
"""

class RF_Classifier(object):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import cross_val_score
    from Market_Viz import Plots
    import numpy as np
    
    def __init__(self):
        pass
    
    def predict(self,dfs):
        #Tune Case via Grid Search
        rf_pred,ytest = self._randomForest(dfs)
        return(rf_pred)

    '''''''''''''''''
    Ensemble Methods
    ''''''''''''''''''
      - Note:   Literature suggests, using iterative testing over many technology sector stocks, that
                in a long term time-frame, RF and Gradient Boost are statistically similar in their
                performance. However, during lower trading windows (5-30 days) RF's outpaced their
                Gradient Boosted counterparts. For this reason we'll proceed with RF's, as we are
                doing an aggressive 20-day trading strategy.
                
                This is likely because GBM's are very susceptible to overfitting noisy data. Also has
                more difficult tuning (3 vs 2). Also, RF's handle highly-correlated data better, which
                is an implicit assumption with stock predictors. Since we have no categorical variables,
                we don't need to worry about advantages/disadvantages there. 
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    def _randomForest(self,dfs):
        train, ytrain, test, ytest = dfs
        
        rg = self._gridSearch()
        rf = self.RandomForestClassifier()
        
        rf_random = self.RandomizedSearchCV(estimator = rf, param_distributions = rg,
                                       n_iter = 10, cv = 3, verbose=3, random_state=42, n_jobs = -1)
        
        
        rf_random.fit(train, ytrain)
        print('Best Estimators Through Random Grid Search:')
        print(rf_random.best_params_)
        print('\n')
        print('..............................')
        print('Tuned Random Forest Ensemble Output:')
        print('..............................')
        
        best_random = rf_random.best_estimator_
        scores = self.cross_val_score(best_random,train,ytrain,cv=5)
        for i, score in enumerate(scores):
            print("Validation Set {} score: {}".format(i, score))
        print('\n')
        
        pred = best_random.predict(test)
        
        self.Plots.ConfusionMat(pred,ytest)
        self.Plots.DrawRoc(best_random,test, ytest)
        
        return(pred,ytest)
        
    def _gridSearch(self):
        # Number of trees in random forest
        n_estimators = [int(x) for x in self.np.linspace(start = 100, stop = 500, num = 100)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in self.np.linspace(2, 20, num = 1)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [5, 10, 15]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1,2,5,10]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        
        random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
        return (random_grid)