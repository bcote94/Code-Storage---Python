# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 19:46:46 2019

@author: Brian Cote
"""

class RF_Classifier(object):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from ML.Viz import Plots
    import numpy as np
    
    def __init__(self, cross_validation, scoring, hyper_parameters, cross_validation_params):
        self.random_grid = self._gridSearch()
        self.cross_validation = cross_validation
        self.scoring = scoring
        self.hyper_parameters = hyper_parameters
        self.cross_validation_params = cross_validation_params
    
    def predict(self,dfs):
        rf_pred = self._rfNoTune(dfs) if not self.cross_validation else self._randomForest(dfs)
        if self.cross_validation:
            import json
            with open('/home/data/hyper_parameters.json', 'w') as file:
                json.dump(self.optimized_params, file)
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
    
    def _rfNoTune(self,dfs):
        train, test, ytrain = dfs
        tree_model = self.RandomForestClassifier(n_estimators=self.hyper_parameters.get('n_estimators'),
                                            criterion="gini",
                                            max_depth=self.hyper_parameters.get('max_depth'),
                                            min_samples_split=self.hyper_parameters.get('min_samples_split'),
                                            min_samples_leaf=self.hyper_parameters.get('min_samples_leaf'),
                                            max_features=self.hyper_parameters.get('max_features'),
                                            bootstrap=self.hyper_parameters.get('bootstrap'),
                                            verbose=1)
        
        print('..............................')
        print('Random Forest Ensemble Output:')
        print('..............................')
        estimator = tree_model.fit(train,ytrain)
        self._print_cv_scores(estimator, train, ytrain)
        return estimator.predict(test)

    def _randomForest(self,dfs):
        train, test, ytrain = dfs        
        rf_random = self.RandomizedSearchCV(estimator = self.RandomForestClassifier(), 
                                            param_distributions = self.random_grid,
                                            n_iter = self.cross_validation_params.get('n_iter'), 
                                            scoring = self.scoring,
                                            cv = self.cross_validation_params.get('cv'), 
                                            verbose=10,
                                            n_jobs = -1)
        rf_random.fit(train, ytrain)
        self.optimized_params = rf_random.best_params_
        print('Best Estimators Through Random Grid Search:')
        print(self.optimized_params)
        print('\n')
        print('..............................')
        print('Tuned Random Forest Ensemble Output:')
        print('..............................')
        estimator = rf_random.best_estimator_
        self._print_cv_scores(estimator, train, ytrain)
        return estimator.predict(test)
    
    def _print_cv_scores(self, estimator, train, ytrain):
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(estimator, 
                                 train, 
                                 ytrain,
                                 cv=self.cross_validation_params.get('cv'),
                                 scoring = self.scoring)
        for i, score in enumerate(scores):
            print("Validation Set {} score: {}".format(i, score))
        print('\n')
        
    def _gridSearch(self):
        # Number of trees in random forest
        n_estimators = [int(x) for x in self.np.linspace(start = 10, stop = 500, num = 20)]
        # Number of features to consider at every split
        max_features = ['sqrt', 'log2', None]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in self.np.linspace(1, 10, num = 1)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10, 15]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1,2,5]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        
        random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
        return (random_grid)