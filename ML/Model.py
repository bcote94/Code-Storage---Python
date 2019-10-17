# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 19:46:46 2019

@author: Brian Cote
"""

class RF_Classifier(object):

    from ML.Viz import Plots
    import numpy as np
    
    def __init__(self):
        pass
    
    def predict(self,dfs):
        #Tune Case via Grid Search
        rf_pred = self._rfNoTune(dfs)
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
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        #Wont normalize the data this time, want to have interpretable coefficients maybe
        train, ytrain, test, ytest = dfs
        
        #Note - thorough testing has shown this to be a decent model setup by default.
        #Run _randomForest() to re-tune it. Note that may take numerous hours.
        tree_model = RandomForestClassifier(n_estimators=311,
                                            criterion="gini",
                                            max_depth=2,
                                            min_samples_split=15,
                                            min_samples_leaf=10,
                                            max_features='sqrt',
                                            bootstrap=True,
                                            oob_score=True,
                                            verbose=0)
        

        print('..............................')
        print('Random Forest Ensemble Output:')
        print('..............................')
        scores = cross_val_score(tree_model.fit(train,ytrain),train,ytrain,cv=5,scoring="accuracy")
        for i, score in enumerate(scores):
            print("Validation Set {} score: {}".format(i, score))
        print('\n')
        
        tree_model.fit(train,ytrain)
        y_pred = tree_model.predict(test)
        
        self.Plots().ConfusionMat(pred=y_pred,ytest=ytest)
        self.Plots().DrawRoc(tree_model,test,ytest)
        
        return(y_pred)

    def _randomForest(self,dfs):
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.model_selection import cross_val_score
        
        train, ytrain, test, ytest = dfs
        
        rg = self._gridSearch()
        rf = RandomForestClassifier()
        print("Initialized")
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = rg,
                                       n_iter = 2, cv = 5, verbose=10, random_state=1, n_jobs = -1)
        
        print("CV Complete")
        rf_random.fit(train, ytrain)
        print('Best Estimators Through Random Grid Search:')
        print(rf_random.best_params_)
        print('\n')
        print('..............................')
        print('Tuned Random Forest Ensemble Output:')
        print('..............................')
        
        best_random = rf_random.best_estimator_
        scores = cross_val_score(best_random,train,ytrain,cv=5)
        for i, score in enumerate(scores):
            print("Validation Set {} score: {}".format(i, score))
        print('\n')
        
        pred = best_random.predict(test)
        
        self.Plots.ConfusionMat(pred,ytest)
        self.Plots.DrawRoc(best_random,test, ytest)
        
        return(pred)
        
    def _gridSearch(self):
        # Number of trees in random forest
        n_estimators = [int(x) for x in self.np.linspace(start = 100, stop = 200, num = 10)]
        # Number of features to consider at every split
        max_features = ['sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in self.np.linspace(2, 10, num = 2)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1,2,5]
        # Method of selecting samples for training each tree
        bootstrap = [True]
        
        random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
        return (random_grid)