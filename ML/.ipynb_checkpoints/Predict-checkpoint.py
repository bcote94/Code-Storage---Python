# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:24:38 2019

@author: Brian Cote
"""

class Stock_Predict(object):
    from ML.Preprocess import Preprocessor
    from ML.Model import RF_Classifier
    import pandas as pd
    
    def __init__(self, params):
        if params.get('cross_validate') is False and params.get('hyper_parameters') is None:
            raise ValueError('If you are not cross validate optimizing, you MUST input a dict-type of models hyperparameters. See: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html for required parameters.')
        if params.get('cross_validate') is True and params.get('cross_validation_params') is None:
            raise ValueError('If you are cross validate optimizing, you MUST input a dict-type of tuning parameter keys for a Randomized Search. See: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html')
        if params.get('cross_validation_params') is None:
            print("Defaulting to 5-Fold Cross-Validation for Model Performance Analysis . . .")
            params['cross_validation_params'] = {'cv':5}
        
        self.preprocess = self.Preprocessor(start_date = params.get('start_date'), 
                                            end_date = params.get('end_date'), 
                                            ticker = params.get('stock'),
                                            idx_days_back = params.get('idx_days_back'),
                                            stock_days_back = params.get('stock_days_back'),
                                            pred_window = params.get('pred_window'),
                                            index_ticker = params.get('index_ticker'))
        
        self.RF = self.RF_Classifier(cross_validation = params.get('cross_validate'),
                                     scoring = params.get('model_metric'),
                                     hyper_parameters = params.get('hyper_parameters'),
                                     cross_validation_params = params.get('cross_validation_params'))
            
        #Change Print Format
        self.pd.set_option('display.float_format', '{:.2f}'.format)
        self.pd.set_option('display.max_columns',20)
        self.pd.set_option('display.expand_frame_repr', False)
        
    def predict(self):
        Index_Stock_Data = self.preprocess.run()
        modelDfs = self.preprocess.trainTestTransform(Index_Stock_Data)
        rf_pred = self.RF.predict(modelDfs)
        if (sum(rf_pred)) == -len(rf_pred):
            print("BEAR GANG BEAR GANG")
        elif (sum(rf_pred)) == len(rf_pred):
            print("BUY BUY BUY")
        else:
            print("REPLY HAZY. TRY AGAIN LATER.")
            print(pd.Series(rf_pred).value_counts())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
