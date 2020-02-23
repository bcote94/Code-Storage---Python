# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:24:38 2019

@author: Brian Cote
"""

class Stock_Predict(object):
    import pandas as pd
    from ML.Preprocess import Preprocessor
    from ML.Model import RF_Classifier
    
    def __init__(self,start_date, end_date, ticker, cv=True):
        self.preprocess = self.Preprocessor(start_date, end_date, ticker)
        self.RF = self.RF_Classifier(cv = cv)
        
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
