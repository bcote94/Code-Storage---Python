# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:24:38 2019

@author: Brian Cote
"""

class Stock_Predict(object):
    import pandas as pd
    from Market_Preprocess import Preprocessor
    from Market_Model import RF_Classifier
    
    def __init__(self,start_date,end_date,ticker,train_min,train_max):
        self.start_date = start_date
        self.end_date   = end_date
        self.ticker     = ticker
        self.train_min  = train_min
        self.train_max  = train_max
        
        #Change Print Format
        self.pd.set_option('display.float_format', '{:.2f}'.format)
        self.pd.set_option('display.max_columns',20)
        self.pd.set_option('display.expand_frame_repr', False)
        
    def predict(self):
        preprocess = self.Preprocessor(self.start_date,self.end_date,self.ticker,self.train_min,self.train_max)
        Stock, SPY = preprocess.run()
        print(Stock.shape)
        print(SPY.shape)
        modelDFs   = preprocess.trainTestTransform(SPY,Stock)
        
        rf_pred    = self.RF_Classifier().predict(modelDFs)
        
        return rf_pred

        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        