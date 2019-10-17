# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:24:38 2019

@author: Brian Cote
"""

class Stock_Predict(object):
    import pandas as pd
    from ML.Preprocess import Preprocessor
    from ML.Model import RF_Classifier
    from ML.Viz import Plots
    
    def __init__(self,start_date,end_date,train_max,ticker):
        self.start_date = start_date
        self.end_date   = end_date
        self.ticker     = ticker
        self.train_max  = train_max
        
        #Change Print Format
        self.pd.set_option('display.float_format', '{:.2f}'.format)
        self.pd.set_option('display.max_columns',20)
        self.pd.set_option('display.expand_frame_repr', False)
        
    def predict(self):
        preprocess = self.Preprocessor(self.start_date,self.end_date,self.train_max,self.ticker)
        Stock, SPY = preprocess.run()
        print(Stock.shape)
        print(SPY.shape)
        modelDFs   = preprocess.trainTestTransform(SPY,Stock,True)
        
        rf_pred    = self.RF_Classifier().predict(modelDFs)
        
        testStock = Stock[Stock.index >= self.train_max]
        profit_mat = self.Plots().tradingStrategy(rf_pred,testStock)
        profits, dca_profits = self._profits(profit_mat)
        
        return profit_mat, profits, dca_profits
    
    def _profits(self,profit_mat):
        import numpy as np
        n = len(profit_mat)
        #Calculates total predicted profits in testing set
        shares_owned, position_cost, profits = np.zeros(n+1), np.zeros(n+1), np.zeros(n+1)
        for i in range(0,n):
            if profit_mat[i,2] > 0:
                shares_owned[i] = shares_owned[i-1] + 1
                position_cost[i] = position_cost[i-1] + profit_mat[i,1]
            
            if profit_mat[i,2] < 0:
                profits[i] = shares_owned[i-1]*profit_mat[i,1] - position_cost[i-1]
                shares_owned[i], position_cost[i] = 0, 0
        
        #Dollar Cost Average Profits - to compare vs Profits
        dca_profits =  sum([sum(profit_mat[n-1,1]-profit_mat[i,1] for i in range(0,n))])

        return sum(profits), dca_profits

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        