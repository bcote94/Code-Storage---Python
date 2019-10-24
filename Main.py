# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:05:43 2019

@author: Brian Cote
"""
#See __init__ for instructions, use cases

from ML.Predict import Stock_Predict

start_date = '2011-01-01'
end_date = '2019-10-15'
train_max = '2019-01-01'

tsla = 'TSLA'
##YTD ML Profits:  $121 per share
##YTD DCA Profits: $-339 (loss) per share

res = Stock_Predict(start_date,end_date,train_max,amzn).predict()
print("Test Set Results for Model:")
print("Profits from ML Model: $",res[1])
print("\nProfits from Dollar Cost Averaging: $",res[2],'\n')

