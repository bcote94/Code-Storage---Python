# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:05:43 2019

@author: Brian Cote
"""

from Market_MachineLearning import Stock_Predict

start_date = '2011-01-01'
end_date = '2019-10-01'
ticker = 'AAPL'
train_min = '2011-09-01'
train_max = '2016-09-01'

stock, spy = Stock_Predict(start_date,end_date,ticker,train_min,train_max).predict()
