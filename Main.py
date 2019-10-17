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

#AMD is an example of an upwardly trending stock overall that has a sharp period of volatility -- can it predict that shift well?
amd = 'AMD'
##YTD ML Profits:  $14 per share
##YTD DCA Profits: $41 per share

tsla = 'TSLA'
##YTD ML Profits:  $121 per share
##YTD DCA Profits: $-339 (loss) per share

amzn = 'AMZN'
##YTD ML Profits:  $1330 per share
##YTD DCA Profits: $224 per share

#See __init__ for my trading strategy for ML. Can you think of a better one? Try coding one! See how it performs.
#Want to guess why it has such drastic success in 'TSLA'/'AMZN' compared to 'AMD'? Hint: Look at the test set plots

res = Stock_Predict(start_date,end_date,train_max,amzn).predict()
print("Test Set Results for Model:")
print("Profits from ML Model: $",res[1])
print("\nProfits from Dollar Cost Averaging: $",res[2],'\n')

