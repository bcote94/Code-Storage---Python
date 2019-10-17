# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:29:58 2019

@author: Brian Cote
"""
###Results: On Upward trending stocks, Dollar Cost Averaging wins out
###         On Downward/Sideways stocks, ML model mitigates losses considerably
##############################################################################
# Trading Strategy:
# I am using a simple trading strategy to measure performance. If I see an up 
# signal, buy a single share. If there's a sell, sell them all at current $.
##############################################################################
#SUGGESTIONS/FUTURE WORK: 
# There are likely more aggressive trading strategies that favor the ML approach
# over DCA, even in upward trending stocks. For example, if in concurrent sell-
# -signals, we saved money to buy more shares the next time, and re-invested
# past profits into future buy-plays, we'd likely see drastically more profits.
##############################################################################
# I can see this being kind of fun for YOLO monthlies/weeklies in a highly
# volatile market, or for informing a hedging strategy where you want to hold
# onto gains through market volatility, specifically where you don't have time
# to wait for market recovery (e.g.: nearing retirement).
##############################################################################