

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

HEADER INFORMATION - TITLES, FORMATS, ETC.

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data
import statsmodels.api as sm
import scipy.stats as sc
from scipy.stats import uniform
from scipy.stats import norm
from numpy import linalg as la
from pylab import *

#Change Print Format
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns',20)
pd.set_option('display.expand_frame_repr', False)

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Data Import
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#Define tickers to download
#tickers = ['AAPL', 'SPY', 'AMD', 'NFLX']

#Pull seven years of data
##70/30 Training split:
###2011-2016: Train
###2017-2018: Test
start_date = '2011-01-01'
end_date = '2018-12-31'

def getStock(ticker):
    from pandas_datareader import data
    x = data.DataReader(ticker,'yahoo', start_date, end_date)
    del x['Adj Close']
    return(x)
    

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Constructing our Covariates and Splitting Data
    We will start with just AAPL data for now
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#Stochastic Oscillator %K and Stochastic %D, which is a Moving Average of %K 
#Useful for identifying closing price relative to its price range over time

#Wiliams %R
   #Momentum indicator measuring over/under selling - similar but slightly more focused than %K
def getKDR(days, data):
    kmat = np.zeros(len(data))
    dmat = np.zeros(len(data))
    rmat = np.zeros(len(data))
    
    for i in range(days,len(data)):
        c = data.Close.iloc[i,]
        low = min(data.Close.iloc[i-days:i,])
        high = max(data.Close.iloc[i-days:i,])
        
        kmat[i] = (c-low)/(high-low)*100    
        rmat[i] = (high-c)/(high-low)*-100
  
    for i in range(days,len(data)):
        for j in (range(0,days-1)):
            dmat[i] = dmat[i] + kmat[i-j]/days
        
    return(kmat, dmat,rmat)


#Momentum Indicators
    # Price Rate of Change is a pure momentum oscillator
    # Momentum is an unnamed indicator, an additive version used in papers
def getMROC(days,data):
    mom = np.zeros(len(data))
    ROC = np.zeros(len(data))
    for i in range(days,len(data)):
        mom[i] = data.Close.iloc[i,] - data.Close.iloc[i-days,]
        ROC[i] = data.Close.iloc[i,]/data.Close.iloc[i-days,] * 100
        
    return(mom, ROC)
    
#Current On_Balance Volume
#On days where price goes up, cumulatively adds volume, and vice versa
def getOBVandIndic(data):
    indic = np.zeros(len(data))
    obv = np.zeros(len(data))
    obv[0] = data.Volume.iloc[0]
    
    for i in range(1,len(data)):
        x0 = data.Close.iloc[i-1,]
        x1 = data.Close.iloc[i,]
        change = x1 - x0
        
        if change > 0:
            obv[i] = obv[i-1] + data.Volume.iloc[i,]
            indic[i] = int(1)
        elif change < 0:
            obv[i] = obv[i-1] - data.Volume.iloc[i,]
            indic[i] = int(-1)
        else:
            obv[i] = obv[i-1]
            indic[i] = int(1)
    
    return(obv, indic)
    
#Measure of stocks volatility. Its average percent change over a range of days.
def getVolatility(data,days):
    vol = np.zeros(len(data))
    dis = np.zeros(len(data))
    for i in range(days,len(data)):
        c=0
        dis[i] = 100*data.Close.iloc[i,]/data.Close.iloc[i-10:i,].mean()
        for j in range(i-days+1,i):
            x1 = data.Close.iloc[j,]
            x0 = data.Close.iloc[j-1,]
            c+= (x1-x0)/x0
        vol[i] = c*100
    
    return(vol, dis)
    
#Moving Average Convergence Divergence
#Constant indicators, irrespective of time
def getMACD(data):
    n = len(data)
    macd = np.zeros(n)
    exp12 = np.zeros(n)
    exp26 = np.zeros(n)
    
    mult12 = 2/(13)
    mult26 = 2/(27)

    sma12 = data.Close.iloc[0:11,].mean()
    exp12[0] = sma12
    for i in range(1,n-11):
        exp12[i] = (data.Close.iloc[11+i,]-exp12[i-1])*mult12 + exp12[i-1]
    
    sma26 = data.Close.iloc[0:25,].mean()
    exp26[0] = sma26
    for i in range(1,n-25):
        exp26[i] = (data.Close.iloc[25+i,]-exp26[i-1])*mult26 + exp26[i-1]
        macd[i] = exp12[i] - exp26[i]
    
    return (macd)

#Volatility index: Average True Range
def ATR(data):
    ATR = np.zeros(len(data))
    TR = np.zeros(len(data))
    for i in range(1,len(data)):
        x0 = data.High.iloc[i,] - data.Low.iloc[i,]
        x1 = abs(data.High.iloc[i,]-data.Close.iloc[i-1,])
        x2 = abs(data.Low.iloc[i,]-data.Close.iloc[i-1,])
        
        TR[i] = max(x0,x1,x2)
    
    data['TR'] = TR
    data['ATR'] = data['TR'].ewm(span=14).mean()
    del data['TR']
    
    return(data['ATR'])

#Popular momentum indicator, determining over/under purchasing. That is, if demand is unjustifiably pushing the stock upward
#This  condition  is  generally  interpreted  as  a  sign  that  the  stock  isovervalued and the price is likely to go down. 
#A stock is said to be oversold when the price goesdown sharply to a level below its true value. This is a result caused due 
#to panic selling. RSIranges  from  0  to  100  and  generally,  when  RSI  is  above  70,  it  may  indicate  that  the  stock  
#is overbought and when RSI is below 30, it may indicate the stock is oversold.
def getRSI(data,days):
    n = len(data)
    rsi = np.zeros(n)
    change = np.zeros(n)
    gain = np.zeros(n)
    loss = np.zeros(n)
    avgGain = np.zeros(n)
    avgLoss = np.zeros(n)
    
    for i in range(1,n):
        change[i] = data.Close.iloc[i] - data.Close.iloc[i-1,]
        
        if change[i] > 0:
            gain[i] += change[i]
        else:
            loss[i] += abs(change[i])
    
    for i in range(days,n):
        x = gain[i-days:i]
        y = loss[i-days:i]
        
        avgGain[i] = sum(x)/days
        avgLoss[i] = sum(y)/days
        rs = avgGain[i]/avgLoss[i]
        rsi[i] = 100 - 100/(1+rs)
    
    return(rsi)


  
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#Runs all the above, gathers all vars in one neat package
def getData(ticker,time):
    data = getStock(ticker)
    data['K'],data['D'],data['WilliamsR'] = getKDR(time,data)
    data['Momentum'],data['ROC'] = getMROC(time,data)
    data['RSI'] = getRSI(data,time)
    data['ATR'] = ATR(data)
    data['Volatility'], data['Disparity'] = getVolatility(data,time)
    data['MACD'] = getMACD(data)
    data['Obv'], data['Output'] = getOBVandIndic(data)
    
    #Random spot check for data integrity
    t1 = int(np.random.uniform(1,len(data)-10,1))
    print(data.iloc[t1:(t1+10),])
    
    return(data)

AAPL = getData('AAPL',5)
SPY = getData('SPY',90)
MSFT = getData('MSFT',5)
AMD = getData('AMD',5)

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Exploratory Plotting: Moving Averages (5/10/20/90/270)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def Exploratory_Plot(data):
    tick = data.Close.loc[:,]
    
    # Calculate the 20 and 100 days moving averages of the closing prices
    monthly_rolling = tick.rolling(window=20).mean()
    quarter_rolling = tick.rolling(window=90).mean()
    yearly_rolling = tick.rolling(window=270).mean()
    
    fig, ax = plt.subplots(figsize=(16,9))    
    
    plt.plot(tick.index, tick)
    plt.plot(monthly_rolling.index, monthly_rolling, label='Monthly Rolling Average')
    plt.plot(quarter_rolling.index, quarter_rolling, label='Quarterly Rolling Average')
    plt.plot(yearly_rolling.index, yearly_rolling, label='Yearly Rolling Average')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Adjusted closing price ($)')
    ax.legend()
    plt.rc('grid',linestyle='dashdot',color='grey')
    plt.grid()   

Exploratory_Plot(SPY)
Exploratory_Plot(AAPL)
Exploratory_Plot(AMD)
Exploratory_Plot(MSFT)

'''
Correlation Plotting
    - Williams %R and Stochastic %K are perfectly correlated. 
        - A box in center of somewhat correlated momentum indicators
        - Will drop %R then, and use K/D instead.
        
    - As expected -- High/Low/Open/Close are perfectly correlated and
                     thus will likely not be good predictors together.
            
    - We can see though, that despite having lots of momentum indicators,
      they aren't all highly correlated. This confirms the literature.
      There are different momentum indicators that measure slightly
      different areas of 'momentum', and we want to include them all.
      
      Hopefully when included alongside on-balance-volume and a naive
      volatility idicator, we can build a strong model. 
      
    - PROC has far less correlation issues than additive momentum -- will remove
      
      Also note: We will have TWO VERSIONS of each of these-
          - Long term Index Momentum/ROC/etc.
          - Short term stock Momentum/ROC/etc.
          
    - For $SPY Longterm index, ROC/RSI/Disparity are very highly correlated. So we drpo those
      only for SPY and keep RSI, as it's the most often used in literature. However, WilliamsR
      is fine, so we'll leave that for SPY.
'''
import seaborn as sns
corrmat2 = AAPL.corr()
corrmat1 = SPY.corr()
f, ax = plt.subplots(figsize=(12,10))
sns.heatmap(corrmat1)

#Drops our unneeded variables
del AAPL['WilliamsR']
del AMD['WilliamsR']
del MSFT['WilliamsR']
del SPY['ROC']
del SPY['Disparity']

'''Final Data'''
Index_Stock_Data = pd.merge(SPY,AAPL,left_index=True,right_index=True,how='outer')

'''Subsetting'''
#Trying to ultimately predict the stock motion
Y = Index_Stock_Data['Output_y']

#Naive covariates
X_naive = Index_Stock_Data[['High_x','Low_x','Open_x','Close_x','Volume_x',
                            'High_y','Low_y','Open_y','Close_y','Volume_y']]

#Advanced covariates


'''
Train/Test
'''
X_Naive_train = X_naive[(X_naive.index > '2011-01-01') & (X_naive.index < '2016-01-01')]
X_Naive_test = X_naive[(X_naive.index >= '2016-01-01')]

Y_train = Y[(Y.index>'2011-01-01') & (Y.index<'2016-01-01')]
Y_test = Y[Y.index>='2016-01-01']

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Descriptive Statistics & Preliminary Analysis
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#All data - pull panel wider
Index_Stock_Data.describe()
Index_Stock_Data.Close_y.describe()

f, ax = plt.subplots(figsize=(12,10))
plt.plot(Index_Stock_Data.MACD_x.index, Index_Stock_Data.MACD_x)
f, ax = plt.subplots(figsize=(12,10))
plt.plot(SPY.Close.index,SPY.Close)

#Specific data class
SPY.describe()    

    
'''
SVM
'''
from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(X_Naive_train,Y_train)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    








