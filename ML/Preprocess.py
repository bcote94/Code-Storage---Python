# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 19:43:42 2019

@author: Brian Cote
"""
class Preprocessor(object):
    import pandas as pd
    import numpy as np
    from pandas_datareader import data as reader
    from ML.Viz import Plots
    import os
    
    def __init__(self,start_date, end_date, ticker, idx_days_back, stock_days_back, pred_window, index_ticker, use_reader):
        self.start_date = start_date
        self.end_date   = end_date
        self.ticker     = ticker
        self.idx_days_back = idx_days_back
        self.stock_days_back = stock_days_back
        self.pred_window = pred_window
        self.index_ticker = index_ticker
        self.use_reader = use_reader

    def run(self):
        #Aggressive monthly trading strategy
        #Stock: 1 week back -- recent data better says literature
        #Index: 1 quarter back -- older data better says literature
        Stock = self._getData(self.ticker, self.stock_days_back, self.pred_window)
        IDX   = self._getData(self.index_ticker, self.idx_days_back, self.pred_window)
        
        #Drops our unneeded variables
        del Stock['WilliamsR']
        del IDX['ROC']
        del IDX['Disparity']
        
        Index_Stock_Data = self.pd.merge(IDX,Stock, left_index=True, right_index=True, how='outer')
        Index_Stock_Data = Index_Stock_Data.replace([self.np.inf,-self.np.inf],self.np.nan).dropna().rename(columns={'WilliamsR':'WilliamsR_x','ROC':'ROC_y'})
        Index_Stock_Data.drop(['Output_x','Open_x','High_x','Low_x','Close_x','Volume_x','Open_y', 'High_y','Low_y','Close_y','Volume_y'],axis=1, inplace=True)
        return(Index_Stock_Data)
    
    ########################################################################
    ####GetData sub-functions
    def _getData(self, ticker,time,window):
        data = self._importStock(ticker)
        data['K'],data['D'],data['WilliamsR'] = self.getKDR(time,data)
        data['Momentum'],data['ROC'] = self.getMROC(time,data)
        data['RSI'] = self.getRSI(data,time)
        data['ATR'] = self.ATR(data)
        data['Volatility'], data['Disparity'] = self.getVolatility(data,time)
        data['MACD'] = self.getMACD(data)
        data['Obv'], data['Output'] = self.getOBVandIndic(data,window)
    
        return data
        
    def _importStock(self,ticker):
        if self.use_reader:
            return self.reader.DataReader(ticker,'yahoo', self.start_date, self.end_date).drop(['Adj Close'],axis=1)
        else:
            stock_file = '/home/data/{0}.csv'.format(ticker)
            if self.os.path.exists(stock_file):
                return self.pd.read_csv(stock_file, low_memory=False).set_index('Date').drop(['Adj Close'],axis=1)
            else:
                raise RuntimeError('Ensure you have {0}\'s stock data imported to {1}'.format(ticker, stock_file))
    
    def getKDR(self,days, data):
        # The Williams %R represents a market’s closing level versus the highest high for the lookback period.
        # Conversely, the Fast Stochastic Oscillator, which moves between 0 and 100, illustrates a market’s close
        # in relation to the lowest low
        kmat = self.np.zeros(len(data))
        dmat = self.np.zeros(len(data))
        rmat = self.np.zeros(len(data))
        
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
        # Price Rate of Change is a pure volatility oscillator
        # Momentum is an unnamed indicator, an additive version used in papers
    def getMROC(self,days,data):
        mom = self.np.zeros(len(data))
        ROC = self.np.zeros(len(data))
        for i in range(days,len(data)):
            mom[i] = data.Close.iloc[i,] - data.Close.iloc[i-days,]
            ROC[i] = (data.Close.iloc[i,]-data.Close.iloc[i-days,])/data.Close.iloc[i-days,]
            
        return(mom, ROC)
        
    #Current On_Balance Volume
    #On days where price goes up, cumulatively adds volume, and vice versa
        #Indicator is based on weekly change of price. Each day is too variable.
    def getOBVandIndic(self,data,window):
        indic = self.np.zeros(len(data))
        obv = self.np.zeros(len(data))
        obv[0] = data.Volume.iloc[0]
        
        for i in range(1,len(data)):
            x0 = data.Close.iloc[i-1,]
            x1 = data.Close.iloc[i,]
            change = x1 - x0
            
            if change > 0:
                obv[i] = obv[i-1] + data.Volume.iloc[i,]
            elif change < 0:
                obv[i] = obv[i-1] - data.Volume.iloc[i,]
            else:
                obv[i] = obv[i-1]
                
        #What we actually care about: Based on today, what is price in next 5 days?
        for i in range(1,len(data)-window):
            if (data.Close.iloc[i+window,] - data.Close.iloc[i,]) >= 0:
                indic[i] = 1
            if (data.Close.iloc[i+window,] - data.Close.iloc[i,]) < 0:
                indic[i] = -1
        
        return(obv, indic)
        
    
    
    #Measure of stocks volatility. Its average percent change over a range of days.
    def getVolatility(self,data,days):
        vol = self.np.zeros(len(data))
        dis = self.np.zeros(len(data))
        for i in range(days,len(data)):
            c=0
            dis[i] = 100*data.Close.iloc[i,]/data.Close.iloc[i-10:i,].mean()
            for j in range(i-days+1,i):
                x1 = data.Close.iloc[j,]
                x0 = data.Close.iloc[j-1,]
                c+= (x1-x0)/x0
            vol[i] = 100*c/days
        
        return(vol, dis)
        
    #Moving Average Convergence Divergence
    #Constant indicators, irrespective of time
    def getMACD(self,data):
        n = len(data)
        macd  = self.np.zeros(n)
        exp12 = self.np.zeros(n)
        exp26 = self.np.zeros(n)
        
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
    def ATR(self,data):
        TR = self.np.zeros(len(data))
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
    def getRSI(self,data,days):
        n = len(data)
        rsi     = self.np.zeros(n)
        change  = self.np.zeros(n)
        gain    = self.np.zeros(n)
        loss    = self.np.zeros(n)
        avgGain = self.np.zeros(n)
        avgLoss = self.np.zeros(n)
        
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
        
    ####End GetData Functions
    ###########################################################################
    
    #Scales or Normalizes Data and then provides a train/models split
    def trainTestTransform(self,Index_Stock_Data):
        from sklearn import preprocessing 
        
        train = Index_Stock_Data[Index_Stock_Data.Output_y!=0].drop('Output_y', axis=1)
        train_y = Index_Stock_Data[Index_Stock_Data.Output_y!=0]['Output_y']
        test = Index_Stock_Data[Index_Stock_Data.Output_y==0].drop('Output_y', axis=1)
        
        modelDfs = []
        scalar = preprocessing.StandardScaler()
        for df in train, test:
            scaled = scalar.fit_transform(df)
            df_scaled = self.pd.DataFrame(scaled, columns=df.columns, index=df.index)
            modelDfs.append(df_scaled)
        modelDfs.append(train_y)


        return modelDfs
