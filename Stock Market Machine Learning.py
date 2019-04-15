
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as dat
import statsmodels.api as sm
import scipy.stats as sc
from scipy.stats import uniform
from scipy.stats import norm
from numpy import linalg as la
from pylab import *

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc, confusion_matrix
from matplotlib import animation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

#Change Print Format
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns',20)
pd.set_option('display.expand_frame_repr', False)

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Data Import Function
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#Pull seven years of data
##70/30 Training split:
###2011-2016: Train
###2017-2018: Test
start_date = '2011-01-01'
end_date = '2018-12-31'

def getStock(ticker):
    x = dat.DataReader(ticker,'yahoo', start_date, end_date)
    del x['Adj Close']
    return(x)

#Runs all the above, gathers all vars in one neat package
def getData(ticker,time,window):
    data = getStock(ticker)
    data['K'],data['D'],data['WilliamsR'] = getKDR(time,data)
    data['Momentum'],data['ROC'] = getMROC(time,data)
    data['RSI'] = getRSI(data,time)
    data['ATR'] = ATR(data)
    data['Volatility'], data['Disparity'] = getVolatility(data,time)
    data['MACD'] = getMACD(data)
    data['Obv'], data['Output'] = getOBVandIndic(data,window)
    
    #Random spot check for data integrity
    t1 = int(np.random.uniform(1,len(data)-10,1))
    print(data.iloc[t1:(t1+10),])
    
    return(data)
    
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

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
               Constructing our Covariates
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
    # Price Rate of Change is a pure volatility oscillator
    # Momentum is an unnamed indicator, an additive version used in papers
def getMROC(days,data):
    mom = np.zeros(len(data))
    ROC = np.zeros(len(data))
    for i in range(days,len(data)):
        mom[i] = data.Close.iloc[i,] - data.Close.iloc[i-days,]
        ROC[i] = (data.Close.iloc[i,]-data.Close.iloc[i-days,])/data.Close.iloc[i-days,]
        
    return(mom, ROC)
    
#Current On_Balance Volume
#On days where price goes up, cumulatively adds volume, and vice versa
    #Indicator is based on weekly change of price. Each day is too variable.
def getOBVandIndic(data,window):
    indic = np.zeros(len(data))
    obv = np.zeros(len(data))
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
        vol[i] = 100*c/days
    
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
'''''''''''''Merging Data & Test/Train Split''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
def finalData(SPY,data,normalize=0):


    #Always join in the index data, in this case SPY
    Index_Stock_Data = pd.merge(SPY,data,left_index=True,right_index=True,how='outer')
    Index_Stock_Data = Index_Stock_Data[Index_Stock_Data.Output_y!=0]
    
    #Cleaning up na and infinite
    np.any(np.isnan(Index_Stock_Data))
    np.all(np.isfinite(Index_Stock_Data))
    
    Index_Stock_Data = Index_Stock_Data.replace([np.inf,-np.inf],np.nan)
    Index_Stock_Data = Index_Stock_Data.dropna()

    '''Subsetting'''
    Y = Index_Stock_Data['Output_y']
    X = Index_Stock_Data.drop(['Output_y','Output_x','Open_x','High_x','Low_x','Close_x','Volume_x',
                               'Open_y', 'High_y','Low_y','Close_y','Volume_y'],axis=1)
    
    #Spot Fix Data Names
    X = X.rename(columns={'WilliamsR':'WilliamsR_x','ROC':'ROC_y'})
    
    if normalize==1:
        '''Normalize the Data'''
        from sklearn import preprocessing    
        
        scalar = preprocessing.StandardScaler()
        scaled_X = scalar.fit_transform(X)
        X_df = pd.DataFrame(scaled_X, columns=X.columns, index=X.index)  
    else:
        X_df = X

    '''
    Train/Test
    '''
    X_train = X_df[(X_df.index > '2011-01-01') & (X_df.index < '2016-01-01')]
    X_test = X_df[(X_df.index >= '2016-01-01')]
    
    Y_train = Y[(Y.index>'2011-01-01') & (Y.index<'2016-01-01')]
    Y_test = Y[Y.index>='2016-01-01']
    
    return(X_train,Y_train,X_test,Y_test)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''OPTIMIZATION'''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


def ConfusionMat(pred,ytest):
    
    #Predictions
    truthArray = []
    for i in range(len(ytest)):
    		# print "prediction: %d\tactual: %d" % (predictions[i], Y_test[i])
    		truthArray.append(1. if pred[i] == ytest[i] else 0.)
    
    	# print len(truthArray)
    print('The Total Predictive Accuracy Is:',100*sum(truthArray)/len(ytest))
    
    #Prints confusion matrix
    cm = confusion_matrix(ytest,pred)
    spe = cm[0,0]/sum(cm[0,])
    sen = cm[1,1]/sum(cm[1,])
    print("Sensitivity is",100*spe,"% and Specificity is",100*sen,"%")
    
    
def DrawRoc(model,test,ytest):
    y_prob = model.predict_proba(test)
    true_probability_estimate = y_prob[:,1]
    
    fpr,tpr,threshold = roc_curve(ytest,true_probability_estimate)
    area = auc(fpr,tpr)
    plt.figure()
    plt.plot(fpr,tpr,linewidth = 2.0,label = "ROC curve (Area= %0.2f)" % area)
    plt.plot([0,1],[0,1],"r--")
    plt.xlabel("False Postive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc = "lower right")
    plt.show(block = False)

    
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                Support Vector Machines
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def SVM(finaldata):

    
    train, ytrain, test, ytest = finaldata
    
    clf = svm.SVC(kernel='linear',C=1)
    lin_svc = clf.fit(train,ytrain)

    pred = lin_svc.predict(test)
    print('..............................')
    print('Support Vector Machine Output:')
    print('..............................')
    print(ConfusionMat(pred,ytest))
    
def svc_param_selection(finaldata, nfolds):
    train, ytrain, test, ytest = finaldata
    
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(train,ytrain)
    grid_search.best_params_
    
    pred = grid_search.predict(test)
    print('..............................')
    print('Tuned Support Vector Machine Output:')
    print('..............................')
    print(ConfusionMat(pred,ytest))
    
    return(pred)
    

'''''''''''''''''
Ensemble Methods
''''''''''''''''''
 "   - Note: Literature suggests, using iterative testing over many technology sector stocks, that
            in a long term time-frame, RF and Gradient Boost are statistically similar in their
            performance. However, during lower trading windows (5-30 days) RF's outpaced their
            Gradient Boosted counterparts. For this reason we'll proceed with RF's, as we are
            doing an aggressive 20-day trading strategy.
            
            This is likely because GBM's are very susceptible to overfitting noisy data. Also has
            more difficult tuning (3 vs 2). Also, RF's handle highly-correlated data better, which
            is an implicit assumption with stock predictors. Since we have no categorical variables,
            we don't need to worry about advantages/disadvantages there. 
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def gridSearch():
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(2, 20, num = 1)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [5, 10, 15]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1,2,5,10]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    return (random_grid)

def randomForestTune(finaldata):
    train, ytrain, test, ytest = finaldata
    
    rg = gridSearch()
    rf = RandomForestClassifier()
    
        
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = rg, 
                                   n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    
    rf_random.fit(train, ytrain)
    print('Best Estimators Through Random Grid Search:')
    print(rf_random.best_params_)
    print('\n')
    print('..............................')
    print('Tuned Random Forest Ensemble Output:')
    print('..............................')
    
    best_random = rf_random.best_estimator_
    scores = cross_val_score(best_random,train,ytrain,cv=5)
    for i, score in enumerate(scores):
        print("Validation Set {} score: {}".format(i, score))
    print('\n')
    
    pred = best_random.predict(test)
    ConfusionMat(pred,ytest)
    DrawRoc(best_random,test, ytest)
    
    return(pred,ytest)
    
def RandomForest(finaldata):

    #Wont normalize the data this time, want to have interpretable coefficients maybe
    train, ytrain, test, ytest = finaldata

    tree_model = RandomForestClassifier(n_estimators=100,
                                        criterion="gini",
                                        max_depth=None,
                                        bootstrap=True,
                                        oob_score=True,
                                        random_state=0)
    
    print('..............................')
    print('Random Forest Ensemble Output:')
    print('..............................')
    scores = cross_val_score(tree_model,train,ytrain,cv=5)
    for i, score in enumerate(scores):
        print("Validation Set {} score: {}".format(i, score))
    print('\n')
    
    tree_model.fit(train,ytrain)
    y_pred = tree_model.predict(test)
    
    ConfusionMat(y_pred,ytest)
    DrawRoc(tree_model,test,ytest)

'''Maybe eventually do Logistic Regression and compare how bad it is comparatively'''

def prePlotting(Stock):
    #Aggressive monthly trading strategy
    #Stock: 1 week back -- recent data better says literature
    #Index: 1 quarter back -- older data better says literature
    Stock = getData(Stock,5,20)
    SPY = getData('SPY',90,20)
    
    #Context plots - rolling averages for closing prices
    Exploratory_Plot(SPY)
    Exploratory_Plot(Stock)   
    
    
    import seaborn as sns
    corrmat1 = SPY.corr()
    f, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(corrmat1)
 
    
    #Drops our unneeded variables
    del Stock['WilliamsR']
    del SPY['ROC']
    del SPY['Disparity']
    
    return(Stock, SPY)

def modelFit(Stock, SPY):
    
    #Get Data for different methods
    NormalizedData = finalData(SPY,Stock,1) 
    RegularData = finalData(SPY,Stock,0)
    
    #Base SVM
    SVM(NormalizedData)
    #Tuned SVM
    svm_pred = svc_param_selection(NormalizedData, 5)
    #Base Case
    RandomForest(RegularData)
    #Tune Case via Grid Search
    rf_pred,ytest = randomForestTune(RegularData)
    return(rf_pred, svm_pred)
    
def tradingStrategy(pred, Stock):
    
    #x and y coords appropriately
    x = [Stock.index[i] for i in range(0,len(Stock),20)]
    y = [Stock.Close[i] for i in range(0,len(Stock),20)]
    
    y_pred = [pred[i] for i in range(0,len(pred),20)]
    
    #Color Mapping
    colorMap = {-1.0:"r",1.0:"b",0.0:"y"}
    c = [colorMap[y_pred[i]] for i in range(len(y_pred))]
    c.append('r')
    
    #Setting up plot and Closing Prices
    fig, ax = plt.subplots(figsize=(16,9))  
    plt.plot()
    plt.plot(Stock.Close, c = "g")
    
    #Scatter
    plt.scatter(x,y, c = c, s=55)
    
    #Labels
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing price ($)')
    ax.legend()
    plt.rc('grid',linestyle='dashdot',color='grey')
    plt.grid()  
    
    #Output for finding total profits
    x = np.asarray(x).ravel()[:-1]
    y = np.asarray(y).ravel()[:-1]
    pro = np.transpose(np.vstack([x,y,y_pred]))
    
    return(pro)


def main(ticker):
    Stock, SPY = prePlotting(ticker)
    rf_pred, svm_pred = modelFit(Stock,SPY)

    testStock = Stock[Stock.index >= '2016-01-01']
    profit_mat = tradingStrategy(rf_pred,testStock)
    
    #Calculates total predicted profits in testing set
    shares_owned, position_cost, profits = np.zeros(len(profit_mat)+1), np.zeros(len(profit_mat)+1), np.zeros(len(profit_mat)+1)
    for i in range(0,len(profit_mat)):
        if profit_mat[i,2] > 0:
            shares_owned[i] = shares_owned[i-1] + 1
            position_cost[i] = position_cost[i-1] + profit_mat[i,1]
        
        if profit_mat[i,2] < 0:
            profits[i] = shares_owned[i-1]*profit_mat[i,1] - position_cost[i-1]
            shares_owned[i], position_cost[i] = 0, 0
            
    dca_profits = sum([sum(profit_mat[36,1]-profit_mat[i,1]) for i in range(0,len(profit_mat))])
    
    print(profits)
    print(dca_profits)
    
    

main('AAPL')
main('AMD')
main('T')



'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
"Correlation Plotting
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
      is fine, so we'll leave that for SPY."
'''

    
    
    
    
    
    