    
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
    
    #Result: ML method outperforms buy-and-hold for sideways/downward stocks
    print(profits)
    print(dca_profits)
    
    

#main('AAPL')
#main('AMD')
#main('T')


    
    
    