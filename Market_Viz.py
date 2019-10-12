# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:02:52 2019

@author: Brian Cote
"""

class Plots(object):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, confusion_matrix
    
    def __init__(self):
        pass
    
    def rollingAverage(self,data):
        tick = data.Close.loc[:,]
        
        # Calculate the moving averages of the closing prices
        monthly_rolling = tick.rolling(window=20).mean()
        quarter_rolling = tick.rolling(window=90).mean()
        yearly_rolling = tick.rolling(window=270).mean()
        
        fig, ax = self.plt.subplots(figsize=(16,9))    
        
        self.plt.plot(tick.index, tick)
        self.plt.plot(monthly_rolling.index, monthly_rolling, label='Monthly Rolling Average')
        self.plt.plot(quarter_rolling.index, quarter_rolling, label='Quarterly Rolling Average')
        self.plt.plot(yearly_rolling.index, yearly_rolling, label='Yearly Rolling Average')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Adjusted closing price ($)')
        ax.legend()
        self.plt.rc('grid',linestyle='dashdot',color='grey')
        self.plt.grid()  
        
    def ConfusionMat(self,pred,ytest):
        #Predictions
        truthArray = []
        for i in range(len(ytest)):
        		# print "prediction: %d\tactual: %d" % (predictions[i], Y_test[i])
        		truthArray.append(1. if pred[i] == ytest[i] else 0.)
        
        	# print len(truthArray)
        print('The Total Predictive Accuracy Is:',100*sum(truthArray)/len(ytest))
        
        #Prints confusion matrix
        cm = self.confusion_matrix(ytest,pred)
        spe = cm[0,0]/sum(cm[0,])
        sen = cm[1,1]/sum(cm[1,])
        print("Sensitivity is",100*spe,"% and Specificity is",100*sen,"%")
    
    
    def DrawRoc(self,model,test,ytest):
        y_prob = model.predict_proba(test)
        true_probability_estimate = y_prob[:,1]
        
        fpr,tpr,threshold = self.roc_curve(ytest,true_probability_estimate)
        area = self.auc(fpr,tpr)
        self.plt.figure()
        self.plt.plot(fpr,tpr,linewidth = 2.0,label = "ROC curve (Area= %0.2f)" % area)
        self.plt.plot([0,1],[0,1],"r--")
        self.plt.xlabel("False Postive Rate")
        self.plt.ylabel("True Positive Rate")
        self.plt.legend(loc = "lower right")
        self.plt.show(block = False)

