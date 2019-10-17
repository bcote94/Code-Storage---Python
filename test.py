# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:40:38 2019

@author: Brian Cote
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

X=np.random.random((100,5))
y=np.random.randint(0,2,(100,))

clf = RandomForestClassifier(random_state=1)
s = cross_val_score(RandomForestClassifier(), X,y,scoring='roc_auc', cv=5)
print(s)
##[ 0.57612457  0.29044118  0.30514706]
print(s)
##[ 0.57612457  0.29044118  0.30514706]

