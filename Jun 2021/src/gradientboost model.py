# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 02:09:13 2021

@author: Abu Bakar
"""

import pickle
from sklearn.metrics import log_loss
from data_preprocessing import x_train, x_test, y_train, y_test
from input import test, sample_sub
from sklearn.ensemble import GradientBoostingClassifier


gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
gbc_prediction = gbc.predict_proba(x_test) 
print(log_loss(y_test, gbc_prediction)) #1.75224


gbc = GradientBoostingClassifier(learning_rate = 0.01,
                                 n_estimators = 1000,
                                 min_samples_leaf = 2,
                                 min_samples_split = 2,
                                 max_depth = 6)
gbc.fit(x_train, y_train)
gbc_prediction = gbc.predict_proba(x_test) 
print(log_loss(y_test, gbc_prediction)) #1.75224



