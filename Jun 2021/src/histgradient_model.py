# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 02:09:13 2021

@author: Abu Bakar
"""
# public score: 1.75439

import pickle
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from data_preprocessing import x_train, x_test, y_train, y_test
from input import test, sample_sub


hist_model = HistGradientBoostingClassifier(max_iter=1000, 
                                         learning_rate=0.01, 
                                         max_depth=30, 
                                         min_samples_leaf=20, 
                                         max_leaf_nodes=10,
                                         random_state=123)

hist_model.fit(x_train, y_train)
hist_prediction = hist_model.predict_proba(x_test)
print(log_loss(y_test, hist_prediction)) #1.74883

#saving the model
pickle.dump(hist_model, open('hist gradient model.h5', 'wb'))

#predicting on test set
hist_preds = hist_model.predict_proba(test)

# output to csv
hist_submission = pd.concat([sample_sub.id, pd.DataFrame(hist_preds, 
        columns=["Class_1", "Class_2", "Class_3","Class_4", "Class_5", "Class_6", "Class_7", "Class_8", "Class_9"])], axis = 1)

hist_submission.to_csv('hist_submission.csv', index = False)
