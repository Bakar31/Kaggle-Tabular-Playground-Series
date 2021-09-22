# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 16:28:21 2021

@author: Abu Bakar
"""
# public score: 1.75749

import pickle
import pandas as pd
from sklearn.metrics import log_loss
from data_preprocessing import x_train, x_test, y_train, y_test
from input import test, sample_sub
from xgboost import XGBClassifier

#model
xgb = XGBClassifier(eta = 0.1,
                    gamma = 0.5,
                    max_depth = 6,
                    max_delta_step = 0,
                    reg_lambda = 1,
                    random_state = 31)
xgb.fit(x_train, y_train)
xgb_prediction = xgb.predict_proba(x_test) 
print(log_loss(y_test, xgb_prediction)) #1.75079

#saving the model
pickle.dump(xgb, open('xgb model.h5', 'wb'))

#predicting on test set
xgb_preds = xgb.predict_proba(test)

# output to csv
xgb_submission = pd.concat([sample_sub.id, pd.DataFrame(xgb_preds, 
        columns=["Class_1", "Class_2", "Class_3","Class_4", "Class_5", "Class_6", "Class_7", "Class_8", "Class_9"])], axis = 1)

xgb_submission.to_csv('xgb_submission.csv', index = False)
