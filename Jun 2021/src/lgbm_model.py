# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 11:23:13 2021

@author: Abu Bakar
"""
#public score = 1.75331

import pickle
import pandas as pd
from sklearn.metrics import log_loss
from data_preprocessing import x_train, x_test, y_train, y_test
from input import test, sample_sub
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(num_leaves = 10,
                      learning_rate = 0.01,
                      n_estimators=1000,
                      min_child_samples=50,
                      reg_lambda = 3.5,
                      random_state = 31)
lgbm.fit(x_train, y_train)
lgbm_prediction = lgbm.predict_proba(x_test) 
print(log_loss(y_test, lgbm_prediction)) #1.7480

pickle.dump(lgbm, open('lgbm model.h5', 'wb'))

#predicting on test set
lgbm_preds = lgbm.predict_proba(test)

# output to csv
lgbm_submission = pd.concat([sample_sub.id, pd.DataFrame(lgbm_preds, 
        columns=["Class_1", "Class_2", "Class_3","Class_4", "Class_5", "Class_6", "Class_7", "Class_8", "Class_9"])], axis = 1)

lgbm_submission.to_csv('lgbm_submission.csv', index = False)