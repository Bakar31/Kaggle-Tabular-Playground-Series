# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 16:11:24 2021

@author: Abu Bakar
"""
# public Score: 1.75047

import pickle
import pandas as pd
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier
from data_preprocessing import x_train, x_test, y_train, y_test
from input import test, sample_sub


cat_model = CatBoostClassifier(depth=8,
                             iterations=1000,
                             learning_rate=0.02,                            
                             eval_metric='MultiClass',
                             loss_function='MultiClass', 
                             bootstrap_type= 'Bernoulli',
                             leaf_estimation_method='Gradient',
                             random_state=13)                        

cat_model.fit(x_train, y_train, verbose=100)
cat_prediction = cat_model.predict_proba(x_test)
print(log_loss(y_test, cat_prediction))  #1.7477
pickle.dump(cat_model, open('catboost model.h5', 'wb'))

#predicting on test set
cat_preds = cat_model.predict_proba(test)

# output to csv
cat_submission = pd.concat([sample_sub.id, pd.DataFrame(cat_preds, 
        columns=["Class_1", "Class_2", "Class_3","Class_4", "Class_5", "Class_6", "Class_7", "Class_8", "Class_9"])], axis = 1)

cat_submission.to_csv('cat_submission.csv', index = False)