# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 20:51:48 2021

@author: Abu Bakar
"""
import pandas as pd
import numpy as np

catboost_file = pd.read_csv('cat_submission.csv')
histgradient_file = pd.read_csv('hist_submission.csv')
lgbm_file = pd.read_csv('lgbm_submission.csv')
xgb_file = pd.read_csv('xgb_submission.csv')

blend = catboost_file.copy()
columns = catboost_file.columns[1:]

blend[columns] = 0.95*0.6*histgradient_file[columns] + 0.95*0.4*lgbm_file[columns] + 0.05*xgb_file[columns] + 0.95*0.5*catboost_file[columns]
blend.to_csv('blend_sub_1.csv', index=False) # public score: 1.75169

blend[columns] = 0.90*0.5*histgradient_file[columns] + 0.95*0.3*lgbm_file[columns] + 0.02*xgb_file[columns] + 0.95*0.5*catboost_file[columns]
blend.to_csv('blend_sub_2.csv', index=False) # public score: 1.75127

blend[columns] = 1.01*catboost_file[columns]-0.01*blend[columns]
blend.to_csv('blend_sub_3.csv', index=False) # public score: 1.75048

blend[columns] = 0.25*histgradient_file[columns] + 0.25*lgbm_file[columns] + 0.15*xgb_file[columns] + 0.35*catboost_file[columns]
blend.to_csv('blend_sub_4.csv', index=False) # public score: 1.75141

blend[columns] = 1.05*catboost_file[columns]-0.05*blend[columns]
blend.to_csv('blend_sub_5.csv', index=False) # public score: 1.75052



def generation(main, support, coeff): 
    sub1  = support.copy()
    sub1v = sub1.values    
    
    sub2  = main.copy() 
    sub2v = sub2.values
       
    imp  = main.copy()    
    impv = imp.values  
    NCLASS = 9
    number = 0
    
    for i in range (len(main)):               
        
        row1 = sub1v[i,1:]
        row2 = sub2v[i,1:]
        row1_sort = np.sort(row1)
        row2_sort = np.sort(row2) 
        
        row = (row2 * coeff) + (row1 * (1.0 - coeff))
        row_sort = np.sort(row)        
        
        for j in range (NCLASS):             
            if ((row2[j] == row2_sort[8]) and (row1[j] != row1_sort[8])):                                
                row = row2 
                number = number + 1            
        
        impv[i, 1:] = row
    
    imp.iloc[:, 1:] = impv[:, 1:]
                                       
    return imp

submission = generation(catboost_file, lgbm_file, 0.40)
submission.to_csv("blend_sub_6.csv",index=False) # public score: 1.75103