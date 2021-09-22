# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 00:36:39 2021

@author: Abu Bakar
"""

from input import train, test

test.drop('id', axis = 1, inplace = True)

#label encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
train['target'] = encoder.fit_transform(train['target'])

#creating dependent and independent matrix of features
x = train.drop(['id', 'target'], axis = 1)
y = train['target']

#splitting dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)