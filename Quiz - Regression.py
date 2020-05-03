#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 21:34:07 2020

@author: sandeepkompella
98102
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import math

sales = pd.read_csv('home_data.csv')
#sales_98039 = sales.loc[sales['zipcode'] == 98039]
sales_98039 = sales[sales['zipcode'] == 98039]
#print(sales_98039)
print(sales_98039['price'].agg('mean'))
#Ans is 2160606.6

condition_one = sales['sqft_living'] > 2000
condition_two = sales['sqft_living'] <= 4000

sales_lt4000 = sales.loc[condition_one & condition_two]
print(sales_lt4000)
print(len(sales_lt4000)/len(sales))
#Ans - 0.42187572294452413

my_features = sales[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']] 
target = sales[['price']]
my_features_train, my_features_test, target_train, target_test = train_test_split(my_features,target,test_size=0.2,random_state = 0,)
model = LinearRegression()
model.fit(my_features_train, target_train)
y_pred = model.predict(my_features_test)
rmsd = np.sqrt(mean_squared_error(target_test, y_pred))      
print("my_Root Mean Square Error \n", rmsd)

advanced_features = sales[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode', 'condition', 'grade', 'waterfront', 'view', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15']]

advanced_features_train, advanced_features_test, target_train, target_test = train_test_split(advanced_features,target,test_size=0.2,random_state = 0)
advanced_model = LinearRegression()
advanced_model.fit(advanced_features_train, target_train)
y_pred1 = advanced_model.predict(advanced_features_test)
rmsd1 = np.sqrt(mean_squared_error(target_test, y_pred1))      
print("advanced_Root Mean Square Error \n", rmsd1)
print(rmsd1-rmsd)

