#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:56:27 2020

@author: sandeepkompella
"""

import turicreate as tc


image_train = tc.SFrame('image_train_data')
image_test = tc.SFrame('image_test_data')

print(image_train['image'])
print(image_train['image'])
# train the classifier on the raw image pixels


raw_pixel_model = tc.logistic_classifier.create(image_train,target='label', features=['image_array'])
print(image_test[0:3]['image'].explore() )   
print(image_test[0:3]['label'] )
print(raw_pixel_model.predict(image_test[0:3]))   
## Evaluate the raw pixel model on the test data
raw_pixel_model.evaluate(image_test)

print(len(image_train))
print(image_train)

deep_features_model = tc.logistic_classifier.create(image_train, target='label', features = ['deep_features'])                                                                           
print(image_test[0:3]['image'].explore())
print(deep_features_model.predict(image_test[0:3]))
print(deep_features_model.evaluate(image_test))                                                                                      
                                                                                      
                                                                                      