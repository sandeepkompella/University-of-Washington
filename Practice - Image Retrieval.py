#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:56:27 2020

@author: sandeepkompella
"""

import turicreate as tc

image_data = tc.SFrame('image_train_data')
print(image_data['image'].explore())
print(image_data)

sketch = tc.Sketch(image_data['label'])



knn_model = tc.nearest_neighbors.create(image_data, features = ['deep_features'],label = 'id')
cat = image_data[18:19]
print(cat['image'].explore())
knn_model.query(cat)
def get_images_from_ids(query_result):
    return image_data.filter_by(query_result['reference_label'],'id')
cat_neighbors = get_images_from_ids(knn_model.query(cat))
print(cat_neighbors['image'].explore())                                                                                      
car = image_data[8:9]
print(car['image'].explore())
get_images_from_ids(knn_model.query(car))['image'].explore()
show_neighbors = lambda i: get_images_from_ids(knn_model.query(image_data[i:i+1]))['image'].explore()
print(show_neighbors(8))
print(show_neighbors(26))
print(show_neighbors(500))                                                                                      
                                                                                      