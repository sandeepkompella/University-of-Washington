#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:02:14 2020

@author: sandeepkompella
"""

import turicreate as tc

song_data = tc.SFrame('song_data')
print(song_data.column_names())
print(len(song_data))
#get the total no of unique users and thier count, use the len function to find the count
#print(song_data['user_id'].unique())
print(len(song_data['user_id'].unique()))

train_data, test_data = song_data.random_split(0.8,seed=0)
# create a song recommender using the simple popularity model
popularity_model = tc.popularity_recommender.create(train_data,user_id='user_id',item_id='song')
#make some predictions now for user 0 and 1
#print(popularity_model.recommend(users = [song_data['user_id'][0]]))
#print(popularity_model.recommend(users = [song_data['user_id'][1]]))
# create a song recommender using the personalization
personalized_model = tc.item_similarity_recommender.create(train_data, user_id='user_id',item_id='song')
print(personalized_model.recommend(users = [song_data['user_id'][0]]))
print(personalized_model.recommend(users = [song_data['user_id'][1]]))
#create a song similar one to with or Without you using the personalization
print(personalized_model.get_similar_items(['With Or Without You - U2']))
print(personalized_model.get_similar_items(['Chan Chan (Live) - Buena Vista Social Club']))

# Do a compare between personalized model with the popularity model using the precision recall curve
model_performance = tc.recommender.util.compare_models(test_data, [popularity_model, personalized_model], user_sample=.05)
