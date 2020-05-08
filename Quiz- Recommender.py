#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:21:03 2020

@author: sandeepkompella
"""

import turicreate as tc

song_data = tc.SFrame('song_data')
#print(song_data.column_names())
#print(len(song_data))
#get the total no of unique users and thier count, use the len function to find the count
#print(song_data['user_id'].unique())
#['user_id', 'song_id', 'listen_count', 'title', 'artist', 'song']
#print(len(song_data['user_id'].unique()))
kw = song_data[song_data['artist'] == 'Kanye West']
print(" the unique count for Kanye West is ", len(kw['user_id'].unique()))

ff = song_data[song_data['artist'] == 'Foo Fighters']
print(" the unique count for Foo Fighters is ",  len(ff['user_id'].unique()))

ts = song_data[song_data['artist'] == 'Taylor Swift']
print(" the unique count for Taylor Swift is ", len(ts['user_id'].unique()))
lg = song_data[song_data['artist'] == 'Lady GaGa']
print(" the unique count for Lady Gaga is ", len(lg['user_id'].unique()))

groupbyaggregate = song_data.groupby(key_column_names='artist', operations={'total_count': tc.aggregate.SUM('listen_count')})
print("The most popular artists are \n", groupbyaggregate.sort('total_count', ascending = False))
print("The least popular artists are \n", groupbyaggregate.sort('total_count', ascending = True))

train_data, test_data = song_data.random_split(0.8,seed=0)
personalized_model = tc.item_similarity_recommender.create(train_data, user_id='user_id',item_id='song')
subset_test_users = test_data['user_id'].unique()[0:10000]
onesongrecommended = personalized_model.recommend(subset_test_users,k=1)
onesongrecommendedcount = onesongrecommended.groupby(key_column_names='song', operations={'count': tc.aggregate.COUNT()})
print(onesongrecommendedcount.sort('count',ascending=False))

