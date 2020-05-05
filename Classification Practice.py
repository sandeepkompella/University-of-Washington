#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 19:05:38 2020

@author: sandeepkompella
"""

#create the csv file as we are working with Numpy and Pandas and not Turicreate

import turicreate as tc
products = tc.SFrame('amazon_baby')
#print(products.column_names())


#Find the product that has the highest count of ratings based on the name
products.groupby('name', operations={'count':tc.aggregate.COUNT()}).sort('count',ascending=False)

#Giraffe_reviews = products[products['name']== 'Vulli Sophie the Giraffe Teether']
# print(Giraffe_reviews)
# print(Giraffe_reviews['rating'].show())

# Now add a column word_count to sFrame with the words and 
#the frequency of occurance in the review field

products['word_count'] = tc.text_analytics.count_words(products['review'])
#print(products['rating'])

#ignore all 3*  reviews and #positive sentiment = 4-star or 5-star reviews

products = products[products['rating']!= 3]
products['sentiment'] = products['rating'] >= 4
print(products)

# Train our sentiment classifier
train_data,test_data = products.random_split(.8,seed=0)
sentiment_model = tc.logistic_classifier.create(train_data,target='sentiment', features=['word_count'], validation_set=test_data)
products['predicted_sentiment'] = sentiment_model.predict(products, output_type = 'probability')
#print(products)

giraffe_reviews = products[products['name']== 'Vulli Sophie the Giraffe Teether']
#print(giraffe_reviews)


giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)
#print(giraffe_reviews)
giraffe_reviews.tail()

# Show the top 2 - most positive reviews
giraffe_reviews[0]['review']
giraffe_reviews[1]['review']

# Show the top 2 - most Negative reviews
giraffe_reviews[-1]['review']
giraffe_reviews[-2]['review']






