#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:59:47 2020

@author: sandeepkompella
"""

import turicreate as tc
products = tc.SFrame('amazon_baby')
#print(products.column_names())


#Find the product that has the highest count of ratings based on the name
products.groupby('name', operations={'count':tc.aggregate.COUNT()}).sort('count',ascending=False)
products['word_count'] = tc.text_analytics.count_words(products['review'])
products = products[products['rating']!= 3]
products['sentiment'] = products['rating'] >= 4
#print(products)

### Added
selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']
for word in selected_words:
    products[word] = products['word_count'].apply(lambda word_count_dic: word_count_dic[word] if word in word_count_dic.keys() else 0)
    print("before")
    print(word, products[word].sum())

### Added
    
# Train our sentiment classifier
train_data,test_data = products.random_split(.8,seed=0)

### Added
sentiment_model = tc.logistic_classifier.create(train_data, target='sentiment', features=['word_count'], validation_set=test_data)
selected_words_model = tc.logistic_classifier.create(train_data,target='sentiment', features=selected_words, validation_set=test_data)
products['predicted_sentiment'] = selected_words_model.predict(products, output_type = 'probability')

coeff = selected_words_model.coefficients.sort('value',ascending=False)
print("before")
print("The Coeffs are:", coeff)
print("The Coeffs are:", coeff.print_rows(num_rows=12))


### Added


sentiment_model.evaluate(test_data, metric='roc_curve')
selected_words_model.evaluate(test_data, metric='roc_curve')



# predictions = selected_words_model.predict(test_data)
# predictions.eval

giraffe_reviews = products[products['name']== 'Vulli Sophie the Giraffe Teether']


baby_champ_reviews = products[products['name'] == 'Baby Trend Diaper Champ']

baby_champ_reviews['predicted_sentiment'] = selected_words_model.predict(baby_champ_reviews, output_type='probability')
baby_champ_reviews = baby_champ_reviews.sort('rating', ascending=False)
print("before")
print(" The selected_words_model.predict is : ")
print(selected_words_model.predict(baby_champ_reviews[0:1], output_type='probability'))

baby_champ_reviews['predicted_sentiment'] = sentiment_model.predict(baby_champ_reviews, output_type='probability')
baby_champ_reviews = baby_champ_reviews.sort('rating', ascending=False)
print("before")
print(" The sentiment_model.predict is : ")
print(sentiment_model.predict(baby_champ_reviews[0:1], output_type='probability'))










