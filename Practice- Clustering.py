#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:41:29 2020

@author: sandeepkompella
"""

import turicreate as tc
people = tc.SFrame('people_wiki')
#print(people.head())
#print(len(people))
obama = people[people['name'] == 'Barack Obama']
#print(obama)
#print(obama['text'])
# clooney = people[people['name'] == 'George Clooney']
# print(clooney['text'])
obama['word_count'] = tc.text_analytics.count_words(obama['text'])
#print(obama['word_count'])

#sort the word counts for the obama article
obama_word_count_table =obama[['word_count']].stack('word_count',new_column_name = ['word','count'])
#print(obama_word_count_table)
sorted_obama_word_count_table = obama_word_count_table.sort('count', ascending=False)
print(" The obama_word_count_table: =", sorted_obama_word_count_table)

#compute TF-IDF. Depends on the whole corpus not on one article

people['word_count'] = tc.text_analytics.count_words(people['text'])
tfidf = tc.text_analytics.tf_idf(people['word_count']) 
#print(tfidf)
people['tfidf'] = tc.text_analytics.tf_idf(people['text'])
#Examine the TF-IDF for the Obama article
obama = people[people['name'] == "Barack Obama"]
obama[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)
#print(obama)
#Compute the distances of Obama to Clinton and Beckham
clinton = people[people['name'] == "Bill Clinton"]
beckham = people[people['name'] == "David Beckham"]
print(tc.distances.cosine(obama['tfidf'][0], clinton['tfidf'][0])) # the lower the cosine the better similarity
print(tc.distances.cosine(obama['tfidf'][0], beckham['tfidf'][0])) # the lower the cosine the better similarity

#Build NN for Doc retrieval

knn = tc.nearest_neighbors.create(people,features=['tfidf'],label='name')
#applying NN model for retrieval and who is nearest to Obama
print(knn.query(obama))

#Query for people similar to Swift

swift = people[people['name'] == "Taylor Swift"]
print(knn.query(swift))
jolie = people[people['name'] == "Angelina Jolie"]
print(knn.query(jolie))        
