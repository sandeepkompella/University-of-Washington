#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 20:35:29 2020

@author: sandeepkompella
"""

import turicreate as tc
people = tc.SFrame('people_wiki')
obama = people[people['name'] == 'Barack Obama']
obama['word_count'] = tc.text_analytics.count_words(obama['text'])
#sort the word counts for the obama article
obama_word_count_table =obama[['word_count']].stack('word_count',new_column_name = ['word','count'])
sorted_obama_word_count_table = obama_word_count_table.sort('count', ascending=False)
#Begin Sandeep for Quiz
john = people[people['name'] == "Elton John"]
john['word_count'] = tc.text_analytics.count_words(john['text'])
john_word_count_table = john[['word_count']].stack('word_count', new_column_name = ['word', 'count'])
sorted_john_word_count_table = john_word_count_table.sort('count',ascending=False)
print("sorted_john_word_count_table")
print(sorted_john_word_count_table)
#End Sandeep for Quiz

#compute TF-IDF. Depends on the whole corpus not on one article
people['word_count'] = tc.text_analytics.count_words(people['text'])
tfidf = tc.text_analytics.tf_idf(people['word_count']) 
people['tfidf'] = tc.text_analytics.tf_idf(people['text'])
#Examine the TF-IDF for the Obama article
obama = people[people['name'] == "Barack Obama"]
obama[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)

#Begin Sandeep for Quiz
john = people[people['name'] == "Elton John"]
john[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)
print(john[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False))

paul = people[people['name'] == "Paul McCartney"]
paul[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)
print(john[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False))
#End Sandeep for Quiz

#Compute the distances of Elton to Victoria Beckham and Paul McCartney
#Begin Sandeep for Quiz
beckham = people[people['name'] == "Victoria Beckham"]
print("cosine distances john victoria")
print(tc.distances.cosine(john['tfidf'][0], beckham['tfidf'][0])) # the lower the cosine the better similarity
print("cosine distances john paul")
paul = people[people['name'] == "Paul McCartney"]

print(tc.distances.cosine(john['tfidf'][0], paul['tfidf'][0])) # the lower the cosine 
#End Sandeep for Quiz

# #Build NN for Doc retrieval
# knn = tc.nearest_neighbors.create(people,features=['tfidf'],label='name')
# #applying NN model for retrieval and who is nearest to Obama
# print(knn.query(obama))
# #Query for people similar to Swift
# swift = people[people['name'] == "Taylor Swift"]
# print(knn.query(swift))

#Build the model with the word counts
#Begin Sandeep for Quiz

knn1 =tc.nearest_neighbors.create(people, features=['word_count'],distance='cosine', label='name') 
knn2 =tc.nearest_neighbors.create(people,features=['tfidf'],distance='cosine', label='name')
print("Knn2 for John and Beckham")
print(knn1.query(john))
print(knn1.query(beckham))
print("Knn1 for John and Beckham")
print(knn2.query(john)) 
print(knn2.query(beckham)) 
#End Sandeep for Quiz


