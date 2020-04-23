# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:50:43 2020

@author: rivah
"""

import pandas as pd  
import numpy as np
import glob
import os


# we're going to read in a dataset of names that already have a gender assigned to them using pandas
# Pandas can easily read csv files into a 'dataframe'
# The list of names comes from the US census data (which is available for a few 19c years)
# Feel free to modify this dataset to include whatever names you wish.

names = pd.read_csv('names_dataset.csv')



# Get the data out of the dataframe into a numpy matrix and keep only the name and gender columns
# Saving the data as a matrix makes it easier to access later
names = names.as_matrix()[:, 1:]


 
# We're using 80% of the data for training
TRAIN_SPLIT = 0.8



#This is a function that will split a name into its different parts

def gender_features(name):
    name = name.lower()
    return {
        'first-letter': name[0], # First letter
        'first2-letters': name[0:2], # First 2 letters
        'first3-letters': name[0:3], # First 3 letters
        'last-letter': name[-1],
        'last2-letters': name[-2:],
        'last3-letters': name[-3:],
    }
    
    
    
# Vectorize the features function because the rest of our tools require lists
features = np.vectorize(gender_features)



    
# Extract the features for the whole dataset
X = features(names[:, 0]) # X contains the features
 


# Get the gender column
y = names[:, 1]           # y contains the targets


 


# now we're going to actually begin training our data
# shuffle the data so its not in any particular order
# We need to split the data into a testing set and a training set
from sklearn.utils import shuffle
X, y = shuffle(X, y)
X_train, X_test = X[:int(TRAIN_SPLIT * len(X))], X[int(TRAIN_SPLIT * len(X)):]
y_train, y_test = y[:int(TRAIN_SPLIT * len(y))], y[int(TRAIN_SPLIT * len(y)):]
 




# Machine learning framework for Python
from sklearn.feature_extraction import DictVectorizer


# Because classifiers don't work with characters, we have to transform the names into a format the the program
                    #can understand
vectorizer = DictVectorizer()
vectorizer.fit(X_train)




#The code will be making gender determinations using a decision tree

from sklearn.tree import DecisionTreeClassifier
 
clf = DecisionTreeClassifier()
clf.fit(vectorizer.transform(X_train), y_train)



# Now, we need to get the data from the JSON file produced by BookNLP

import json  # This library will let us read in JSON files directly into our code

with open("YOUR_FILENAME_HERE", "r") as read_file:
    data = json.load(read_file) # We're going to save the data as "data"


# Now, the function is going to search through the data list and look for the field "n" which stands for "name" in BookNLP
def find_names(data):
 
    if 'n' in data:
        yield data['n']
    for k in data:
        if isinstance(data[k], list):  #if item 'k' is in the data list
            for i in data[k]:
                for j in find_names(i):
                    yield j


#Locate instances of speech in the BookNLP results
def find_speech(data):
 
    if 'speaking' in data:
        yield data['speaking']
    for k in data:
        if isinstance(data[k], list):  #if item 'k' is in the data list
            for i in data[k]:
                for j in find_speech(i):
                    yield j
                    


#List of dialogue
speech= list(find_speech(data))



# Save the names into a list variable
nameslist = list(find_names(data))


#Classify the entire list of names
genders = (clf.predict(vectorizer.transform(features(nameslist))))
i = 0
q = 0
female_names = []
male_names = []

#If you print the list of female names, this will tell you if there even exists a woman who is named




import json
import glob
import os
from six import iteritems

from collections import defaultdict

#nested lookup functions adapted from https://github.com/russellballestrini/nested-lookup/blob/master/nested_lookup/nested_lookup.py

def nested_lookup(key, document, wild=False, with_keys=False):
    """Lookup a key in a nested document, return a list of values"""
    if with_keys:
        d = defaultdict(list)
        for k, v in _nested_lookup(key, document, wild=wild, with_keys=with_keys):
            d[k].append(v)
        return d
    return list(_nested_lookup(key, document, wild=wild, with_keys=with_keys))

def _nested_lookup(key, document, wild=False, with_keys=False):
    """Lookup a key in a nested document, yield a value"""
    if isinstance(document, list):
        for d in document:
            for result in _nested_lookup(key, d, wild=wild, with_keys=with_keys):
                yield result

    if isinstance(document, dict):
        for k, v in iteritems(document):
            if key == k or (wild and key.lower() in k.lower()):
                if with_keys:
                    yield k, v
                else:
                    yield v
            elif isinstance(v, dict):
                for result in _nested_lookup(key, v, wild=wild, with_keys=with_keys):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in _nested_lookup(key, d, wild=wild, with_keys=with_keys):
                        yield result


attributes= {} #dictionary
j = 0
y = 0
p = 0
d = 0
u = 0
m = 0
f = 0
final_gender = ""
keys = []
for i in data:
    while j < (len(data['characters'])):
        key = data[i][j]

        name = nested_lookup('n', key)
        gender = clf.predict(vectorizer.transform(features(name)))
        
        #if name is preceded with mr. or mrs. then change the gender
        if gender[0] == 'F':
            if name[0][:3] == 'Mr.':
                gender = 'M'
                male_names.append(name[q])
            else:
                female_names.append(name[q])
        if gender[0] == 'M':
            if name[0][:3] == 'Mrs':
                gender[0] = 'F'
                female_names.append(name[q])
            else: 
                male_names.append(name[q])
                    
        
        if gender[0] == 'F':
            attributes.update({y : ([{'name': name[0]},
                                  {'gender': gender[0]},
                                  {'speaking': nested_lookup('w', nested_lookup('speaking', key))}
                ])})
            keys.append(y)
        p = p + 1
        y = y + 1
        j = j + 1 


print(female_names)
print(male_names)

#Bag of male terms
male_terms = ["father", "Father", "brother", "Brother", "husband", "Husband", "Son", "son", "he", "He",
              "his", "His", "him", "Him", "Sir", "sir", "papa", "Papa"]


################################SPEAKING ABOUT MEN###########################
about_men = male_names + male_terms


i = 0
while i < (len(keys)):
    q = 0
    while q < len(about_men):
        res= {k: [v.remove(x) for x in v if about_men[q] in x] for k, v in attributes[keys[i]][2].items()}
        q = q + 1
    i = i + 1


##############################SPEAKING TO WOMEN##################################################
female_terms = ["her", "Her", "hers", "Hers", "her's", "Her's", "madam", "Madam", "Miss", "miss", "she", "She", 
                "mother", "Mother", "sister", "Sister", "mama", "Mama", "wife", "Wife"]
to_women = female_names + female_terms
print(to_women)
i = 0

##change this so that if ANY of the female names or pronouns are in dialogue, we save it
##The pronouns will tell us if a woman is being spoken ABOUT, but sometimes this can still be useful
final = []
while i < (len(keys)):
    q = 0
    while q < len(female_names):
        res= {k: [final.append({str(attributes[keys[i]][0]): x}) for x in v if to_women[q] in x] for k, v in attributes[keys[i]][2].items()}

        q = q + 1
    i = i + 1

#The final results are saved as a python dictionary which you can save as a .json file or convert to csv
print(final)

