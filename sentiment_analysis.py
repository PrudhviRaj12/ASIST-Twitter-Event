
# coding: utf-8

# In[ ]:

import os
import numpy as np
#os.chdir('Twit')


# ###Importing Data File

# In[ ]:

data = open('C:/Users/Ajinkya/Desktop/ASIST/Twit/training.txt', 'r',encoding="utf8").read().splitlines()


# ###Structuring Data to Further Analysis

# In[ ]:

tweets = []
targets = []
for d in data:
    tweets.append(d[2:])
    targets.append(int(d[0]))

tweets = np.array(tweets)
targets = np.array(targets)


# Random Selector to seperate some examples (1000)
# chosen at random for testing (these will not be used for training)
# 
# Since we are doing this pretty naively, there might be a duplicate
# examples in the test set, but for now, we can ignore that

# In[ ]:

random_selector = np.random.randint(0, len(data), 1000)


# ###Selecting Test Data based on the random indices generated

# In[ ]:

test_data = tweets[random_selector]
test_targets = targets[random_selector]


# ###Collecting Training Data

# In[ ]:

training_data = []
training_targets = []

for d in range(0, len(data)):
    if d not in random_selector:
        training_data.append(data[d][2:])
        training_targets.append(int(data[d][0]))

training_data = np.array(training_data)
training_targets = np.array(training_targets)


# ###Counting words across all the tweets, irrespective of their class.

# In[ ]:

word_count = dict()
for tweet in training_data:
    for word in tweet.split():
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] +=1


# ###Calculating the probability of a word occuring in a tweet, irrespective of the class

# In[ ]:

word_prob = dict()
total_words = sum(word_count.values()) + 0.0

for word in word_count:
    word_prob[word] = word_count[word]/total_words


# ###Seperating Positive and Negative Classes Positive = 1, Negative = 0

# In[ ]:

positive_class = np.where(training_targets == 1)[0]
negative_class = np.where(training_targets == 0)[0]

positive_tweets = training_data[positive_class]
negative_tweets = training_data[negative_class]


# ###Word Occurences in Positive Tweets

# In[ ]:

positive_class_count = {}
for tweet in positive_tweets:
    for word in tweet.split():
        if word not in positive_class_count:
            positive_class_count[word] = 1
        else:
            positive_class_count[word] +=1


# ###Word Occurences in Negative Tweets

# In[ ]:

negative_class_count = {}
for tweet in negative_tweets:
    for word in tweet.split():
        if word not in negative_class_count:
            negative_class_count[word] = 1
        else:
            negative_class_count[word] +=1


# ###To maintain balance to reduce errors while testing, for every word that is in the entire vocabulary, but not in seperate class vocabulary, we add the word that class vocabulary and give it a count of zero

# In[ ]:

for word in word_prob:
    if word not in positive_class_count:
        positive_class_count[word] = 0
    if word not in negative_class_count:
        negative_class_count[word] = 0


# ###Small Value to avoid divide by zero error

# In[ ]:

epsilon = 0.00001


# ###Conditional Probabilites of a word occuring given the tweet is positive or negative

# In[ ]:

positive_class_prob, negative_class_prob = {}, {}    
for word in word_prob:
    summer = positive_class_count[word] + negative_class_count[word]
    positive_class_prob[word] = (positive_class_count[word]/summer) +epsilon
    negative_class_prob[word] = (negative_class_count[word]/summer) +epsilon


# ###Test Sample
# 
# To avoid long multiplications that would eventually
# combine to zero, we use logarithm of the value.
# 
# log (a x b) = log(a) + log(b), which converts multiplication
# into addition, thereby reducing the possiblity of the value extending to zero.

# In[ ]:

sample = test_data[0]
target_needed = test_targets[0]

positive = 0
negative = 0
for word in sample.split():
    if word in positive_class_prob:
        positive += (np.log(word_prob[word]) + np.log(positive_class_prob[word]))
    if word in negative_class_prob:
        negative += (np.log(word_prob[word]) + np.log(negative_class_prob[word]))
        
predicted_sentiment = int(positive > negative)
print (sample, target_needed)
if predicted_sentiment == 1:
    print ("Positive")
else:
    print ("Negative")


# ###Finding the accuracy of our model

# In[ ]:

correct_predictions = 0
for t in range(0, len(test_data)):    
    sample = test_data[t]
    target_needed = test_targets[t]
    
    positive = 0
    negative = 0
    for word in sample.split():
        if word in positive_class_prob:
            positive += (np.log(word_prob[word]) + np.log(positive_class_prob[word]))
        if word in negative_class_prob:
            negative += (np.log(word_prob[word]) + np.log(negative_class_prob[word]))
            
    predicted_sentiment = int(positive > negative)
    
    if predicted_sentiment == target_needed:
        correct_predictions += 1
        
print ("Number of Correct Predictions: " + str(correct_predictions))
print ("Accuracy: " + str(correct_predictions/(len(test_data) + 0.0)))


# In[ ]:

from collections import defaultdict

class_1_count = dict()
for tweet in tweets_party1:
    for word in tweet.split():
        if word not in class_1_count:
            class_1_count[word] = 1
        else:
            class_1_count[word] +=1
            
class_0_count = dict()
for tweet in tweets_party0:
    for word in tweet.split():
        if word not in class_0_count:
            class_0_count[word] = 1
        else:
            class_0_count[word] +=1
    
class_0_cond, class_1_cond = {}, {}
for word in word_count_2:
    if word in class_0_count and word in class_1_count:
        summer = class_0_count[word] + class_1_count[word] + 0.0
        print (word, summer)
        class_0_cond[word] = class_0_count[word]/summer
        class_1_cond[word] = class_1_count[word]/summer


# In[ ]:



