
# coding: utf-8

# In[ ]:

import os
import sys
from tweepy import API
from tweepy import OAuthHandler
from tweepy import Cursor
import json

consumer_key = #Consumer key
consumer_secret = #consumer secret

access_key = #access key
access_secret = #access secret

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

client = API(auth)


# ##Home Timeline

# In[ ]:

for status in Cursor(client.home_timeline).items(20):
    print (status.text)
    print ("\n")


# ##Hash Tags

# In[ ]:

for status in Cursor(client.search, q = "#MeToo").items(20):
    print (status.text)
    print ("\n")


# ##Keywords

# In[ ]:

for status in Cursor(client.search, q = 'Deep Learning').items(20):
    print (status.text)
    print ("\n")


# ##Collect Tweets For Further Analysis

# In[ ]:

tweets_list = []
c = 0 #counter
for status in Cursor(client.search, q = "#Metoo").items(2000):
    c+=1
    print (c)
    tweets_list.append(status)


# ##Collect the texts from the tweet data

# In[ ]:

tweets_text = []
for t in tweets_list:
    tweets_text.append(t.text)


# ##Count Word Frequencies

# In[ ]:

word_count_dict = {}
for text in tweets_text:
    for word in text.split():
        if word not in word_count_dict:
            word_count_dict[word] = 1
        else:
            word_count_dict[word] = word_count_dict[word] + 1

print ("Size of the vocabulary: " + str(len(word_count_dict)))


# ##Sorting Word Frequencies to Find most frequent words

# In[ ]:

import operator
sorted_frequencies = sorted(word_count_dict.items(), key = operator.itemgetter(1), reverse = True)
print (sorted_frequencies[0:10])


# ##Inverse Document Frequency

# In[ ]:

document_frequency = {}
for term in word_count_dict:
    occurence_count = 0
    for text in tweets_text:
        if term in text:
            occurence_count +=1
    document_frequency[term] = occurence_count


# ##Sorting the Document Frequencies

# In[ ]:

sorted_doc_frequencies = sorted(document_frequency.items(), key = operator.itemgetter(1), reverse = True)
print (sorted_doc_frequencies[0:10])


# ##Calculating Term Frequency - Inverse Document Frequency

# In[ ]:

tfidf_scores = {}
vocabulary = word_count_dict.keys()

for word in vocabulary:
    tfidf_scores[word]  = word_count_dict[word]/ (document_frequency[word] + 0.00001)


# ##Sorting the TFIDF Scores

# In[ ]:

sorted_tfidf_scores = sorted(tfidf_scores.items(), key = operator.itemgetter(1), reverse = True)
print (sorted_tfidf_scores[0:10])


# #Word Clouds

# ##Simple Word Cloud
# 
# ```
# Installing wordcloud
# 
# 1. download: http://www.lfd.uci.edu/~gohlke/pythonlibs/#wordcloud
# 
# 2. cd to the file path
# 
# 3. Run this command python -m pip install <filename>
# ```
# 

# ###No Preprocessing

# In[ ]:

import wordcloud
import matplotlib.pyplot as plt

#magic line for inline plots

get_ipython().magic('matplotlib inline')


input_text = ' '
for texts in tweets_text:
    for words in texts.split():
            input_text = input_text + ' ' + words

wc = wordcloud.WordCloud().generate(input_text)
plt.imshow(wc, interpolation = 'bilinear')


# ###A "little" preprocessing

# In[ ]:

check_word = 'https'
input_text = ' '
for texts in tweets_text:
    for words in texts.split():
        if check_word not in words:
            input_text = input_text + ' ' + words

wc = wordcloud.WordCloud().generate(input_text)
plt.imshow(wc, interpolation = 'bilinear')


# ###One More Step

# In[ ]:

check_word_1 = 'https'
check_word_2 = 'RT'

input_text = ' '
for texts in tweets_text:
    for words in texts.split():
        if check_word_1 not in words and check_word_2 not in words:
            input_text = input_text + ' ' + words
            
wc = wordcloud.WordCloud().generate(input_text)
plt.imshow(wc, interpolation = 'bilinear')


# This idea can be generalized by creating a list of terms we don't want and 
# checking their existance at every iteration

# ##Traffic Analysis

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime


# ###Create a list with the dates and times of tweets

# In[ ]:

tweets_dates = []
for dates in tweets_list:
    tweets_dates.append(dates.created_at)
print ("Sample Format: " + str(tweets_dates[0]))


# ###Creating a Time Series for the Tweets

# In[ ]:

sample_indexing = pd.DatetimeIndex(tweets_dates)
dummy_value = np.ones(len(tweets_dates))
series = pd.Series(dummy_value, index = sample_indexing)


# ###Aggregating Tweets within a certain interval

# In[ ]:

sampling_time = '2Min'
aggregator = series.resample(sampling_time).sum().fillna(0)


# ###Plotting the tweet frequencies as a function of time

# In[ ]:

fig, ax = plt.subplots()
hours = mdates.MinuteLocator(interval= 2)
date_formatter = mdates.DateFormatter('%H:%M')

datemin = min(tweets_dates)#datetime(2017, 10, 14, 18, 15)
datemax = max(tweets_dates)#datetime(2017, 10, 14, 19, 00)

ax.xaxis.set_major_locator(hours)
ax.xaxis.set_major_formatter(date_formatter)
ax.set_xlim(datemin, datemax)
max_freq = aggregator.max()
#ax.set_ylim(0, max_freq)
ax.plot(aggregator.index, aggregator)


# ##Locating the peaks in the plot and move to next step

# ###Setting Peaks

# In[ ]:

peaklow = datetime(2017, 10, 18, 21, 23)
peakhigh = datetime(2017, 10, 18, 21, 25)


# ###Retrieving the tweet that generated massive traffic in that time frame

# In[ ]:

ci = 0 # Counter
retweet_count = []
filtered_tweets = []
for tweets in tweets_list:
    if tweets.created_at >= peaklow and  tweets.created_at < peakhigh :
        ci +=1
        print (ci)
        retweet_count.append(tweets.retweet_count)
        filtered_tweets.append(tweets.text)


# ###Printing

# In[ ]:

print ("Maximum Number of Retweets: " +str(max(retweet_count)))
print ("Index of the Tweet with Maximum Number of Retweets: " +str(np.argmax(retweet_count)))
print ("Tweet that generated maximum traffic: \n ") 
print (filtered_tweets[np.argmax(retweet_count)]) 


# ##Visualizing Global Activity

# ###Collecting Longitudes and Latitudes

# In[ ]:

longitude = []
latitude= []

for tweets in tweets_list:
    if tweets.coordinates:
        longitude.append(tweets.coordinates['coordinates'][0])
        latitude.append(tweets.coordinates['coordinates'][1])


# ##Visualization

# In[ ]:

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

m =  Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,            llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
m.drawcoastlines()
m.fillcontinents(color='white',lake_color='aqua')
m.drawparallels(np.arange(-90.,91.,30.))
m.drawmeridians(np.arange(-180.,181.,60.))
m.drawmapboundary(fill_color='aqua')
xpt,ypt = m(longitude,latitude)
lonpt, latpt = m(xpt,ypt,inverse=True)
m.plot(xpt,ypt,'bo', color = 'red')  # plot a blue dot there


# In[ ]:



