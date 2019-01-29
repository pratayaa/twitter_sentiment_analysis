# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 13:52:43 2019

@author: PRATAYA
"""
import re 
import numpy
import pickle
import tweepy

from tweepy import OAuthHandler

consumer_keys = 'd9VWMrzkGvzLrVxbfVrfLesgR'
consumer_secret = '24fWV0GgYBvCnuKvgnCranXSOBwpH6f7xbWDNOYzlf4cIziQrR'
access_tokens = '1088700139022307330-DjsN6LLeZ70NHCeK3pixbfe8AgdnsU'
access_secret = '3obhX10ZgBOQvNfcn5uCczSFUJ34KtM4PNUKS7d8A7BP8'

auth = OAuthHandler(consumer_keys, consumer_secret)
auth.set_access_token(access_tokens, access_secret)
args = ['UPSC']
api = tweepy.API(auth, timeout = 10)

list_tweets = []
query = args[0]
if len(args) == 1:
    for status in tweepy.Cursor(api.search, q=query+" -filter:retweets", lang = 'en', result_type = 'recent').items(100):
        list_tweets.append(status.text)

with open('classifier.pickle', 'rb') as f:
    classifier = pickle.load(f)
with open('tfidf_model.pickle', 'rb') as f:
    tfidf = pickle.load(f)

total_pos = 0
total_neg = 0

# Preprocessing the tweets and predicting sentiment
for tweet in list_tweets:
    tweet = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", tweet)
    tweet = tweet.lower()
    tweet = re.sub(r"that's","that is",tweet)
    tweet = re.sub(r"there's","there is",tweet)
    tweet = re.sub(r"what's","what is",tweet)
    tweet = re.sub(r"where's","where is",tweet)
    tweet = re.sub(r"it's","it is",tweet)
    tweet = re.sub(r"who's","who is",tweet)
    tweet = re.sub(r"i'm","i am",tweet)
    tweet = re.sub(r"she's","she is",tweet)
    tweet = re.sub(r"he's","he is",tweet)
    tweet = re.sub(r"they're","they are",tweet)
    tweet = re.sub(r"who're","who are",tweet)
    tweet = re.sub(r"ain't","am not",tweet)
    tweet = re.sub(r"wouldn't","would not",tweet)
    tweet = re.sub(r"shouldn't","should not",tweet)
    tweet = re.sub(r"can't","can not",tweet)
    tweet = re.sub(r"couldn't","could not",tweet)
    tweet = re.sub(r"won't","will not",tweet)
    tweet = re.sub(r"\W"," ",tweet)
    tweet = re.sub(r"\d"," ",tweet)
    tweet = re.sub(r"\s+[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+[a-z]$"," ",tweet)
    tweet = re.sub(r"^[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+"," ",tweet)
    sent = classifier.predict(tfidf.transform([tweet]).toarray())
    if sent[0] == 1:
        total_pos += 1
    else:
        total_neg += 1
    
# Visualizing the results
import matplotlib.pyplot as plt
import numpy as np
objects = ['Positive','Negative']
y_pos = np.arange(len(objects))

plt.bar(y_pos,[total_pos,total_neg],alpha=0.5)
plt.xticks(y_pos,objects)
plt.ylabel('Number')
plt.title('Number of Postive and NEgative Tweets')

plt.show()





