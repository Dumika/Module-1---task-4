import os
import tweepy
from tweepy import OAuthHandler
import pandas as pd
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob

#Classification

with open('bayesTrain.csv', 'r', encoding="utf-8") as train: #training data set
     cl = NaiveBayesClassifier(train)

with open('bayesTrain.csv', 'r', encoding="utf-8") as test:  #testing data set
    Accuracy = cl.accuracy (test)

data = ["කටුනායක ගුවන් තොටුපල පිටවීමේ ඇතුල්වීමේ ආලින්දයත් අමුත්තන්ට වැසේ "]
text = cl.classify(data)
print(text)
print (Accuracy)

#Acc = cl.show_informative_features(5)

#NER
from monkeylearn import MonkeyLearn
ml = MonkeyLearn('cc10e828cafbfe08d80a7dfd1dff4bfa9968b93b')
#data = ["20210507 කොවිඩ්19 රෝගය පාලනය ආයුර්වේද රෝහල් පද්ධතිය යොදා ගැනීම කඩිනම් කෙරේ "]
model_id = 'ex_yAXTMBc4'
result = ml.extractors.extract(model_id, data)
print(result.body)


