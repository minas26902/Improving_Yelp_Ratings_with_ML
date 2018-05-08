import cgi
from flask import Flask, render_template
import pandas as pd
import numpy as np
from glob import glob
import pickle
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.utils import resample
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, RNN
from keras.layers.embeddings import Embedding
from keras.models import load_model
import warnings

form = cgi.FieldStorage()
searchterm =  form.getvalue('searchbox')

def word_to_predict(sentence):
    model = load_model(os.path.join('Data', 'models', 'yelp_trained.hd5'))
    list_review = []
    tokenizer = get_tokenizer(sentence)
    tokenizer.fit_on_texts(list_review)
    list_review.append(sentence)
    # list_review.append("hello javascript is very hard and i am tired of doing it, but i will preservere.")
    list_review = tokenizer.texts_to_sequences(list_review)
    list_review = pad_sequences(list_review, maxlen=250)
    x = model.predict(list_review)
    a = pd.DataFrame(x)
    a.columns = [1,2,3,4,5]

    score = a.idxmax(axis=1)
    score = score[0]
    dict = {}
    for i in range(5):
        dict[str(i+1)] = round(((a.loc[0: ,i+1][0])*100),1)
    return (str(dict['1']) + "%" + " confident of rating being 1, \n" + str(dict['2']) + "%" + " confident of rating being 2, \n" + 
         str(dict['3']) + "%" + " confident of rating being 3, \n" + str(dict['4']) + "%" + " confident of rating being 4, \n" 
    + str(dict['5']) + "%" + " confident of rating being 5. \n"
         "Therefore, I predict the rating to be " + str(score))

word_to_predict(searchterm)