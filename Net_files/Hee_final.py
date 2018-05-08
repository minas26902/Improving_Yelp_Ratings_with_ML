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
warnings.simplefilter('ignore', FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

DATA_ROOT = 'Data'
SEED = 2018
vocab_size = 200
MAX_REVIEW_LEN = 250
BATCH_SIZE = 100


def get_tokenizer(vocab_size, train_text=None):
    tokenizer_file_name = os.path.join(DATA_ROOT, 'tokenizers', 'tokenizer_' + str(vocab_size) + '.pkl')
    time_start = datetime.now()
    if os.path.isfile(tokenizer_file_name):
        print('Loading tokenizer...')
        with open(tokenizer_file_name, 'rb') as file:
            tokenizer = pickle.load(file)
    else:
        print('Training tokenizer...')
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(train_text)
        
        with open(tokenizer_file_name, 'wb') as file:
            pickle.dump(tokenizer, file)
        
    print('Got tokenizer for vocab size: ' + str(vocab_size) + ' in ' + str(datetime.now() - time_start))
    return tokenizer

def word_to_predict(sentence):
    list_review = []
    tokenizer = get_tokenizer(vocab_size)
    tokenizer.fit_on_texts(list_review)
    list_review.append(sentence)
    list_review = tokenizer.texts_to_sequences(list_review)
    list_review = pad_sequences(list_review, maxlen=MAX_REVIEW_LEN)
    x = model.predict(list_review)
    a = pd.DataFrame(x)
    a.columns = [1,2,3,4,5]

    score = a.idxmax(axis=1)
    score = score[0]
    z = dict = {}
    for i in range(5):
        dict[str(i+1)] = round(((a.loc[0: ,i+1][0])*100),1)
    return (print(str(dict['1']) + "%" + " confident of rating being 1, \n" + str(dict['2']) + "%" + " confident of rating being 2, \n" + 
         str(dict['3']) + "%" + " confident of rating being 3, \n" + str(dict['4']) + "%" + " confident of rating being 4, \n" 
    + str(dict['5']) + "%" + " confident of rating being 5. \n"
         "Therefore, I predict the rating to be " + str(score)))

#Load model
model = load_model(os.path.join(DATA_ROOT, 'models', 'yelp_trained.hd5'))

while True:
    a = input('Would you like to submit a review? (y/n)')
    if a == "y":
        sentence = input('Enter review here: ')
        word_to_predict(sentence)
    elif a =="n":
        print("Goodbye")
        break
    else:
        print("Enter either y/n")

word_to_predict(sentence)