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


app = Flask(__name__)


def get_tokenizer():
    tokenizer_file_name = os.path.join('Data', 'tokenizers', 'tokenizer_' + str(200) + '.pkl')
    time_start = datetime.now()
    if os.path.isfile(tokenizer_file_name):
        print('Loading tokenizer...')
        with open(tokenizer_file_name, 'rb') as file:
            tokenizer = pickle.load(file)
    else:
        print('Training tokenizer...')
        tokenizer = Tokenizer(num_words=200)
        # tokenizer.fit_on_texts(train_text)
        
        with open(tokenizer_file_name, 'wb') as file:
            pickle.dump(tokenizer, file)
        
    print('Got tokenizer for vocab size: ' + str(200) + ' in ' + str(datetime.now() - time_start))
    return tokenizer

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/GuessGame.html")
def test():
    # render_template('GuessGame.html')

    return render_template('GuessGame.html')

@app.route("/neural/<sentence>")
def word_to_predict(sentence):
    model = load_model(os.path.join('Data', 'models', 'yelp_trained.hd5'))
    list_review = []
    tokenizer = get_tokenizer()
    tokenizer.fit_on_texts(list_review)
    list_review.append(sentence)
    list_review = tokenizer.texts_to_sequences(list_review)
    list_review = pad_sequences(list_review, maxlen=250)
    x = model.predict(list_review)
    a = pd.DataFrame(x)
    a.columns = [1,2,3,4,5]

    score = a.idxmax(axis=1)
    score = score[0]
    # dict = {}
    # for i in range(5):
    #     dict[str(i+1)] = round(((a.loc[0: ,i+1][0])*100),1)
    # return (print(str(dict['1']) + "%" + " confident of rating being 1, \n" + str(dict['2']) + "%" + " confident of rating being 2, \n" + 
    #      str(dict['3']) + "%" + " confident of rating being 3, \n" + str(dict['4']) + "%" + " confident of rating being 4, \n" 
    # + str(dict['5']) + "%" + " confident of rating being 5. \n"
    #      "Therefore, I predict the rating to be " + str(score)))
    return print('chicken')

if __name__ == "__main__":
    app.run(debug=True)