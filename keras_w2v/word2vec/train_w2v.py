# -*- coding: utf-8 -*-
import pandas as pd # provide sql-like data manipulation tools. very handy.
pd.options.mode.chained_assignment = None
import numpy as np # high dimensional vector computing library.
from copy import deepcopy
from string import punctuation
from random import shuffle
LabeledSentence = gensim.models.doc2vec.LabeledSentence # we'll talk about this down below
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import numpy as np
from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()
import sys
from keras import callbacks
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Doc2Vec

def load_w2v():
    _fname = "word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin"
    w2vModel = Doc2Vec.load_word2vec_format(_fname, binary=True)
    return w2vModel 


class DataIterator:
    def __init__(self, data_path, batch_size = 1000):
        pos_files, neg_files = get_data_file_list(data_path)
        self.pos_iter = iter(pos_files)
        self.neg_iter = iter(neg_files)
        self.batchSize = batch_size

    def getNext(self):
        vectors = []
        values = []
        while (len(vectors) < self.batchSize):
            file = next (self.pos_iter, None)
            if file is None:
                break
            vec = np.load(join(vector_files_path, file))
            vectors.append(vec)
            values.append([1])

            file = next(self.neg_iter, None)
            if file is None:
                break
            vec = np.load(join(vector_files_path, file))
            vectors.append(vec)
            values.append([0])
        return np.array(vectors), np.array(values)
    
def get_generator(data_path):
    files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    while 1:
        for file in files:
            x = np.load(join(data_path, file))
            x = np.array([x])
            label = get_label(file)
            y = np.full((1, 1), [label])
            yield (x, y)

def get_label(filename):
    if filename.startswith("pos"):
        return 1
    return 0
    
#Fit model with generator
    model.fit_generator(get_generator("training data path"),
                        samples_per_epoch=25000, nb_epoch=40,
                        validation_data=get_generator("test data path"), nb_val_samples=25000,
                        callbacks=cbks)




def train():
    timesteps = 350
    dimensions = 300
    batch_size = 64
    epochs_number = 40
    model = Sequential()
    model.add(LSTM(200,  input_shape=(timesteps, dimensions),  return_sequences=False))
    model.add(Dropout(0.2))
       model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
    fname = 'weights/keras-lstm.h5'
    model.load_weights(fname)
    cbks = [callbacks.ModelCheckpoint(filepath=fname, monitor='val_loss', save_best_only=True),
            callbacks.EarlyStopping(monitor='val_loss', patience=3)]
    #get all available data samples from data iterators
    train_iterator = DataIterator(train_data_path, sys.maxint)
    test_iterator = DataIterator(test_data_path, sys.maxint)
    train_X, train_Y = train_iterator.get_next()
    test_X, test_Y = test_iterator.get_next()
    model.fit(train_X, train_Y, batch_size=batch_size, callbacks=cbks, nb_epoch=epochs_number,
              show_accuracy=True, validation_split=0.2, shuffle=True)
    loss, acc = model.evaluate(test_X, test_Y, batch_size, show_accuracy=True)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))