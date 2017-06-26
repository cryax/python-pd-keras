# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import json
import os
import numpy as np
import codecs
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.utils import simple_preprocess
from keras.engine import Input
from keras.layers import Embedding, merge
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout, Activation
from keras.models import load_model
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import load_model
from six.moves import cPickle
from Libs.tictoc import Tictoc
from sklearn.metrics import classification_report
import itertools
# tokenizer: can change this as needed
tokenize = lambda x: simple_preprocess(x)


def update_w2v_model(new_mentions):
    model = Word2Vec.load('_Test/word2vec/w2v-vietnamese/baomoi.w2v.model')
    model.build_vocab(new_mentions)
    model.train(new_mentions)
    
def create_embeddings(data_dir, embeddings_path='embeddings.npz', vocab_path='map.json', **params):
    """
    Generate embeddings from a batch of text
    :param embeddings_path: where to save the embeddings
    :param vocab_path: where to save the word-index map
    """
 
    class SentenceGenerator(object):
        def __init__(self, dirname):
            self.dirname = dirname
 
        def __iter__(self):
            for fname in os.listdir(self.dirname):
                for line in open(os.path.join(self.dirname, fname)):
                    yield tokenize(line)
 
#    sentences = SentenceGenerator(data_dir)
    model = Word2Vec.load('_Test/word2vec/w2v-vietnamese/baomoi.w2v.model')
#    model = Word2Vec(sentences, **params)
    weights = model.syn0
    np.save(open(embeddings_path, 'wb'), weights)
 
    vocab = dict([(k, v.index) for k, v in model.vocab.items()])
    with open(vocab_path, 'w') as f:
        f.write(json.dumps(vocab))
    return model
 
def load_vocab(vocab_path='map.json'):
    """
    Load word -> index and index -> word mappings
    :param vocab_path: where the word-index map is saved
    :return: word2idx, idx2word
    """
 
    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word
 
 
def word2vec_embedding_layer(embeddings_path='embeddings.npz'):
    """
    Generate an embedding layer word2vec embeddings
    :param embeddings_path: where the embeddings are saved (as a numpy file)
    :return: the generated embedding layer
    """
     
    weights = np.load(open(embeddings_path, 'rb'))
    layer = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights])
    return layer

def convert_mention2vec(mention,model,mention_length):
    try:
        list_w = mention.split(' ')
        if(len(list_w)<mention_length):
            list_w +=['Dummy']*(mention_length-len(list_w))
        if(len(list_w)>mention_length):
            list_w = list_w[:mention_length]
        final_w = []
        for w in list_w:
            if(w in model):
                w_v = model[w]           
            else:
                w_v = np.zeros((300,))
            final_w.append(w_v)
        return final_w
    except Exception as es:
        print(es)
        return 
def read_data(f1,f2):
    negative_rows=[]
    positive_rows =[]
    with codecs.open(f1) as f:
        positive_rows = list(set(f.readlines()))
    print('POS:',len(positive_rows))
    with codecs.open(f2) as f:
        negative_rows = list(set(f.readlines()))
    print('NEG:',len(negative_rows))
    label_positive = [1]*len(positive_rows)
    label_negative = [0]*len(negative_rows)
    
    return positive_rows,label_positive,negative_rows,label_negative

def classify_mentions(mentions,model):
    
    ress = model.predict(mentions)
#    Tictoc.toc('SPAM CLASSIFY')

    labels = []    
    for res in ress:
        if(res > 0.5):
            labels.append(1)
        else:
            labels.append(0)
    
    return labels

def test_model(X_test,y_test,model):
    
    print('len y = 0: ',len(filter(lambda y: y==0,y_test)))
    # evaluate model with sklearn
    predicted_classes = classify_mentions(X_test,model)
#    predicted_classes = map(lambda x: 1 if x==-1 else x,predicted_classes)
    target_names = ['class0','class1']
    print(classification_report(y_test, predicted_classes, target_names=target_names, digits = 6))
    return predicted_classes,y_test

def train_model_domain():
    vnmodel = create_embeddings('', size=100, min_count=5, window=5, sg=1, iter=25)
    
    file_name_pos = '_Test/word2vec/data_shortmention/not_negative_smartphone.csv'
    file_name_neg = '_Test/word2vec/data_shortmention/negative_smartphone.csv'
    data_pos,label_pos,data_neg,label_neg = read_data(file_name_pos,file_name_neg)

    text_vec_pos = data_pos#map(lambda x:convert_mention2vec(x,vnmodel),data_pos)
    text_vec_neg = data_neg#map(lambda x:convert_mention2vec(x,vnmodel),data_neg)
    X_sample = text_vec_pos+text_vec_neg
    y_sample = label_pos+label_neg
    
    '''Update w2v model'''
#    try:
#        X_sample_flatten = list(set(list(itertools.chain.from_iterable(X_sample))))
#
#        vnmodel.build_vocab(X_sample_flatten)
#        vnmodel.train(X_sample_flatten)
#    except Exception as es:
#        print(es)
    '''Update w2v model'''
    
    
    seed = 113
    np.random.seed(seed)
    np.random.shuffle(X_sample)
    np.random.seed(seed)
    np.random.shuffle(y_sample)
    
    X_train = X_sample[:int(.85*len(X_sample))]
    y_train = y_sample[:int(.85*len(y_sample))]
    print (len(X_train))
    X_test = X_sample[int(.85*len(X_sample)):]
    y_test = y_sample[int(.85*len(y_sample)):]

    
    
    model = Sequential()
    model.add(Convolution1D(nb_filter=64, filter_length=3, border_mode='same', activation='relu',\
                            input_shape =(50,300)))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    for i in range(10):  
       print('-----------------------EPOCH:',i)
       for range_ind in range(246):
          ind_from = range_ind * 64
          ind_to= ind_from + 64
          x_train = X_train[ind_from:ind_to-1]
          y = y_train[ind_from:ind_to-1]
          x_train = map(lambda x:convert_mention2vec(x,vnmodel,50),x_train)
          model.fit(x_train, y, nb_epoch=1, batch_size= 64)
    
    '''TEST'''
    X_test = map(lambda x:convert_mention2vec(x,vnmodel,50),X_test)
    test_model(X_test,y_test,model)
    

if __name__ == '__main__':
    train_model_domain()
#    sentences = LineSentence("_Test/word2vec/text8-files/text8-rest")
##    predicted_classes,y_test = test_model()
##    os.environ['GLOG_minloglevel'] = '2' 
#    file_name_pos = '_Test/word2vec/data_shortmention/not_negative_smartphone.csv'
#    file_name_neg = '_Test/word2vec/data_shortmention/negative_smartphone.csv'
#    data_pos,label_pos,data_neg,label_neg = read_data(file_name_pos,file_name_neg)
##    Word2Vec.most_similar
#    new_mentions = data_pos+data_neg
#    model2 = Word2Vec.load('_Test/word2vec/w2v-vietnamese/baomoi.w2v.model')#vi_data_cosine_cbow.json
#    data_neg = map(lambda x:x.split(),data_neg)
#    new_mentions = map(lambda x:x.split(),new_mentions)
#    model2.build_vocab(data_neg,update=True)
#    model2.train(data_neg)
##    model.save("modelbygensim.txt")
#    
#    data_pos = map(lambda x:x.split(),data_pos)
##    model.build_vocab(data_pos,update=True)
    

