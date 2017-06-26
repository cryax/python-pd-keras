# -*- coding: utf-8 -*-
import gensim.models

# setup logging
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# train the basic model with text8-rest, which is all the sentences
# without the word - queen
if __name__ == '__main__':
    model = gensim.models.Word2Vec()
    sentences = gensim.models.word2vec.LineSentence("_Test/word2vec/text8-files/text8-rest")
    model.build_vocab(sentences)
    model.train(sentences,total_words=600000,epochs=32)
    
#    sentences2 = gensim.models.word2vec.LineSentence("_Test/word2vec/text8-files/text8-queen")
#    model.build_vocab(sentences2, update=True)
#    model.train(sentences2,total_words=700000)