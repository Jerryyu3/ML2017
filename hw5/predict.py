import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, adam
from keras.utils import np_utils
import sys
import csv
import string
import pickle
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import Counter

def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r') as f:
    
        tags = []
        articles = []
        tags_list = []
    
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
        
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]

            articles.append(article)

        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list) 

def f1_score(y_true,y_pred):
    thresh = 0.4;
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred, axis=-1)

    precision = tp/(K.sum(y_pred,axis=-1)+K.epsilon());
    recall = tp/(K.sum(y_true,axis=-1)+K.epsilon());
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

def predict(X_test):

    model = load_model('best_0525_3.hdf5',custom_objects={'f1_score':f1_score});

    with open("word_index_0525_v2_3.pickle",'rb') as handle:
        word_index = pickle.load(handle);

    with open("label_mapping.pickle",'rb') as handle:
        tag_list = pickle.load(handle);

    tokenizer = Tokenizer();
    tokenizer.word_index = word_index
    test_sequences = tokenizer.texts_to_sequences(X_test)

    
    with open("word_list_0525_v2_3.pickle",'rb') as handle:
        word_list = pickle.load(handle);
    prune_test_seq = []
    for i in test_sequences:
        temp = []
        for j in i:
            if j in word_list: temp.append(j);
        prune_test_seq.append(temp)
    test_sequence = prune_test_seq

    ### padding to equal length
    print ('Padding sequences.')
    max_article_length = 306  #train_sequences.shape[1]
    test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)
    
    
    Y_pred = model.predict(test_sequences)
    thresh = 0.4
    with open(sys.argv[2],'w') as output:
        print ('\"id\",\"tags\"',file=output)
        Y_pred_thresh = (Y_pred > thresh).astype('int')
        for index,labels in enumerate(Y_pred_thresh):
            labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
            labels_original = ' '.join(labels)
            print ('\"%d\",\"%s\"'%(index,labels_original),file=output)
    


def main():
  (_, X_test,_) = read_data(sys.argv[1],False)
  predict(X_test);

if __name__=='__main__':
  main()
