import numpy as np
from nltk import sent_tokenize, word_tokenize
import csv
import itertools
import nltk
import sys
from datetime import datetime
import rnnnumpy
import rnntheano
import utils

def load_reddit(vocabulary_size):
        unknown_token = "UNKNOWN_TOKEN"
        sentence_start_token = "SENTENCE_START"
        sentence_end_token = "SENTENCE_END"

        print(sent_tokenize("Testing. NLTK sent_tokenize"))
        print(word_tokenize("Testing. NLTK word_tokenize"))
        print(nltk.FreqDist(word_tokenize("Testing. NLTK FreqDist of word_tokenize")).items())

        temp = [1,2,3,4,5]

        print "Reading CSV file: datasets/subset.csv"
        #with open('datasets/reddit-comments-2015-08.csv', 'rb') as f:
        with open('datasets/subset.csv', 'rb') as f:
            reader = csv.reader(f, skipinitialspace=True)
            reader.next()
            sentences = []
            sentences2 = []
            # Split full comments into sentences
            for x in reader:
                sentences.append(nltk.sent_tokenize(x[0].decode('utf-8').lower()))
            # Flatten to one iterable 
            #TODO: Could just use sentences +=
            sentences = itertools.chain(*sentences)
            sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]

        print "Parsed {} sentences".format(len(sentences))

        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))

        print "Found %d unique word tokens." % len(word_freq.items())

        vocab = word_freq.most_common(vocabulary_size-1)
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(unknown_token)
        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

        print "Using vocabulary size %d." % vocabulary_size
        print "The least frequent word in our vocabulary is '%s' and appeared %d times" % (vocab[-1][0], vocab[-1][1])

        print " "

        print "Example tokenized sentence before Pre-processing: {}".format(tokenized_sentences[2])
        for i, sent in enumerate(tokenized_sentences):
            """
            word = None
            for w in sent:
                if w in word_to_index:
                    word = w
                else:
                    word = unknown_token
            """
            tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent] #TODO: Make mine

        print "Example sentence: {}".format(sentences[10])
        print "Example tokenized sentence after Pre-processing: {}".format(tokenized_sentences[2])

        X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences]) #TODO: Make mine
        y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences]) #TODO: Make mine

        print "Example sentence as x: {}".format(X_train[10])
        print "Example sentence as y: {}".format(y_train[10])

        print " "

        return X_train, y_train
