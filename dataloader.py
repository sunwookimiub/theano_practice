import numpy as np
import gzip
import six.moves.cPickle as pickle
import theano
import theano.tensor as T
import numpy
import os
from nltk import sent_tokenize, word_tokenize
import csv
import itertools
import nltk
import sys
from datetime import datetime
import rnn
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

def load_mnist():
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    dataset = 'datasets/mnist.pkl.gz'

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'datasets/mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'datasets/mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
