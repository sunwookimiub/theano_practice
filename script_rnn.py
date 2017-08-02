import numpy as np
import itertools
import argparse
import rnn
import utils
import dataloader as dl
import theano.tensor as T

def parseArguments():
    parser = argparse.ArgumentParser(description='RNN Implementation on Reddit Comments')
    parser.add_argument('-v', "--vocabulary_size", type=int, default=1, help='Vocabulary Size / Word Dimension')
    parser.add_argument('-e', "--epoch", type=int, default=1, help='Number of Epochs')
    parser.add_argument('-l', "--learning_rate", type=float, default=1.0, help='Learning Rate')
    parser.add_argument('-d', "--hidden_dim", type=int, default=1, help='Hidden Dimensions')
    parser.add_argument('-b', "--bptt_truncate", type=int, default=1, help='BPTT Truncate Level')
    return parser.parse_args()

def main():
    args = parseArguments()

    vocabulary_size = 8000
    X_train, y_train = dl.load_reddit(vocabulary_size)

    # Get subset
    X_train = X_train[:100]
    y_train = y_train[:100]

    learning_rate = 0.005

    x = T.ivector('x')
    input_params = {'word_dim': args.vocabulary_size, 'learning_rate': args.learning_rate, 'hidden_dim': args.hidden_dim, 'bptt_truncate': args.bptt_truncate}
    rnnalgs = {
                'Theano Impl.': rnn.RNNTheano(x, vocabulary_size, input_params), 
              }

    for learnername, learner in rnnalgs.iteritems():
        print "Running {0} on {1}".format(learnername, learner.params)
        print "Testing: Expected loss for random predictions: {}".format(np.log(vocabulary_size))
        print "Testing: Actual loss: {}".format(learner.calculate_loss(X_train[:10], y_train[:10]))
        print "Begin Learning with SGD and BPTT"
        num_examples_seen = 0
        for epoch in range(args.epoch):
            for i in range(len(y_train)):
                learner.sgd_step(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1
            loss = learner.calculate_loss(X_train, y_train)
            print "Epoch {}: {} Training Loss after {} examples seen.".format(epoch, loss, num_examples_seen)
        #utils.save_model_parameters_theano("rnn-theano-%d-%d.npz" % (learner.hidden_dim, model.word_dim), model)

if __name__ == "__main__":
    main()
