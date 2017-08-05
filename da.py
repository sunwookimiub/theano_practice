import utils
import theano
import numpy as np
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class dA(object):
    def __init__(self, input, params={}):
        self.params = {'corruption_level': 0., 'hidden_dim': 0}
        utils.update_dictionary_items(self.params, params)

        n_input = 28 * 28
        w_int = np.sqrt(6./(self.params['hidden_dim']+n_input))
        self.W = self.init_weights(-4*w_int, 4*w_int, (n_input, self.params['hidden_dim']))
        self.W_prime = self.W.T

        self.b_prime = self.shared_floatX(np.zeros(n_input)) # Bias of input
        self.b = self.shared_floatX(np.zeros(self.params['hidden_dim'])) # Bias of hidden

        self.x = input
        self.weights = [self.W, self.b, self.b_prime]

        self.__theano_build__()

    def shared_floatX(self, X): # Make hyperparameters runnable on GPU
        return theano.shared(np.asarray(X, dtype=theano.config.floatX))

    def init_weights(self, low, high, shape, coef=1.0):
        return self.shared_floatX(np.random.uniform(low, high, shape))

    def get_corrupted_input(self, input, corruption_level):
        # create a Theano random generator that gives symbolic random values
        np_rng = np.random.RandomState(123)
        theano_rng = RandomStreams(np_rng.randint(2 ** 30))
        return theano_rng.binomial(size=input.shape, p=1-corruption_level,
                                        dtype=theano.config.floatX) * input

    def model(self, input):
        h = T.nnet.sigmoid(T.dot(input, self.W) + self.b)
        pyx = T.nnet.sigmoid(T.dot(h, self.W_prime) + self.b_prime)
        return pyx

    def __theano_build__(self):
        theano.config.floatX = 'float32'
        tilde_x = self.get_corrupted_input(self.x, self.params['corruption_level'])
        z = self.model(tilde_x)

        cost = T.mean(T.sum(T.nnet.binary_crossentropy(z, self.x), axis=1)) # Vector of each minibatch's cross-entropy cost
        grads = T.grad(cost=cost, wrt=self.weights)

        lr = T.scalar('lr')
        self.train = theano.function(inputs=[self.x, lr], outputs=cost, updates= [[p, p - g * lr] for p,g in zip(self.weights, grads)], allow_input_downcast=True)
