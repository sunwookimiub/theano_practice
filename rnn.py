import theano
import theano.tensor as T
import numpy as np
import utils

class RNNTheano:
    def __init__(self, input, params={}):
        self.params = {'word_dim': 1, 'hidden_dim': 1, 'bptt_truncate': 1}
        utils.update_dictionary_items(self.params, params)
        intvl = np.sqrt(1./self.params['hidden_dim'])
        self.U = self.init_weights(-intvl, intvl, (self.params['hidden_dim'], self.params['word_dim'])) 
        self.V = self.init_weights(-intvl, intvl, (self.params['word_dim'], self.params['hidden_dim'])) 
        self.W = self.init_weights(-intvl, intvl, (self.params['hidden_dim'], self.params['hidden_dim'])) 

        self.x = input
        self.y = T.ivector('y')

        self.weights = [self.U, self.V, self.W]

        self.__theano_build__()

    def shared_floatX(self, X): # Make hyperparameters runnable on GPU
        return theano.shared(np.asarray(X, dtype=theano.config.floatX))

    def init_weights(self, low, high, shape, coef=1.0):
        return self.shared_floatX(coef * np.random.uniform(low, high, shape)) # coef = 4?

    def model(self, x_t, s_t_prev): #Forward Prop
        s_t = T.tanh(self.U[:,x_t] + self.W.dot(s_t_prev))
        o_t = T.nnet.softmax(self.V.dot(s_t))
        return [o_t[0], s_t]

    def __theano_build__(self):
        [o,s], _ = theano.scan(fn=self.model, sequences=self.x, outputs_info=[None, dict(initial=T.zeros(self.params['hidden_dim']))], truncate_gradient=self.params['bptt_truncate'])

        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, self.y))
        lr = T.scalar('lr')
        grads = T.grad(cost=o_error, wrt=self.weights)

        self.forward_propagation = theano.function([self.x], o)
        self.predict = theano.function([self.x], prediction)
        self.ce_error = theano.function([self.x,self.y], o_error)
        self.train = theano.function([self.x, self.y, lr], [], updates=
                [[p, p - g * lr] for p,g in zip(self.weights, grads)])

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])

    def calculate_loss(self, X, Y):
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)

"""
class RNNNumpy:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim # word_dim: Size of our vocabulary
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        self.weights = [self.U, self.V, self.W]

    def model(self, x): #Forward Prop
        s = np.zeros((T+1, self.hidden_dim)) # Save all hidden states, +1 for the initial 0s
        o = np.zeros((T, self.word_dim)) # Save all outputs
        for t in np.arange(T): # Loop over total number of timesteps
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1])) # Indexing U = multiplying U with one hot vector
            o[t] = utils.softmax(self.V.dot(s[t])) # For each word make 8000 predictions of probabilities of the next word
        return [o,s]

    def predict(self, x):
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0
        for i in np.arange(len(y)):
            o, s = self.model(x[i])
            correct_word_predictions = o[np.arange(len(y[i])), y[i]] #TODO: Change to mine
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def calculate_loss(self, x, y):
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x,y) / N

    def bptt(self, x, y):
				T = len(y)
				# Perform forward propagation
				o, s = self.forward_propagation(x)
				# We accumulate the gradients in these variables
				dLdU = np.zeros(self.U.shape)
				dLdV = np.zeros(self.V.shape)
				dLdW = np.zeros(self.W.shape)
				delta_o = o
				delta_o[np.arange(len(y)), y] -= 1.
				# For each output backwards...
				for t in np.arange(T)[::-1]:
						dLdV += np.outer(delta_o[t], s[t].T)
						# Initial delta calculation
						delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
						# Backpropagation through time (for at most self.bptt_truncate steps)
						for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
								dLdW += np.outer(delta_t, s[bptt_step-1])
								dLdU[:,x[bptt_step]] += delta_t
								# Update delta for next step
								delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
				return [dLdU, dLdV, dLdW]

    def sgd_step(self, x, y, learning_rate):
        dLdU, dLdV, dLdW = self.bptt(x,y)
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW
"""
