import os
import argparse
import numpy as np
import theano.tensor as T
from dataloader import load_mnist
import da
import utils

def parseArguments():
    parser = argparse.ArgumentParser(description='Denoising Autoencoder on MNIST dataset')
    parser.add_argument('-b', "--batch_size", type=int, default=1, help='Batch Size')
    parser.add_argument('-c', "--corruption_level", type=float, default=1.0, help='Corruption Level')
    parser.add_argument('-e', "--epoch", type=int, default=1, help='Number of Epochs')
    parser.add_argument('-l', "--learning_rate", type=float, default=1.0, help='Learning Rate')
    parser.add_argument('-d', "--hidden_dim", type=int, default=1, help='Hidden Dimensions')
    return parser.parse_args()

def main():
    args = parseArguments()

    datasets = load_mnist()
    train_set_x, train_set_y = datasets[0]
    train_set_x = train_set_x.get_value()
    n_train_batches = train_set_x.shape[0] // args.batch_size

    x = T.fmatrix('x')  # the data is presented as rasterized images
    input_params = {
                      'corruption_level': args.corruption_level, 'epoch': args.epoch, 'learning_rate': args.learning_rate, 'hidden_dim': args.hidden_dim
                   }
    modelda = da.dA(x, input_params)

    for epoch in range(args.epoch):
        for batch_index in range(n_train_batches):
            cost = modelda.train(train_set_x[batch_index * args.batch_size: (batch_index+1) * args.batch_size], args.learning_rate)
        print 'Training epoch %d, cost ' % epoch, np.mean(cost, dtype='float64')

    #output_folder = 'dA_plots'
    #os.chdir(output_folder)
    #utils.save_results(da, corruption_level)
    #os.chdir('../')


if __name__ == '__main__':
    main()
