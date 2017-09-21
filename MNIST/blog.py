# http://blog.christianperone.com/2015/08/convolutional-neural-networks-and-feature-extraction-with-python/

# pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
# pip install https://github/.comdnouri/nolearn/archive/master.zip#egg=nolearn
# conda install mingw libpython
# conda install m2w64-toolchain
# conda install pygpu
# pip install pydotplus
# conda install graphviz
# https://developer.nvidia.com/cuda-downloads (ensure to install version 9.0)
# restart PC
# install C++ in VS2017 (to ultimately get 'cl.exe') and insert
# C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64 into PATH environment variable
# Also installed Visual C++ 2015 (14.0) compiler (stand-alone, doesnt require entire VS)
# my .theanorc:
# [global]
# floatX = float32
# device = cuda
#
# [cuda]
# root = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0
#
# [dnn]
# enabled = False
#
# install http://www.graphviz.org/Download_windows.php and add install location to PATH
#


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from urllib import request
import pickle
import os
import gzip
import numpy as np
import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print("Device: " + theano.config.device)
num_epochs = 1000


def load_dataset():
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    filename = 'mnist.pkl.gz'
    if not os.path.exists(filename):
        print("Downloading MNIST dataset...")
        request.urlretrieve(url, filename)
        print("Downloaded MNIST dataset...")
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    x_train, y_train = data[0]
    x_val, y_val = data[1]
    x_test, y_test = data[2]
    x_train = x_train.reshape((-1, 1, 28, 28))
    x_val = x_val.reshape((-1, 1, 28, 28))
    x_test = x_test.reshape((-1, 1, 28, 28))
    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)
    return x_train, y_train, x_val, y_val, x_test, y_test


def create_net_shape(**kwargs):
    kwargs['input_shape'] = (None, 1, 28, 28)
    args = {'layers': [('input', layers.InputLayer),
                       ('conv2d1', layers.Conv2DLayer),
                       ('maxpool1', layers.MaxPool2DLayer),
                       ('conv2d2', layers.Conv2DLayer),
                       ('maxpool2', layers.MaxPool2DLayer),
                       ('dropout1', layers.DropoutLayer),
                       ('dense', layers.DenseLayer),
                       ('dropout2', layers.DropoutLayer),
                       ('output', layers.DenseLayer),
                       ],
            # layer conv2d1
            'conv2d1_num_filters': 32,
            'conv2d1_filter_size': (5, 5),
            'conv2d1_nonlinearity': lasagne.nonlinearities.rectify,
            'conv2d1_W': lasagne.init.GlorotUniform(),
            # layer maxpool1
            'maxpool1_pool_size': (2, 2),
            # layer conv2d2
            'conv2d2_num_filters': 32,
            'conv2d2_filter_size': (5, 5),
            'conv2d2_nonlinearity': lasagne.nonlinearities.rectify,
            # layer maxpool2
            'maxpool2_pool_size': (2, 2),
            # dropout1
            'dropout1_p': 0.5,
            # dense
            'dense_num_units': 256,
            'dense_nonlinearity': lasagne.nonlinearities.rectify,
            # dropout2
            'dropout2_p': 0.5,
            # output
            'output_nonlinearity': lasagne.nonlinearities.softmax,
            'output_num_units': 10,
            # optimization method params
            'update': nesterov_momentum,
            'update_learning_rate': 0.01,
            'update_momentum': 0.9,
            'max_epochs': num_epochs,
            'verbose': 1}

    # override default arguments with specified arguments
    for key in kwargs.keys():
        args[key] = kwargs[key]

    return NeuralNet(**args)


def show_test(index_image):
    *_, x_test, y_test = load_dataset()
    plt.imshow(x_test[index_image][0], cmap=cm.binary)
    plt.show()


def train():
    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset()

    net1 = create_net_shape()

    # Train the network
    nn = net1.fit(x_train, y_train)
    return nn
