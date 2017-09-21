import numpy as np
import underoverfitting
import random
from nolearn.lasagne import NeuralNet


def train(truth_f, truth_f_class_range_size, create_net_shape):
    """

    :param truth_f: A function that takes a normalized number in [0,1] and returns a class index
    :param truth_f_class_range_size: the highest possible result of truth_f
    :param create_net_shape: A wrapping the constructor to lasagne.NeuralNet
    :return:
    """
    random.seed(10)
    n = 1000
    percentage_dev = 0.1

    train_data_1d = np.random.uniform(0, 1, int((1-percentage_dev) * n)).astype(np.float32)
    dev_data_1d = np.random.uniform(0, 1, int(percentage_dev * n)).astype(np.float32)
    train_data = np.array([[x, x] for x in train_data_1d]).astype(np.float32)
    dev_data = np.array([[x, x] for x in dev_data_1d]).astype(np.float32)

    train_truths = np.array([truth_f(x[0]) for x in train_data]).astype(np.int32)
    dev_truths = np.array([truth_f(x[0]) for x in dev_data]).astype(np.int32)

    underoverfitting.train_and_predict_and_plot(train_data,
                                                train_truths,
                                                dev_data,
                                                dev_truths,
                                                create_net_shape,
                                                output_num_units=truth_f_class_range_size)


def create_nn(args, overriding_args):
    for key in overriding_args.keys():
        args[key] = overriding_args[key]

    nn = NeuralNet(**args)
    nn.initialize()
    return nn
