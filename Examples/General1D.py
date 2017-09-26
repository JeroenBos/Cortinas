import numpy as np
import underoverfitting
import random
import Hyperparameter
import greedydescent
from LasagneComputerAndEstimator import LasagneComputerAndEstimator, lasagne_weigh
import Visualization
import HyperparameterDimension


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


def create_data(truth_f):
    random.seed(10)
    n = 1000
    percentage_dev = 0.1

    train_data_1d = np.random.uniform(0, 1, int((1 - percentage_dev) * n)).astype(np.float32)
    dev_data_1d = np.random.uniform(0, 1, int(percentage_dev * n)).astype(np.float32)
    train_data = np.array([[x, x] for x in train_data_1d]).astype(np.float32)
    dev_data = np.array([[x, x] for x in dev_data_1d]).astype(np.float32)

    train_truths = np.array([truth_f(x[0]) for x in train_data]).astype(np.int32)
    dev_truths = np.array([truth_f(x[0]) for x in dev_data]).astype(np.int32)

    return (train_data, train_truths, dev_data, dev_truths), train_truths.shape


def minimize(truth_f, default_nn_args, hyperdimensions):
    """

    :param truth_f: A function that takes a normalized number in [0,1] and returns a class index
    :param default_nn_args:
    :param hyperdimensions:
    :return:
    """

    assert 'output_num_units' in default_nn_args.keys()
    assert 'input_shape' in default_nn_args.keys()
    assert 'max_epochs' in default_nn_args.keys()
    if isinstance(hyperdimensions, HyperparameterDimension.HyperparameterDimension):
        hyperdimensions = [hyperdimensions]

    accuracies = []
    data, input_shape = create_data(truth_f)
    # default_nn_args['input_shape'] = input_shape
    computer = LasagneComputerAndEstimator(default_nn_args, evaluate=lambda nn: underoverfitting.train(nn, data))
    seed = Hyperparameter.Hyperparameter(hyperdimensions, None)
    for node in greedydescent.minimize(computer, [seed], computer.try_compute_cost, weigh=lasagne_weigh):
        accuracies.append(node.error)
        print(node.x)
        pts = Visualization.underoverfitting.scale_batch(accuracies)
        Visualization.underoverfitting.plot(pts)

