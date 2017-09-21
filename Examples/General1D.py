import numpy as np
import underoverfitting
import random


def train(truth_f, create_net_shape):
    random.seed(10)
    n = 1000
    percentage_dev = 0.1

    train_data_1d = np.random.uniform(0, 1, int((1-percentage_dev) * n)).astype(np.float32)
    dev_data_1d = np.random.uniform(0, 1, int(percentage_dev * n)).astype(np.float32)
    train_data = np.array([[x, x] for x in train_data_1d]).astype(np.float32)
    dev_data = np.array([[x, x] for x in dev_data_1d]).astype(np.float32)

    train_truths = np.array([truth_f(x) for x in train_data]).astype(np.int32)
    dev_truths = np.array([truth_f(x) for x in dev_data]).astype(np.int32)

    underoverfitting.train_and_predict_and_plot(train_data, train_truths, dev_data, dev_truths, create_net_shape)