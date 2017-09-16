import unittest
import nn_size
from nolearn.lasagne import NeuralNet
from lasagne import layers
import lasagne


class SizeComputation1(unittest.TestCase):

    def test_size_computation1(self):
        size = nn_size.compute_size_for(self.get_nn)(v=0)
        self.assertEqual(size, 160362)

    cached_net = None

    @staticmethod
    def get_nn(v):
        if SizeComputation1.cached_net is None:
            SizeComputation1.cached_net = SizeComputation1.compute_nn(v)
        return SizeComputation1.cached_net

    @staticmethod
    def compute_nn(_v):
        nn = NeuralNet(
            layers=[('input', layers.InputLayer),
                    ('conv2d1', layers.Conv2DLayer),
                    ('maxpool1', layers.MaxPool2DLayer),
                    ('conv2d2', layers.Conv2DLayer),
                    ('maxpool2', layers.MaxPool2DLayer),
                    ('dropout1', layers.DropoutLayer),
                    ('dense', layers.DenseLayer),
                    ('dropout2', layers.DropoutLayer),
                    ('output', layers.DenseLayer)],
            # input layer
            input_shape=(None, 1, 28, 28),
            # layer conv2d1
            conv2d1_num_filters=32,
            conv2d1_filter_size=(5, 5),
            conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
            conv2d1_W=lasagne.init.GlorotUniform(),
            # layer maxpool1
            maxpool1_pool_size=(2, 2),
            # layer conv2d2
            conv2d2_num_filters=32,
            conv2d2_filter_size=(5, 5),
            conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
            # layer maxpool2
            maxpool2_pool_size=(2, 2),
            # dropout1
            dropout1_p=0.5,
            # dense
            dense_num_units=256,
            dense_nonlinearity=lasagne.nonlinearities.rectify,
            # dropout2
            dropout2_p=0.5,
            # output
            output_nonlinearity=lasagne.nonlinearities.softmax,
            output_num_units=10,
            # optimization method params
            update=lasagne.updates.nesterov_momentum,
            update_learning_rate=0.01,
            update_momentum=0.9,
            max_epochs=2,
            verbose=1,)
        nn.initialize()  # takes 16 seconds
        return nn

