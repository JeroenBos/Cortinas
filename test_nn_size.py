import unittest
import nn_size
from MNIST.blog import create_net_shape


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
        nn = create_net_shape(max_epochs=2)
        nn.initialize()  # takes 16 seconds
        return nn
