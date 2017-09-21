import unittest
from Visualization.underoverfitting import plot
import MNIST.blog
import underoverfitting
import time
import copy


class TestPlots(unittest.TestCase):

    def test_underoverfitting(self):
        plot([(0.9, 0.8), (0.6, 0.65), (0.77, 0.81), (0.89, 0.86)])

    def test_MNINST(self):
        train_data, train_truths, dev_data, dev_truths, _, __ = MNIST.blog.load_dataset()

        p = underoverfitting.train_and_predict(train_data, train_truths, dev_data, dev_truths,
                                               MNIST.blog.create_net_shape,
                                               max_epochs=2,
                                               output_num_units=10)[-1]

        self.assertIsInstance(p[0], float)
        self.assertIsInstance(p[1], float)

    def test_plot_MNIST(self):
        train_data, train_truths, dev_data, dev_truths, _, __ = MNIST.blog.load_dataset()

        underoverfitting.train_and_predict_and_plot(train_data, train_truths, dev_data, dev_truths,
                                                    MNIST.blog.create_net_shape,
                                                    max_epochs=5,
                                                    output_num_units=10)

    def test_plot_thread(self):
        pts = [[0.1, 0.2], [0.3, 0.4]]
        plot(copy.copy(pts))

        time.sleep(3)
        pts.append([0.15, 0.16])
        pts.append([0.07, 0.08])
        plot(pts)
