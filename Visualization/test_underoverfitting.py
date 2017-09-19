import unittest
from Visualization.underoverfitting import plot
import MNIST.blog
import underoverfitting


class TestPlots(unittest.TestCase):

    def test_underoverfitting(self):
        plot([(0.9, 0.8), (0.6, 0.65), (0.77, 0.81), (0.89, 0.86)])

    def test_MNINST(self):
        train_data, train_truths, dev_data, dev_truths, _, __ = MNIST.blog.load_dataset()

        p = underoverfitting.train_and_predict(train_data, train_truths, dev_data, dev_truths, max_epochs=2)[-1]

        self.assertIsInstance(p[0], float)
        self.assertIsInstance(p[1], float)


