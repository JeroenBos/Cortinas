import unittest
from Visualization.underoverfitting import plot


class TestPlots(unittest.TestCase):

    def test_underoverfitting(self):
        plot([(0.9, 0.8), (0.6, 0.65), (0.77, 0.81), (0.89, 0.86)])
