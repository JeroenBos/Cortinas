import unittest
import greedydescent


class TestLinearEstimator(unittest.TestCase):

    def test_simple(self):
        y = greedydescent.linear_fit_estimator(1, 1, 2, 2, 3)
        self.assertEqual(y, 3)

    def test_simple_floats(self):
        y = greedydescent.linear_fit_estimator(1.0, 1.0, 2.0, 2.0, 4.0)
        self.assertEqual(y, 4.0)

    def test_xs_must_differ(self):
        with self.assertRaises(ValueError):
            greedydescent.linear_fit_estimator(1.0, 1.0, 1.0, 1.0, 4.0)


class TestParabolaEstimator(unittest.TestCase):

    def test_simple(self):
        y = greedydescent.fit_estimator(0, 0, 1, 1, 2, 4, 3)
        self.assertEqual(y, 9)

    def test_not_so_simple(self):
        y = greedydescent.fit_estimator(-2, 20, 1, -7, 2, 8, 0.5)  # 6 x^2 - 3 x - 10
        self.assertEqual(y, -10.0)

    def test_deferral_to_linear(self):
        y = greedydescent.fit_estimator(1, None, 1, 1, 2, 2, 3)
        self.assertEqual(y, 3)



