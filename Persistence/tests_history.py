import unittest
import lasagne
from Persistence.history import *
from lasagne import layers
from io import StringIO
from pydoc import locate


class TestPersistence(unittest.TestCase):
    def test_simple_parameters(self):
        parameters = {'layers': [('input', 'test')],
                      'dense1_num_units': 1000,
                      'dense1_num_leading_axes': 1}
        stream = StringIO()
        save_parameters(parameters, stream)
        stream.seek(0)
        result = read_parameters(stream)

        self.assertCountEqual(parameters, result)

    def test_nonlinearity_parameter(self):
        parameters = {'layers': [('input', 'test')],
                      'dense1_num_units': 1000,
                      'dense1_nonlinearity': lasagne.nonlinearities.rectify,
                      'dense1_num_leading_axes': 1}
        stream = StringIO()
        save_parameters(parameters, stream)
        stream.seek(0)
        result = read_parameters(stream)

        self.assertCountEqual(parameters, result)
        self.assertEqual('lasagne.nonlinearities.rectify', result['dense1_nonlinearity'])

    def test_layer_parameter(self):
        parameters = {'layers': [('input', layers.InputLayer)],
                      'dense1_num_units': 1000,
                      'dense1_nonlinearity': lasagne.nonlinearities.rectify,
                      'dense1_num_leading_axes': 1}
        stream = StringIO()
        save_parameters(parameters, stream)
        stream.seek(0)
        result = read_parameters(stream)

        self.assertCountEqual(parameters, result)
        self.assertEqual('input', result['layers'][0][0])
        layer_type_description = result['layers'][0][1]
        layer_type = locate(layer_type_description)
        self.assertEqual(layers.InputLayer, layer_type)




