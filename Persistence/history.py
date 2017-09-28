from nolearn.lasagne import NeuralNet
import lasagne
import io
import types


def save_parameters(parameters, stream):
    if isinstance(stream, str):
        with io.open(stream, 'a') as stream:
            return save_history(parameters, stream)

    assert 'layers' in parameters.keys()

    def tostring(p):
        if isinstance(p, (types.FunctionType, type)):
            type_or_module = next(m for m in [lasagne.nonlinearities, lasagne.layers] if hasattr(m, p.__name__))
            return type_or_module.__name__ + '.' + p.__name__
        else:
            return str(p)

    stream.write('v2\n')
    stream.write(",".join(('{}={}'.format(layer_key, tostring(layer)) for layer_key, layer in parameters['layers'])))

    stream.write('\n')
    stream.write(",".join(('{}={}'.format(key, tostring(parameters[key])) for key in parameters.keys()
                           if key != 'layers')))


def read_parameters(stream):
    if isinstance(stream, str):
        with io.open(stream, 'r') as stream:
            return read_history(stream)
    version_line = stream.readline()
    assert version_line == 'v2\n'
    layers_line = stream.readline()[:-1]
    layers = []
    for parameter in layers_line.split(','):
        key, value = parameter.split('=')
        layers.append((key, value))

    parameters_line = stream.readline()
    parameters = {'layers': layers}
    for parameter in parameters_line.split(','):
        key, value = parameter.split('=')
        parameters[key] = value
    return parameters


def save_history(self: NeuralNet, stream):
    if isinstance(stream, str):
        with io.open(stream, 'a') as stream:
            return save_history(self, stream)
    stream.write('v1\n')
    keys = ['epoch', 'valid_accuracy']
    for key in keys:
        stream.write(key + ', ')
    for entry in self.train_history_:
        for key in keys:
            stream.write(str(entry[key]) + ', ')
        stream.write('\n')


def read_history(stream):
    if isinstance(stream, str):
        with io.open(stream, 'r') as stream:
            return read_history(stream)
    version_line = stream.readline()
    assert version_line == 'v1'
    keys_line = stream.readline()
    keys = [key.strip() for key in keys_line.split(',')]
    lines = stream.readlines()
    data = [float(value.strip()) for value in lines.split(',')]
    return keys, data


NeuralNet.save_history = save_history
NeuralNet.read_history = read_history
