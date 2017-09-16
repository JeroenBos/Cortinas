from functools import reduce
import operator


def compute_size_for(get_or_create_nn):
    """
Get a function that computes the size of the nolearn.lasagne.NeuralNet given a hyperparameter v
    :get_or_create_nn: A function that retrieves a nolearn.lasagne.NeuralNet given a hyperparameter v
    """

    def compute_size(v):
        nn = get_or_create_nn(v)
        shapes = [param.get_value().shape for param in nn.get_all_params(trainable=True) if param]
        nparams = reduce(operator.add, [reduce(operator.mul, shape) for shape in shapes])
        return nparams  # TODO: take into account the number of epochs and maybe data size
    return compute_size

