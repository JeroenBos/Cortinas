import theano
import lasagne
import MNIST.blog
import numpy as np
import math


def train_and_predict(train_data, train_truths, dev_data, dev_truths, **nn_kwargs):
    print("Device: " + theano.config.device)

    nn = MNIST.blog.create_net_shape(**nn_kwargs)

    nn.fit(train_data, train_truths)

    train_accuracy = nn.train_history_[-1]['valid_accuracy']

    dev_accuracy = get_accuracy(nn.layers_[0], nn.layers_[-1], dev_data, dev_truths, 100)

    return train_accuracy, dev_accuracy


def get_accuracy(input_layer, output_layer, data, truths, dev_batch_size):
    dev_batch_size = dev_batch_size if dev_batch_size != 0 else data.shape[0]
    n_batches = int(math.ceil(data.shape[0] / dev_batch_size))

    compute_output = theano.function([input_layer.input_var],
                                     lasagne.layers.get_output(output_layer, deterministic=True))

    val_predictions = []
    for batch_index in range(0, n_batches):
        dev_batch = data[dev_batch_size * batch_index: dev_batch_size * (batch_index + 1)]
        val_output = compute_output(dev_batch)
        val_predictions.extend(np.argmax(val_output, axis=1))

    accuracy = np.mean(val_predictions == truths)
    return accuracy






