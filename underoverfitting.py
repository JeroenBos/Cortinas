import theano
import lasagne
import numpy as np
import math
import Visualization.underoverfitting


def train_and_predict(train_data, train_truths, dev_data, dev_truths, create_nn, dev_func=None, **nn_kwargs):
    print("Device: " + theano.config.device)
    assert 'output_num_units' not in nn_kwargs.keys()
    assert 'input_shape' not in nn_kwargs.keys()
    assert 'on_epoch_finished' not in nn_kwargs.keys()

    result = []

    def dev(nn2, history):
        train_accuracy = history[-1]['valid_accuracy']
        dev_accuracy = get_accuracy(nn2.layers_[0], nn2.layers_[-1], dev_data, dev_truths, 100)
        result.append((train_accuracy, dev_accuracy))

        if dev_func is not None:
            dev_func(result)

    truths_count = len(set(train_truths))
    nn = create_nn(**nn_kwargs, on_epoch_finished=[dev], input_shape=train_data.shape, output_num_units=truths_count)
    nn.fit(train_data, train_truths)

    return result


def train_and_predict_and_plot(train_data, train_truths, dev_data, dev_truths, create_nn, **nn_kwargs):
    def plot(accuracies):
        pts = Visualization.underoverfitting.scale_batch(accuracies)
        Visualization.underoverfitting.plot(pts)

    return train_and_predict(train_data, train_truths, dev_data, dev_truths, create_nn, plot, **nn_kwargs)


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
