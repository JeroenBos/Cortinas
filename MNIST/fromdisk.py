import pickle

import lasagne
from nolearn.lasagne import NeuralNet
from sklearn.metrics import confusion_matrix
import blog
import matplotlib.pyplot as plt
from nolearn.lasagne import visualize
import io

filename_nn = "C:\\t"
print('from disk: ' + filename_nn)


def compute_accuracy(trues, predictions):
    if len(trues) != len(predictions):
        raise "unequal sized arrays specified"
    result = 0
    for (true, prediction) in zip(trues, predictions):
        if true == prediction:
            result = result + 1
    return result / len(trues)


def indices_of_false_predictions(trues, predictions):
    if len(trues) != len(predictions):
        raise "unequal sized arrays specified"
    result = []
    i = 0
    for (true, prediction) in zip(trues, predictions):
        if true != prediction:
            result.append(i)
        i = i + 1
    return result


def read_net():
    net = blog.create_net_shape()
    net.load_params_from(filename_nn)
    return net


def get_predictions(): # gets tuple of trues and predictions
    net = read_net()

    *_, x_test, y_test = blog.load_dataset()
    predictions = net.predict(x_test)
    return y_test, predictions


def show_false_prediction():
    y_test, predictions = get_predictions()
    indices_false = indices_of_false_predictions(y_test, predictions)
    if len(indices_false) == 0:
        raise "100% correct prediction"
    print(*indices_false, sep='\n')
    for index_false in indices_false:
        blog.show_test(index_false)


def test_accuracy():
    y_test, predictions = get_predictions()
    accuracy = compute_accuracy(y_test, predictions)
    print("Accuracy: " + str(accuracy))

    with open("C:\\accuracy.txt", "a") as file:
        file.write(str(blog.num_epochs) + "\t" + str(accuracy) + "\n")


net2 = read_net()
dot = visualize.make_pydot_graph(lasagne.layers.get_all_layers(net2.layers_[4]))
with io.open("C:\\f.png", "wb") as f:
    f.write(dot.create_png())

# show_false_prediction()
blog.train().save_params_to(filename_nn)
# visualize.plot_occlusion(net2)
# plt.show()

test_accuracy()
# visualize.draw_to_file(net2, "C:\\nn.png")  # draws the diagram of the layers
# visualize.make_pydot_graph(net2.layers_[0]) # draws the diagram of the single layer
# visualize.plot_conv_weights(net2.layers_['conv2d1']) # shows a lot of 5x5 features
import numpy as np
from itertools import product

def p(figsize=(5, 5)):
    layer = net2.layers_['dense']
    W = layer.W.get_value()
    shape = W.shape
    nrows = np.ceil(np.sqrt(shape[0])).astype(int)
    ncols = nrows

    # print only single image rather than many, if this is the dense layer


    print(W.shape)
    feature_map = range(shape[1])[0]

    figs, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    for i, (r, c) in enumerate(product(range(nrows), range(ncols))):
        if i >= shape[0]:
            break
        axes[r, c].imshow(W[i, feature_map], cmap='gray',
                          interpolation='none')
    return plt

p().show()

