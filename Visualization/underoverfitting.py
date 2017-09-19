import os
import matplotlib.pyplot as plt
from scipy.misc import imread
import matplotlib.cbook as cbook

path = os.path.dirname(os.path.realpath(__file__))
_background = imread(cbook.get_sample_data(os.path.join(path, 'background underoverfitting.png')))


def plot(pts):
    plt.ion()
    plt.imshow(_background, zorder=0, extent=[0, 1, 0, 1])

    x = [p[0] for p in pts]
    y = [p[1] for p in pts]

    plt.scatter(x, y, zorder=1, marker='x')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    axes = plt.gca().axes
    axes.get_xaxis().set_ticks([])
    axes.get_xaxis().set_label_coords(.75, -0.025)
    axes.get_yaxis().set_ticks([])
    axes.get_yaxis().set_label_coords(-0.025, .75)

    plt.xlabel('underfitting')
    plt.ylabel('overfitting')
    plt.text(1.02, 1.02, 'random')
    plt.text(-0.14, -0.04, 'perfect')
    plt.subplots_adjust(left=0.3)
    plt.pause(0.01)


def scale(train_accuracy, dev_accuracy):
    """ Converts training and dev accuracies to corresponding locations in the plot"""
    return 1 - train_accuracy, 1 - dev_accuracy


def scale_batch(accuracies):
    """ Converts a list of 2-tuples containing the training and dev accuracies for different hyperparameters,
    and converts them to corresponding locations in the plot"""
    return [scale(train_accuracy, dev_accuracy) for train_accuracy, dev_accuracy in accuracies]

