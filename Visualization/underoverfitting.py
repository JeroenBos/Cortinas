import os
import matplotlib.pyplot as plt
from scipy.misc import imread
import matplotlib.cbook as cbook
import multiprocessing
import queue

path = os.path.dirname(os.path.realpath(__file__))
_background = imread(cbook.get_sample_data(os.path.join(path, 'background underoverfitting.png')))
refresh_rate = 0.1  # in seconds
q = None


def plot(new_pts):
    global q
    if q is None:
        q = multiprocessing.JoinableQueue()
        process = multiprocessing.Process(target=_worker, args=(q,))
        process.start()
    q.put(new_pts)


def _worker(local_queue):
    while True:
        new_pts = None
        try:
            new_pts = local_queue.get(False)
            while True:
                new_pts = local_queue.get(False)
                local_queue.task_done()
        except queue.Empty:
            if new_pts is not None:
                _update_plot(new_pts)
                local_queue.task_done()
        # on empty or non-empty queue: yield control back to the plot
        plt.pause(refresh_rate)


def _update_plot(pts):
    plt.ion()
    plt.show()
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
    axes.set_prop_cycle(None)

    plt.xlabel('underfitting')
    plt.ylabel('overfitting')
    plt.text(1.02, 1.02, 'random')
    plt.text(-0.14, -0.04, 'perfect')
    plt.subplots_adjust(left=0.3)

    plt.draw()
    plt.pause(refresh_rate)


def scale(train_accuracy, dev_accuracy):
    """ Converts training and dev accuracies to corresponding locations in the plot"""
    return 1 - train_accuracy, 1 - dev_accuracy


def scale_batch(accuracies):
    """ Converts a list of 2-tuples containing the training and dev accuracies for different hyperparameters,
    and converts them to corresponding locations in the plot"""
    return [scale(train_accuracy, dev_accuracy) for train_accuracy, dev_accuracy in accuracies]
