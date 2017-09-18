import os
import matplotlib.pyplot as plt
from scipy.misc import imread
import matplotlib.cbook as cbook


def plot(pts):
    path = os.path.dirname(os.path.realpath(__file__))
    background = imread(cbook.get_sample_data(os.path.join(path, 'background underoverfitting.png')))
    plt.imshow(background, zorder=0, extent=[0, 1, 0, 1])

    x = [p[0] for p in pts]
    y = [p[1] for p in pts]

    plt.scatter(x, y, zorder=1)
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
    plt.show()


plot([(0.9, 0.8), (0.6, 0.65), (0.77, 0.81), (0.89, 0.86)])
