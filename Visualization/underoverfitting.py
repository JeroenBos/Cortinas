import os
import matplotlib.pyplot as plt
from scipy.misc import imread
import matplotlib.cbook as cbook

path = os.path.dirname(os.path.realpath(__file__))
background = imread(cbook.get_sample_data(os.path.join(path, 'background underoverfitting.png')))
plt.imshow(background, zorder=0, extent=[0, 1, 0, 1])

pts = [(0.9, 0.8), (0.6, 0.65), (0.77, 0.81), (0.89, 0.86)]  # dummy data

x = [p[0] for p in pts]
y = [p[1] for p in pts]

plt.scatter(x, y, zorder=1)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().axes.get_xaxis().set_ticks([])
plt.gca().axes.get_yaxis().set_ticks([])


plt.show()
