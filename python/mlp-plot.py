import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier

from numpy import genfromtxt


csv = genfromtxt('data.csv', delimiter=',')
data = csv[:,:2]
categories = csv[:,2]


cmap_light = ListedColormap(['#BBFF77', '#FFBB77', '#AAAAFF'])
cmap_bold = ListedColormap(['#55FF00', '#FF5500', '#0000FF'])


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
models = (MLPClassifier(hidden_layer_sizes=(4,), activation = "relu"),
          MLPClassifier(hidden_layer_sizes=(4,4), activation = "relu"),
          MLPClassifier(hidden_layer_sizes=(4,4,4), activation = "relu"),
          MLPClassifier(hidden_layer_sizes=(8,8), activation = "relu")
                    )
models = (clf.fit(data, categories) for clf in models)

# title for the plots
titles = ('MLP, 1 hidden layer, 4 hidden units',
          'MLP, 2 hidden layer, 4 hidden units',
          'MLP, 3 hidden layer, 4 hidden units',
          'MLP, 1 hidden layer, 8 hidden units')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = data[:, 0], data[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=cmap_light, alpha=0.8)
    ax.scatter(X0, X1, c=categories, cmap=cmap_bold, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
