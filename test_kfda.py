from sklearn.datasets import fetch_openml
from sklearn.utils.multiclass import unique_labels
import numpy as np
import matplotlib.pyplot as plt

from kfda import Kfda


def rbf(variance):
    return lambda a, b: np.exp((((a - b) / np.sqrt(variance))**2).sum(-1) / 2)


cls = Kfda(kernel=rbf(1e8))
X = np.load('data.npy')
y = np.load('labels.npy')

print("Fitting")
cls.fit(X, y)

points = cls.project_points(X).reshape(10, 10, -1)

for pointset in points:
    plt.plot(pointset[:, 0], pointset[:, 1], '.')
plt.show()
