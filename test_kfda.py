from sklearn.datasets import fetch_openml
from sklearn.utils.multiclass import unique_labels
import numpy as np
import matplotlib.pyplot as plt

from kfda import Kfda

cls = Kfda()
# X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# new_dataset = []
# new_labels = []
# for label in unique_labels(y):
#     sample = list(X[y==label][:100])
#     new_dataset.extend(sample)
#     new_labels.extend([label] * 100)


# new_dataset = np.array(new_dataset)
# new_labels = np.array(new_labels)
# np.save('data', new_dataset)
# np.save('labels', new_labels)
X = np.load('data.npy')
y = np.load('labels.npy')

print("Fitting")
cls.fit(X, y)
points = cls.project_points(X).reshape(10, 10, -1)
for pointset in points:
    plt.plot(pointset[:, 0], pointset[:, 1], '.')
plt.show()
