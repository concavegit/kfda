from sklearn.datasets import fetch_openml
from sklearn.utils.multiclass import unique_labels
import numpy as np

from kfda import Kfda

cls = Kfda()
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

new_dataset = []
new_labels = []
for label in unique_labels(y):
    sample = list(X[y==label][:10])
    new_dataset.extend(sample)
    new_labels.extend([label] * 10)


new_dataset = np.array(new_dataset)
new_labels = np.array(new_labels)
X = new_dataset
y = np.array(new_labels)

cls.fit(X, y)
