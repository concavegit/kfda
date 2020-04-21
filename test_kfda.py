from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

from kfda import Kfda


def rbf(variance):
    return lambda a, b: np.exp((((a - b) / np.sqrt(variance))**2).sum(-1) / 2)


def poly(d):
    return lambda a, b: (np.inner(a, b) + 1)**d


# cls = Kfda(kernel=poly(9), n_components=8)
cls = Kfda(kernel=rbf(1e6), n_components=9)
# cls = Kfda(kernel=None, n_components=8)
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=2000, test_size=2000, stratify=y)

print('fitting')
cls.fit(X_train, y_train)
print('scoring')

test_score = cls.score(X_test, y_test)
train_score = cls.score(X_train, y_train)
print(f'Test Score: {test_score}')
print(f'Train Score: {train_score}')

test_embeddings = cls.project(X_test)
train_embeddings = cls.project(X_train)
np.savetxt('test_embeddings.tsv', test_embeddings, delimiter='\t')
np.savetxt('train_embeddings.tsv', train_embeddings, delimiter='\t')
np.savetxt('test_labels.tsv', y_test, delimiter='\t', fmt="%s")
np.savetxt('train_labels.tsv', y_train, delimiter='\t', fmt="%s")
