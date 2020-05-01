import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

import kfda

X, y = fetch_openml('Olivetti_Faces', version=1, return_X_y=True)
y = y.astype('u8')
mask = y < 7
y = y[mask]
X = X[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=.7, stratify=y)

cls = kfda.Kfda(kernel=kfda.rbf(1e8), n_components=2)
print('fitting...')
cls.fit(X_train, y_train)
print('fitted, scoring...')

test_score = cls.score(X_test, y_test)
train_score = cls.score(X_train, y_train)
print(f'Test Score: {test_score}')
print(f'Train Score: {train_score}')

train_embeddings = cls.project(X_train)
test_embeddings = cls.project(X_test)


colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
unique_labels = np.unique(y)

for color, label in zip(colors, unique_labels):
    train_embeddings = X_train[y_train == label]
    test_embeddings = X_test[y_test == label]
    plt.scatter(train_embeddings[:, 0], train_embeddings[:, 1],
                c=color, marker='^', label=f'Train {label}')
    plt.scatter(
        test_embeddings[:, 0], test_embeddings[:, 1], c=color, label=f'Test {label}')

ax = plt.gca()
ax.set_title(f'Test and Train Embeddings, Accuracy: {test_score:.2g}')
plt.legend()

plt.savefig('demo.png')
plt.show()
