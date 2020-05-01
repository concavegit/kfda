from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

import kfda


X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X = (X - 127.5) / 127.5

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=2000, test_size=2000, stratify=y)

# for normalization in range(10, 20):
for components in range(1, 10):
    cls = kfda.Kfda(kernel='polynomial', n_components=components, degree=2)
    print(f'fitting components={components}')
    cls.fit(X_train, y_train)
    print('Scores:')

    test_score = cls.score(X_test, y_test)
    train_score = cls.score(X_train, y_train)
    print(f'Test Score: {test_score}')
    print(f'Train Score: {train_score}')

# test_embeddings = cls.project(X_test)
# train_embeddings = cls.project(X_train)

# np.savetxt('test_embeddings.tsv', test_embeddings, delimiter='\t')
# np.savetxt('train_embeddings.tsv', train_embeddings, delimiter='\t')
# np.savetxt('test_labels.tsv', y_test, delimiter='\t', fmt="%s")
# np.savetxt('train_labels.tsv', y_train, delimiter='\t', fmt="%s")
