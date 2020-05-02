from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

from kfda import Kfda


X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X = (X - 127.5) / 127.5

# train_size=8000 takes up around 12 GB of memory.
# If you don't have that available, lower this number.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=10000, stratify=y)

cls = Kfda(kernel='rbf', n_components=9)
print('Fitting...')
train_embeddings = cls.fit_transform(X_train, y_train)
print('Scores:')
test_score = cls.score(X_test, y_test)
print(f'Test Score: {test_score}')
train_score = cls.score(X_train, y_train)
print(f'Train Score: {train_score}')

print('Generating embeddings...')

test_embeddings = cls.transform(X_test)

np.savetxt('mnist_test_embeddings.tsv', test_embeddings, delimiter='\t')
np.savetxt('mnist_test_labels.tsv', y_test, delimiter='\t', fmt="%s")
np.savetxt('mnist_train_labels.tsv', y_train, delimiter='\t', fmt="%s")
print('Embeddings saved to *.tsv! Plug them into https://projector.tensorflow.org/ for embedding visualizations.')
