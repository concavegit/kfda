from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

from kfda import Kfda


X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X = (X - 127.5) / 127.5


# If you don't have that much memory available, lower this number.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=10000, stratify=y)


# Remove nines from the training set
X_train_nines = X_train[y_train == '9']
X_train = X_train[y_train != '9']
y_train_nines = y_train[y_train == '9']
y_train = y_train[y_train != '9']

# Train
cls = Kfda(kernel='rbf', n_components=8)
print('Fitting...')
train_embeddings = cls.fit_transform(X_train, y_train)

# Show the algorithm a single nine.
# The same weights are used and there is no lengthy retraining.
print('Adding the 9 class')
cls.fit_additional(X_train_nines[:1], y_train_nines[:1])
print('Scores:')

# Show the results
test_score = cls.score(X_test, y_test)
print(f'Test Score: {test_score}')
train_score = cls.score(X_train, y_train)
print(f'Train Score: {train_score}')

print('Generating embeddings...')
test_embeddings = cls.transform(X_test)

np.savetxt('mnist_fewshot_test_embeddings.tsv',
           test_embeddings, delimiter='\t')
np.savetxt('mnist_fewshot_test_labels.tsv', y_test, delimiter='\t', fmt="%s")
np.savetxt('mnist_fewshot_train_labels.tsv', y_train, delimiter='\t', fmt="%s")
print('Embeddings saved to *.tsv! Plug them into https://projector.tensorflow.org/ for embedding visualizations.')
