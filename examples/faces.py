import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

from kfda import Kfda

X, y = fetch_openml('Olivetti_Faces', version=1, return_X_y=True)
y = y.astype('u8')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=.7, stratify=y)

cls = Kfda(kernel='poly', n_components=50, degree=2)
cls.fit(X_train, y_train)
print('fitted, scoring...')

test_score = cls.score(X_test, y_test)
print(f'Test Score: {test_score}')
train_score = cls.score(X_train, y_train)
print(f'Train Score: {train_score}')

print('Generating embeddings...')
test_embeddings = cls.project(X_test)
train_embeddings = cls.project(X_train)

# Plug these into https://www.google.com/search?q=tensorflow+projector
# to visualize embeddings.
np.savetxt('face_test_embeddings.tsv', test_embeddings, delimiter='\t')
np.savetxt('face_train_embeddings.tsv', train_embeddings, delimiter='\t')
np.savetxt('face_test_labels.tsv', y_test, delimiter='\t', fmt="%s")
np.savetxt('face_train_labels.tsv', y_train, delimiter='\t', fmt="%s")
print('Embeddings saved to *.tsv! Plug them into https://projector.tensorflow.org/ for embedding visualizations.')
