from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.linalg import eig
import numpy as np


class Kfda(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2, kernel=lambda a, b: np.inner(a, b)):
        self.kernel = kernel
        self.n_components = n_components

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        n_datapoints = y.size
        m_star = np.zeros(n_datapoints)

        for i in range(n_datapoints):
            for datapoint in X:
                m_star[i] += self.kernel(X[i], datapoint) / n_datapoints

        m_classes = {}
        H_classes = {}
        K_classes = {}

        for label in y:
            m_class = m_classes[label] = np.zeros(n_datapoints)
            class_datapoints = X[y == label]
            n_class_datapoints = class_datapoints.shape[0]
            H_classes[label] = np.eye(
                n_class_datapoints) - 1 / n_class_datapoints
            K_class = K_classes[label] = np.zeros(
                (n_datapoints, n_class_datapoints))

            for i, datapoint in enumerate(X):
                for j, (class_datapoint) in enumerate(class_datapoints):
                    m_class[i] += self.kernel(
                        X[i], class_datapoint) / n_class_datapoints

                    K_class[i, j] = self.kernel(datapoint, class_datapoint)

        M = np.zeros((n_datapoints, n_datapoints))
        N = np.zeros((n_datapoints, n_datapoints))

        for label in y:
            m_class = m_classes[label]
            m_class_centered = m_class - m_star
            M += np.outer(m_class_centered, m_class_centered)

            K_class = K_classes[label]
            H_class = H_classes[label]
            N += K_class.dot(H_class).dot(K_class.T)

        w, v = eig(M, N)
        v = v[np.argsort(w)]

        self.weights_ = v

        return self

    def predict(self, X):
        check_is_fitted(self)

        X = check_array(X)

        return 0
