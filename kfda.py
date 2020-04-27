from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.linalg import eig
from sklearn.neighbors import NearestCentroid
import numpy as np


def centering_matrix(n):
    return np.eye(n) - 1 / n


def kernel_matrix(kernel, a, b):
    rows = a.shape[0]
    cols = b.shape[0]

    a_repeated = np.repeat(a, cols, 0)
    b_repeated = np.tile(b, (rows, 1))

    return kernel(a_repeated, b_repeated).reshape(rows, cols)


def rbf(variance):
    return lambda a, b: np.exp((((a - b) / np.sqrt(variance))**2).sum(-1) / 2)


def linear(a, b):
    return (a  * b).sum(-1)


def poly(d):
    return lambda a, b: (linear(a, b) + 1)**d


class Kfda(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2, kernel=lambda a, b: np.inner(a, b)):
        self.kernel = kernel
        self.n_components = n_components

        if kernel is None:
            self.kernel = linear

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        n_datapoints = y.size
        n_classes = self.classes_.size

        classes_datapoints = {}

        m_star = kernel_matrix(self.kernel, X, X).mean(1)

        m_classes = np.empty((n_classes, n_datapoints))
        M_classes = np.empty((n_classes, n_datapoints, n_datapoints))
        N_classes = np.empty((n_classes, n_datapoints, n_datapoints))

        for i, label in enumerate(self.classes_):
            class_datapoints = classes_datapoints[label] = X[y == label]
            n_class_datapoints = class_datapoints.shape[0]

            H = centering_matrix(n_class_datapoints)
            K = kernel_matrix(self.kernel, X, class_datapoints)
            K.mean(1, out=m_classes[i])

            m_centered = m_classes[i] - m_star
            np.outer(m_centered, m_centered, M_classes[i])
            K.dot(H).dot(K.T, N_classes[i])

        M = M_classes.sum(0)
        N = N_classes.sum(0)

        # Find weights
        w, v = eig(M, N)

        v = v[:, :self.n_components]

        self.weights_ = v.real

        # Compute centers
        self.centers_ = m_classes.dot(self.weights_)

        # Train nearest centroid classifier
        self.clf_ = NearestCentroid()
        self.clf_.fit(self.centers_, self.classes_)

        return self

    def project(self, X):
        check_is_fitted(self)
        return kernel_matrix(self.kernel, X, self.X_).dot(self.weights_)

    def predict(self, X):
        check_is_fitted(self)

        X = check_array(X)

        projected_points = self.project(X)
        predictions = self.clf_.predict(projected_points)

        return predictions
