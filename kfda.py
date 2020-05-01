from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.sparse.linalg import eigs
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import pairwise_kernels
import numpy as np


def centering_matrix(n):
    return np.eye(n) - 1 / n


def complex_to_real(x):
    return np.concatenate([x.real, x.imag], -1)


class ComplexNearestCentroid(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.clf_ = NearestCentroid()

    def fit(self, X, y):
        X, y = check_X_y(complex_to_real(X), y)
        self.clf_.fit(X, y)

    def predict(self, X):
        check_is_fitted(self)

        X = check_array(complex_to_real(X))

        return self.clf_.predict(X)


class Kfda(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2, kernel='linear', **kwargs):
        self.kernel = kernel
        self.n_components = n_components
        self.kwargs = kwargs

        if kernel is None:
            self.kernel = 'linear'

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        n_datapoints = y.size
        n_classes = self.classes_.size

        classes_datapoints = {}

        m_star = pairwise_kernels(
            X, X, metric=self.kernel, **self.kwargs).mean(1)

        m_classes = np.empty((n_classes, n_datapoints))
        M_classes = np.empty((n_classes, n_datapoints, n_datapoints))
        N_classes = np.empty((n_classes, n_datapoints, n_datapoints))

        for i, label in enumerate(self.classes_):
            class_datapoints = classes_datapoints[label] = X[y == label]
            n_class_datapoints = class_datapoints.shape[0]

            H = centering_matrix(n_class_datapoints)
            K = pairwise_kernels(X, class_datapoints,
                                 metric=self.kernel, **self.kwargs)
            K.mean(1, out=m_classes[i])

            m_centered = m_classes[i] - m_star
            np.outer(m_centered, m_centered, M_classes[i])
            K.dot(H).dot(K.T, N_classes[i])

        M = M_classes.sum(0)
        N = N_classes.sum(0)

        # Find weights
        w, v = eigs(M, self.n_components, N, which='LM')

        self.weights_ = v.real

        # Compute centers
        self.centers_ = m_classes.dot(self.weights_)

        # Train nearest centroid classifier
        self.clf_ = ComplexNearestCentroid()
        self.clf_.fit(self.centers_, self.classes_)

        return self

    def project(self, X):
        check_is_fitted(self)
        return pairwise_kernels(X, self.X_, metric=self.kernel, **self.kwargs).dot(self.weights_)

    def predict(self, X):
        check_is_fitted(self)

        X = check_array(X)

        projected_points = self.project(X)
        predictions = self.clf_.predict(projected_points)

        return predictions
