from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.linalg import eig
from sklearn.neighbors import NearestCentroid
import numpy as np


def centering_matrix(n):
    return np.eye(n) - 1 / n


class Kfda(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2, kernel=lambda a, b: np.inner(a, b)):
        self.kernel = kernel
        self.n_components = n_components

        if kernel is None:
            self.kernel = lambda a, b: np.inner(a, b)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        n_datapoints = y.size

        classes_datapoints = {}

        # Find M and N
        M = np.zeros((n_datapoints, n_datapoints))
        N = np.zeros((n_datapoints, n_datapoints))

        m_star = np.array([self.kernel(datapoint, X).mean()
                          for datapoint in X])

        for label in self.classes_:
            class_datapoints = classes_datapoints[label] = X[y == label]
            n_class_datapoints = class_datapoints.shape[0]

            H = centering_matrix(n_class_datapoints)
            K = np.array([self.kernel(datapoint, class_datapoints)
                         for datapoint in X])
            m = K.mean(-1)

            m_centered = m - m_star
            M += np.outer(m_centered, m_centered)
            N += K.dot(H).dot(K.T)

        # Find weights
        w, v = eig(M, N)

        w_finite_mask = ~(np.isinf(w) | np.isnan(w))
        w = w[w_finite_mask]
        v = v[:, w_finite_mask]
        v = v[:, np.argsort(w)[:-self.n_components-1:-1]]

        self.weights_ = v.astype(np.float64)

        # Compute centers
        self.centers_ = np.array([
            np.mean([
                self.weights_.T.dot(self.kernel(X, class_datapoint))
                for class_datapoint in classes_datapoints[label]], 0)
            for label in self.classes_])

        return self

    def project_point_1d(self, x):
        return np.dot(self.kernel(self.X_, x), self.weights_)

    def project_points(self, X):
        return np.array([self.project_point_1d(x) for x in X])

    def predict(self, X):
        check_is_fitted(self)

        X = check_array(X)
        clf = NearestCentroid()
        clf.fit(self.centers_, self.classes_)

        projected_points = self.project_points(X)
        predictions = clf.predict(projected_points)

        return predictions
