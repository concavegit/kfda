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

        m_star = np.array([self.kernel(datapoint, X).mean()
                          for datapoint in X])

        m_classes = {}
        H_classes = {}
        K_classes = {}

        classes_datapoints = {}
        n_classes_datapoints = {}

        for label in self.classes_:
            class_datapoints = classes_datapoints[label] = X[y == label]
            n_class_datapoints = n_classes_datapoints[label] = class_datapoints.shape[0]
            H_classes[label] = np.eye(
                n_class_datapoints) - 1 / n_class_datapoints

            K_classes[label] = np.array(
                [self.kernel(datapoint, class_datapoints) for datapoint in X])
            m_classes[label] = K_classes[label].mean(-1)

        M = np.zeros((n_datapoints, n_datapoints))
        N = np.zeros((n_datapoints, n_datapoints))

        M = np.sum([np.outer(m_class - m_star, m_class - m_star)
                   for m_class in m_classes.values()], 0)
        N = np.sum([K_classes[label].dot(H_classes[label]).dot(
            K_classes[label].T) for label in self.classes_], 0)


        w, v = eig(M, N)

        w_mask = ~np.isinf(w) & ~np.isnan(w)
        w = w[w_mask]
        v = v[:, w_mask]
        v = v[:, np.argsort(w)[:-self.n_components-1:-1]]

        self.weights_ = v.astype(np.float64)

        self.centers_ = np.empty((self.classes_.size, self.n_components))

        for i, label in enumerate(self.classes_):
            for class_datapoint in classes_datapoints[label]:
                self.centers_[
                    i] += self.weights_.T.dot(self.kernel(X, class_datapoint))
            self.centers_[i] /= n_classes_datapoints[label]

        return self

    def project_point_1d(self, x):
        return np.dot(self.kernel(self.X_, x), self.weights_)

    def project_points(self, X):
        return np.array([self.project_point_1d(x) for x in X])

    def predict(self, X):
        check_is_fitted(self)

        X = check_array(X)

        return 0
