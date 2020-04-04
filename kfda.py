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
            m_star[i] = self.kernel(X[i], X).mean()

        m_classes = {}
        H_classes = {}
        K_classes = {}

        classes_datapoints = {}
        n_classes_datapoints = {}

        for label in self.classes_:
            m_class = m_classes[label] = np.zeros(n_datapoints)
            class_datapoints = classes_datapoints[label] = X[y == label]
            n_class_datapoints = n_classes_datapoints[label] = class_datapoints.shape[0]
            H_classes[label] = np.eye(
                n_class_datapoints) - 1 / n_class_datapoints
            K_class = K_classes[label] = np.zeros(
                (n_datapoints, n_class_datapoints))

            for i, datapoint in enumerate(X):
                K_class[i] = self.kernel(datapoint, class_datapoints)
                m_class[i] = K_class[i].mean()

        M = np.zeros((n_datapoints, n_datapoints))
        N = np.zeros((n_datapoints, n_datapoints))

        for label in self.classes_:
            m_class = m_classes[label]
            m_class_centered = m_class - m_star
            M += np.outer(m_class_centered, m_class_centered)

            K_class = K_classes[label]
            H_class = H_classes[label]
            N += K_class.dot(H_class).dot(K_class.T)

        w, v = eig(M, N)
        v = v[:, np.argsort(w)[:-self.n_components-1:-1]]

        self.weights_ = v

        self.centers_ = np.empty((self.classes_.size, self.n_components))

        for i, label in enumerate(self.classes_):
            for class_datapoint in classes_datapoints[label]:
                self.centers_[
                    i] += self.weights_.T.dot(self.kernel(X, class_datapoint))
            self.centers_[i] /= n_classes_datapoints[label]

        return self

    def project_point_1d(self, x):
        return self.weights_ * self.kernel(self.X_, x)

    def predict(self, X):
        check_is_fitted(self)

        X = check_array(X)

        return 0
