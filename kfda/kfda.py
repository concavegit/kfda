from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.sparse.linalg import eigs
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import pairwise_kernels
import numpy as np


def centering_matrix(n):
    """Return a centering matrix H such that H * X = X - mean(X, axis=0).

    Parameters
    ----------
    n : int the amount of rows the centering matrix should have

    Returns
    -------
    H : ndarray of shape (n, n)

    Examples
    --------
    >>> centering_matrix(4)
    array([[ 0.75, -0.25, -0.25, -0.25],
           [-0.25,  0.75, -0.25, -0.25],
           [-0.25, -0.25,  0.75, -0.25],
           [-0.25, -0.25, -0.25,  0.75]])
    """

    return np.eye(n) - 1 / n


def vectorize_complex(X):
    """Make a new array of the real components concatenated with the
    imaginary components of a complex array.

    The real and imaginary components are concatenated along the last
    axis.

    Parameters
    ----------
    X : array-like the input array

    Returns
    -------
    A : ndarray

    Examples
    --------
    >>> X = np.array([[1 + 1j, 2 + 3j], [0 + 1j, 3 + 2j], [1 + 2j, 2 + 2j]])
    >>> vectorize_complex(X)
    array([[1., 2., 1., 3.],
           [0., 3., 1., 2.],
           [1., 2., 2., 2.]])
    """
    return np.concatenate([X.real, X.imag], -1)


class ComplexNearestCentroid(BaseEstimator, ClassifierMixin):
    """Complex nearest centroid classifier.

    This is a wrapper of NearestCentroid from scikit-learn that
    enables complex components by turning each complex element into
    two real components.

    Parameters
    ----------
    **kwds : optional keyword parameters
        Parameters to send to scikit-learn's NearestCentroid.

    Attributes
    ----------
    clf_ : NearestCentroid
        The internal NearestCentroid object.

    See also
    --------
    sklearn.neighbors.NearestCentroid: nearest centroid classifier
    """

    def __init__(self, **kwds):
        self.clf_ = NearestCentroid(**kwds)

    def fit(self, X, y):
        """Fit the ComplexNearestCentroid model according to the given
        training data.

        The parameters are the same as scikit-learn's NearestCentroid,
        but X is complex.

        Parameters
        ----------
        X : {array-like, sparse matrix} complex of shape (n_samples,
            n_features) Training vector, where n_samples is the number
            of samples and n_features is the number of features.
        y : array, shape = [n_samples]
            Target values (integers)
        """
        X, y = check_X_y(vectorize_complex(X), y)
        self.clf_.fit(X, y)

    def predict(self, X):
        """Perform classification on an array of test vectors X.
        The predicted class C for each sample in X is returned.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        C : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)

        X = check_array(vectorize_complex(X))

        return self.clf_.predict(X)


class Kfda(BaseEstimator, ClassifierMixin):
    """Kernel Fisher Discriminant Analysis classifier.

    Each class is represented by a centroid using projections in a
    Hilbert space.

    See https://arxiv.org/abs/1906.09436 for mathematical details.

    Parameters
    ----------
    n_components : int the amount of Fisher directions to use.
        This is limited by the amount of classes minus one.
        See the paper for further discussion of this limit.

    kernel : str, ['linear', 'polynomial', 'sigmoid', 'rbf','laplacian', 'chi2']
        The kernel to use.
        Use **kwds to pass arguments to these functions.
        See
        https://scikit-learn.org/stable/modules/metrics.html#polynomial-kernel
        for more details.

    **kwargs : parameters to pass to the kernel function.

    Attributes
    ----------
    centroids_ : array_like of shape (n_classes, n_samples) that
        represent the class centroids.

    classes_ : array of shape (n_classes,)
        The unique class labels

    weights_ : array of shape (n_components, n_samples) that
        represent the fisher components.

    clf_ : The internal ComplexNearestCentroid classifier used in prediction.
    """

    def __init__(self, n_components=2, kernel='linear', **kwargs):
        self.kernel = kernel
        self.n_components = n_components
        self.kwargs = kwargs

        if kernel is None:
            self.kernel = 'linear'

    def fit(self, X, y):
        """
        Fit the NearestCentroid model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array, shape = [n_samples]
            Target values (integers)
        """
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
        w, self.weights_ = eigs(M, self.n_components, N, which='LM')

        # Compute centers
        self.centroids_ = m_classes.dot(self.weights_)

        # Train nearest centroid classifier
        self.clf_ = ComplexNearestCentroid()
        self.clf_.fit(self.centroids_, self.classes_)

        return self

    def project(self, X):
        """Project the points in X onto the fisher directions.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features) to be
            projected onto the fisher directions.
        """
        check_is_fitted(self)
        return pairwise_kernels(
            X, self.X_, metric=self.kernel, **self.kwargs
        ).dot(self.weights_)

    def predict(self, X):
        """Perform classification on an array of test vectors X.

        The predicted class C for each sample in X is returned.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        C : ndarray of shape (n_samples,)
        """

        check_is_fitted(self)

        X = check_array(X)

        projected_points = self.project(X)
        predictions = self.clf_.predict(projected_points)

        return predictions
