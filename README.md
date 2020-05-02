# Kernel FDA

[![PyPI version](https://badge.fury.io/py/kfda.svg)](https://badge.fury.io/py/kfda)

This repository implements Kernel Fisher Discriminant Analysis (Kernel FDA) as described in [https://arxiv.org/abs/1906.09436](https://arxiv.org/abs/1906.09436).
FDA, equivalent to Linear Discriminant Analysis (LDA), is a classification method that projects vectors onto a smaller subspace.
This subspace is optimized to maximize between-class scatter and minimize within class scatter, making it an effective classification method.
Kernel FDA improves on regular FDA by enabling nonlinear subspaces using the [kernel trick](https://en.wikipedia.org/wiki/Kernel_method).
The
[`examples`](https://github.com/concavegit/kfda/tree/master/examples)
and the
[Colab Notebook](https://colab.research.google.com/drive/1nnVphyZ_0QKYZbmdJaIBjm-zYO4xwF0b).
demonstrate 97% accuracy by only training on one-seventh of the MNIST dataset.

FDA and Kernel FDA classify vectors by comparing their projection in the fisher subspace to class centroids, adding a new class is just a matter of adding a new centroid.
Thus, this model is implemented here with the hope of using Kernel FDA as a oneshot learning algorithm.

## Usage
`Kfda` uses `scikit-learn`'s interface.

- Initializing: `cls = Kfda(n_components=2, kernel='linear')` for a classifier that a linear kernel with 2 components.
  For kernel of degree 2, use `Kfda(n_components=2, kernel='poly', degree=2)` for a polynomial kernel of degree 2.
  See https://scikit-learn.org/stable/modules/metrics.html#polynomial-kernel for a list of kernels and their parameters, or the [source code docstrings](https://github.com/concavegit/kfda/blob/master/kfda/kfda.py) for a complete description of the parameters.

- Fitting: `cls.fit(X, y)`

- Prediction: `cls.predict(X)`

- Scoring: `cls.score(X, y)`

## Examples
See [`examples`](https://github.com/concavegit/kfda/tree/master/examples) for examples on MNIST, faces, and oneshot learning.

After running them, you can plug corresponding pairs of generated
`*embeddings.tsv` and `*labels.tsv` into Tensorflow's
[Embedding Projector](https://projector.tensorflow.org/)
to visualize the embeddings.
For example, running `mnist.py` and then loading
`mnist_test_embeddings.tsv` and `mnist_test_labels.tsv` shows the
following using the UMAP visualizer:

![MNIST Kernel FDA embeddings](https://github.com/concavegit/kfda/blob/master/img/mnist.png?raw=true)

## Notebook
Another place to see example usage is the
[Colab Notebook](https://colab.research.google.com/drive/1nnVphyZ_0QKYZbmdJaIBjm-zYO4xwF0b).

## Caveats
Similar to SVM, the most glaring constraint of KFDA is the memory limit in training.
Training a Kernel FDA classifier requires creating matrices that are `n_samples` by `n_samples` large, meaning the memory requirement grows with respect to `O(n_samples^2)`.

The accuracy, while high (0.97 on MNIST), seems to be limited by the training set size.
With a training size of 10000 and a testing size of 60000, performance on MNIST averages around 0.97 accuracy using 9 fisher directions and the RBF kernel:

```python
cls = Kfda(kernel='rbf', n_components=9)
```

This may be due to the constrained training size.
Accuracy can be improved without increasing training size by implementing invariant kernels that would implicitly handle scale and rotation without requiring an extended dataset.

## Oneshot Learning
Oneshot learning means that an algorithm can learn a new class with as little as one sample.
This is possible for Kernel FDA because it finds a subspace that purposefully spreads out distinct classes.
Introducing a new label involves simply adding another centroid for use in prediction.
See the
[Colab Notebook](https://colab.research.google.com/drive/1nnVphyZ_0QKYZbmdJaIBjm-zYO4xwF0b).
or the
[example](https://github.com/concavegit/kfda/blob/master/examples/mnist_oneshot.py) for examples.
