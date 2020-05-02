# Kernel FDA

[![PyPI version](https://badge.fury.io/py/kfda.svg)](https://badge.fury.io/py/kfda)

This repository implements Kernel Fisher Discriminant Analysis (Kernel FDA) as described in [https://arxiv.org/abs/1906.09436](https://arxiv.org/abs/1906.09436).
FDA, equivalent to Linear Discriminant Analysis (LDA), is a classification method that projects vectors onto a smaller subspace.
This subspace is optimized to maximize between-class scatter and minimize the within class scatter, making it an effective classification method.
Kernel FDA improves on regular FDA by enabling evaluation in a nonlinear subspace using the [kernel trick](https://en.wikipedia.org/wiki/Kernel_method).

## Examples
See [`examples`](examples) for examples.

After running them, you can plug corresponding pairs of generated
`*embeddings.tsv` and `*labels.tsv` into Tensorflow's
[Embedding Projector](https://projector.tensorflow.org/)
to visualize the embeddings.
For example, running `mnist.py` and then loading
`mnist_test_embeddings.tsv` and `mnist_test_labels.tsv` shows the
following using the UMAP visualizer:

![MNIST Kernel FDA embeddings](https://github.com/concavegit/kfda/blob/master/img/mnist.png?raw=true)

## Notebook
Alternatively, see the
[Colab Notebook](https://colab.research.google.com/drive/1nnVphyZ_0QKYZbmdJaIBjm-zYO4xwF0b#scrollTo=6Pfpr7DDQota).
