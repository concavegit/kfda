# Kernel FDA
This repository implements Kernel FDA as described in [https://arxiv.org/abs/1906.09436](https://arxiv.org/abs/1906.09436).

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
