# `deepee`

`deepee` is a library for differentially private deep learning in PyTorch. More precisely, `deepee` implements the Differentially Private Stochastic Gradient Descent (DP-SGD) algorithm originally described by [Abadi et al.](https://arxiv.org/pdf/1607.00133.pdf). Despite the name, `deepee` works with any (first order) optimizer, including Adam, AdaGrad, etc. 

It wraps a regular `PyTorch` model and takes care of calculating per-sample gradients, clipping, noising and accumulating gradients with an API which closely mimics the `PyTorch` API of the original model.

Check out the documentation [here]()

