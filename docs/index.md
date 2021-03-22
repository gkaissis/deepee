# About

deepee is a small library for differentially private deep learning in PyTorch. More precisely, deepee implements the Differentially Private Stochastic Gradient Descent (DP-SGD) algorithm. Despite the name, deepee works with any (first order) optimizer, including Adam, AdaGrad, etc. 

It wraps a regular PyTorch model and takes care of calculating per-sample gradients, clipping, noising and accumulating gradients with an API that closely mimics the original algorithm's description.

# Installation
Since deepee uses a cryptographically secure random number generator and works on both CPU and GPU, it does not come bundled with any specific packages. You can install deepee with:

`pip install deepee`

# Authors
deepee was written by George Kaissis and Alex Ziller, the original creators of PriMIA. 