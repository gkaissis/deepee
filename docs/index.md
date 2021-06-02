# About

`deepee` is a library for differentially private deep learning in PyTorch. More precisely, `deepee` implements the Differentially Private Stochastic Gradient Descent (DP-SGD) algorithm originally described by [Abadi et al.](https://arxiv.org/pdf/1607.00133.pdf). Despite the name, `deepee` works with any (first order) optimizer, including Adam, AdaGrad, etc. 

It wraps a regular `PyTorch` model and takes care of calculating per-sample gradients, clipping, noising and accumulating gradients with an API which closely mimics the `PyTorch` API of the original model.

# Design principles

The DP-SGD algorithm requires two key steps, which differ from "normal" neural network training. For every minibatch, the algorithm requires to:

1. Obtain the gradients of each individual sample in the batch to calculate their L2-norm and _clip_ it to a pre-set threshold
2. Average the gradients and add Gaussian noise to the average before taking an optimisation step.

The first of these two steps is complicated, as deep learning frameworks like `PyTorch` are not designed to provide per-sample gradients by default. `deepee` works around this limitation by creating a "copy" of the network for every sample in the batch and then doing a parallel forward and backward pass on each sample simultaneously. This technique has two benefits:

1. `deepee` works with *any* neural network architecture that can be defined in `PyTorch` without any user modification
2. The process is very efficient as it happens in parallel and doesn't create significant memory overhead. The copies of the models are kept as references to the original weights and thus it's not required to create "real" model copies. The only memory overhead thus results from keeping the per-sample gradients in memory for a short time during each batch.


# Basic usage
The key component of the framework is the `PrivacyWrapper` class, which wraps a `PyTorch module` and takes care of the training process behind the scenes. The clipping norm and noise multiplier parameters can be set here. `deepee` will automatically check if a model contains incompatible layers (such as Batch Normalisation) and throw an error. If such layers exist in the network, the `ModelSurgeon` can be used to replace them with compatible layers such as Group Normalisation, Instance Normalisation etc.. `deepee` also offers automatic privacy accounting and will interrupt the training (and optionally save the last model) when the privacy budget is exhausted. This process is abstracted using the `PrivacyWatchdog` class. 

An example for the use of `deepee` showcasing all these concepts can be found [here](examples.md).

# For paper readers
If you would like to reproduce the experiments from our paper, please switch to the `paper` branch. Instructions for reproduction can be found in the README of the branch.
# Installation
You can install deepee with:

`pip install deepee`

`deepee` does not come with any hard-coded dependencies to maintain high compatibility, so these must be installed separately. `deepee` is tested with `pytorch>1.8`(CPU and GPU) on Ubuntu Linux (min. 20.04) and MacOS >11.3. No GPU support is available for MacOS. 

The cryptographically secure random number generator (CSPRNG) used by `deepee` must also be installed separately. The framework will function without the CSPRNG, however we stress that it should **not** be used in production environments without this feature. 

To install `torchcsprng`, follow the instructions on [this](https://pypi.org/project/torchcsprng/) page.

Lastly, `SciPy` is required, which can be installed with `pip install scipy`.

For Linux, the full installation can therefore look something like this:

```
pip3 install torchcsprng==0.2.0 torch==1.8.0+cu101 torchvision==0.9.0 -f https://download.pytorch.org/whl/cu101/torch_stable.html 
pip3 install scipy deepee
```


# Contributing
`deepee` is licensed under the Apache 2.0. license. Contributions are welcome via PR. To contribute, please install the following additional dependencies: `mypy, black, pytest, testfixtures`. Packaging is carried out using `poetry`. 

