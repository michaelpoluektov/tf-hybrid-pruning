# tf-hybrid-pruning

TB2 individual project -- exploring hybrid optimisation methods for data-free inference optimisations on mobile devices

## Motivation

The goal of this project is to create a data-free technique to compress Convolutional Neural Networks (CNN) for inference on mobile hardware. The intended use case is to support optimisations for a wide variety of hardware architectures and resource constraints.
This project would enable developers to train a single network then optimise it for deployment for arbitrary devices such as embedded systems, mobile phones or web applications.

## Approach

There are 3 steps to this project:

1. Reducing the complexity of our model
2. Implementing TFLite operators that can take advantage of that reduced complexity
3. Tweaking step 1 to fit step 2 better and vice-versa

### Reducing complexity

So far the following ways of reducing model complexity are being considered:

- Weight pruning based on magnitude
- Activation pruning based on magnitude
- Re-structuring the model under the assumptions that certain activations can be treated as constants
- Re-structuring the model under the assumptions that certain activations can be assumed to be identical

The first 2 points can be done using simple binary search on each layer, pruning based on the percentile of magnitudes.
The 3rd point can be done using binary search, pruning the activations with the lowest standard deviations. This may be sub-optimal if the standard deviation is not representative of the activation's importance (such as the edges of the image)
The 4th point can be done by finding highly correlated activations, but that may be too slow.

### Implementing TFLite operators

This will most likely involve implementing a sparse Conv2D layer with activation pruning. We assume the possibility to use P and V extensions.

### Tweaking

Can we co-design the pruning method and the operator?

## File structure

>

    .
    ├── setup.txt                    # Very non-descriptive setup guide
    ├── cifar/                       # Pruning on a CIFAR model
    ├── mnist_test/                  # Used to test if the installation
    └── README.md

>
