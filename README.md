# tf-hybrid-pruning

TB2 individual project -- exploring hybrid optimisation methods for data-free inference optimisations on mobile devices.

## Motivation

The goal of this project is to create a data-free technique to compress Convolutional Neural Networks (CNN) for inference on mobile hardware. The intended use case is to support optimisations for a wide variety of hardware architectures and resource constraints. This project would enable developers to train a single network then optimise it for deployment for arbitrary devices such as embedded systems, mobile phones or web applications.

## Approach

As of now the project only supports TensorFlow and ResNet50. Finding compression factors for other network architectures should be straightforward, but the user will need to define the decomposition block in the script that generates the model. This limitation will be rectified as soon as TensorFlow adds support for "model surgery", which is currently being worked according to a recent PR [ADD PR LINK].

This project implements a hybrid compression technique based on tensor decompositions with added sparsity. The decomposition technique in use is a modified version of a Tucker decomposition, and the pruning structure can be arbitrarily adjusted to fit the requirements of the target hardware. Each `Conv2D` layer is split into four smaller `Conv2D` layers as represented by the diagram below:

[ADD DIAGRAM]

This framework supports two compression approaches:

- An evaluation led approach, where we fix the maximum acceptable loss of accuracy and maximise the compression factor. This approach requires access to a minimal test dataset.
- A fixed compression approach, where the user defines a compression factor based on memory constraints or inference times, and the library aims to meet those constraints with minimal reconstruction loss. This approach does not require any data, but may lead to a severe degradation in accuracy if the hyper-parameters aren't chosen appropriately or the compression factor is too high.

## Installation

1. Clone this repository.
2. Create a Python virtual environment and install dependancies:

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

To compress ResNet50 with an evaluation led approach with maximum 2% accuracy loss, and unstructured sparsity, run the following:
[ADD COMMAND]
The following pruning structures were implemented:

- `--unstructured` for unstructured sparsity
- `--filter` for filter sparsity (prune kernel along the last axis)
- `--channel` for channel sparsity (prune kernel along the third axis)
- `--block=(size, size)` for block sparsity, where size determines the size of the block
  For a fixed compression approach, the user needs to provide a function determining the importance of each layer and a function to determine the reconstruction error.
  [MORE USAGE]

Run `[SCRIPT NAME] --help` for more arguments.

##

## File structure

>

    .
    ├── .gitignore                                # Git ignore file
    ├── models                                    # Model files
    |   ├── keras                                 # Keras SaveModel models
    │   └── lite                                  # TFLite models
    ├── notebooks                                 # Jupyter notebooks
    │   └── Test.ipynb                            # Testing and EDA
    ├── README.md                                 # This file
    ├── requirements.txt                          # Package dependencies
    ├── scripts                                   # Scripts for various tasks
    │   ├── benchmarks.sh                         # Benchmarking script (loop over compress.py)
    │   ├── compress.py                           # Main compression script
    │   ├── data.py                               # Data visualisation
    │   └── train.py                              # Training script, generate base model
    └── src                                       # Source code
        ├── custom_resnet.py                      # Code to generate custom ResNet50 model with decompositions
        ├── dataset.py                            # Custom Dataset object for CIFAR-100
        ├── model.py                              # Generate base model compatible with CIFAR-100
        ├── pruning.py                            # Pruning loop
        └── utils.py                              # Various utilities and datastructes

>

## TODO

- finish readme
- make into library
- plot lots of stats
- add 1x1 support
- quantise?
- test everything x1000
