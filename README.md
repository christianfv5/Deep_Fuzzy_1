# Deep Convolutional Neuro-Fuzzy Inference System

This repository contains the source code for the *Deep Convolutional
Neuro-Fuzzy Inference System* paper I co-authored with Mojtaba
Yeganejou and Dr. Scott Dick.

The paper investigates a new type of neural network that utilizes
neuro-fuzzy systems to create more interpretable models. Conventional
neural network layers are used for feature extraction, and a
neuro-fuzzy classifier to output the class probabilities.

## Getting Started

### Prerequisites

* Jupyter Notebook
* Keras
* TensorFlow
* NumPy
* SciKit-Learn
* SciKit-Fuzzy
* UMAP-Learn
* H5Py
* iNNvestigate

The majority of the code is written in Jupyter Notebooks to make
sharing results and the corresponding code easier.

## Relevant Code

The code has been uploaded in the state it was written and is
unorganized. This section points out the parts of the
code that are most relevant the paper.

### Membership Function Implementation

[model.py](mnist/model.py)

The code the for the membership function layer. It is implemented in 
Keras as a custom layer. Unfortunately, the `get_config` method is not
implemented and an exceptions is thrown when trying to load a custom 
model using this layer. The following code snippet circumvents the 
problem.

```
model = load_model(
  "path/to/model.h5",
  custom_objects={"LogGaussMF": lambda **x: LogGaussMF(rules=10, **x)})
```

### ResNet Implementation

[resnet_backend.py](mnist/resnet_backend.py)

Method to create a ResNet backend (everything before the final layer).
The code is take from the Keras ResNet implentation for CIFAR-10 found 
[here](https://keras.io/examples/cifar10_resnet/) and modified.

[cifar10_resnet.py](mnist/cifar10_resnet.py)

The Keras Resnet implementation modified so it can be imported as a module.
Original can be found [here](https://keras.io/examples/cifar10_resnet/).

### LeNet Experiments

[mnist-lenet.ipynb](mnist/mnist-lenet.ipynb)

[fmnist-lenet.ipynb](fashion-mnist/fmnist-lenet.ipynb)

[cifar10-lenet.ipynb](cifar-10/cifar10-lenet.ipynb)

The experiments run with the LeNet based model. The conventional neural 
network and neuro-fuzzy variation are implemented and trained in these
notebooks.

### ResNet Experiments

[mnist-resnet.ipynb](mnist/mnist-resnet.ipynb)

[fmnist-resnet.ipynb](fashion-mnist/fmnist-resnet.ipynb)

[cifar-10-resnet.ipynb](cifar-10/cifar-10-resnet.ipynb)

The experiments run with the ResNet based model. The conventional neural 
network and neuro-fuzzy variation are implemented and trained in these
notebooks.

## Authors

* **Ryan Kluzinski** - *Initial work* - [rkluzinski](https://github.com/rkluzinski)
* **Mojtaba Yeganejou** - *GK Clustering Code*

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Dr. Scott Dick - Co-authored the paper and supervised my work term as a research assistant.
* Mojtaba Yeganejou - Co-authored the paper and mentored me during my work term.

