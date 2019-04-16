import gzip
import numpy as np
import matplotlib.pyplot as plt


def load_idx(filename, dims):
    with gzip.open(filename, "rb") as infile:
        # consume magic number
        infile.read(4)
        # consume dimensions data
        infile.read(4 * len(dims))

        return np.frombuffer(infile.read(np.prod(dims)), dtype=np.uint8).reshape(dims)
    
    
def original_data():
    # training data
    train_images = load_idx("data/train-images-idx3-ubyte.gz", [60000, 28, 28])
    train_labels = load_idx("data/train-labels-idx1-ubyte.gz", [60000])

    # testing data
    test_images = load_idx("data/t10k-images-idx3-ubyte.gz", [10000, 28, 28])
    test_labels = load_idx("data/t10k-labels-idx1-ubyte.gz", [10000])
    
    return (train_images, train_labels), (test_images, test_labels)
    
    
def processed_data():
    (train_images, train_labels), (test_images, test_labels) = original_data()

    train_x = train_images.astype(np.float) / 255
    train_y = np.zeros((60000, 10))
    train_y[np.arange(60000),train_labels] = 1

    test_x = test_images.astype(np.float) / 255
    test_y = np.zeros((10000, 10))
    test_y[np.arange(10000),test_labels] = 1
    
    return (train_x, train_y), (test_x, test_y) 