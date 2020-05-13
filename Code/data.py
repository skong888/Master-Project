import numpy as np
import pickle
import os
import sys

def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_data():
    """Loads CIFAR10 dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = 'cifar-10-batches-py'

    num_train_samples = 50000

    x_train_local = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train_local = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train_local[(i - 1) * 10000: i * 10000, :, :, :],
         y_train_local[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test_local, y_test_local = load_batch(fpath)
    y_train_local = np.reshape(y_train_local, (len(y_train_local)))
    y_test_local = np.reshape(y_test_local, (len(y_test_local)))

    train_labels = np.zeros([y_train_local.shape[0], 10])
    n = 0
    for element in y_train_local:
        train_labels[n, int(element)] = 1
        n += 1

    test_labels = np.zeros([y_test_local.shape[0], 10])
    n = 0
    for element in y_test_local:
        test_labels[n, element] = 1
        n += 1

    x_train_local = x_train_local.transpose(0, 2, 3, 1)
    x_test_local = x_test_local.transpose(0, 2, 3, 1)

    return (x_train_local, train_labels), (x_test_local, test_labels)



