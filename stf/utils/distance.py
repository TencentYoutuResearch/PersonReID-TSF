"""Numpy version of euclidean distance, etc.
Notice the input/output shape of methods, so that you can better understand
the meaning of these methods."""
import numpy as np


def normalize(nparray, order=2, axis=0):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)


def compute_dist(array1, array2, type='euclidean'):
    """Compute the euclidean or cosine distance of all pairs.
    Args:
      array1: numpy array with shape [m1, n]
      array2: numpy array with shape [m2, n]
      type: one of ['cosine', 'euclidean']
    Returns:
      numpy array with shape [m1, m2]
    """
    assert type in ['cosine', 'euclidean']
    if type == 'cosine':
        array1 = normalize(array1, axis=1)
        array2 = normalize(array2, axis=1)
        dist = np.matmul(array1, array2.T)
        return dist
    else:
        # shape [m1, 1]
        square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
        # shape [1, m2]
        square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
        #squared_dist = - 2 * array1.dot(array2.T) + square1 + square2
        #squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2

        dist = np.zeros((array1.shape[0], array2.shape[0]))
        for i, row in enumerate(array1):
            dist[i, :] = np.matmul(row[None, :], array2.T)
        dist = -2 * dist + square1 + square2
        dist[dist < 0] = 0
        dist = np.sqrt(dist)
        return dist


def compute_dist_2(array1, array2, type='euclidean'):
    """Compute the euclidean or cosine distance of all pai
    Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    type: one of ['cosine', 'euclidean']
    Returns:
    numpy array with shape [m1, m2]"""
    assert type in ['cosine', 'euclidean']

    print 'computing distance', array1.shape, array2.shape
    if type == 'cosine':
        array1 = normalize(array1, axis=1)
        array2 = normalize(array2, axis=1)
        dist = np.matmul(array1, array2.T)
        return dist
    else:
        seperated_squared_dist = []
        squared_dist = []
        for n1 in range(len(array1)):
            squared_list = []
            for n2 in range(len(array2)):
                one = np.sum(np.square(array1[n1] - array2[n2]), axis=1)
                squared_list.append(one)
            seperated_squared_dist.append(squared_list)
        squared_dist = seperated_squared_dist
        dist = np.sum(seperated_squared_dist, axis=2)
        return dist, squared_dist
