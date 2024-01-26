import numpy as np


def rnd_argmax(x: np.ndarray[int | float]) -> int:
    """Takes a list/array of numbers and returns the largest value.
    In case of multiple indexes corresponding to the largest values, returns one of them at random.

    :param x: Array[int | float] - containers of numbers

    :returns: int - the index corresponding to the largest value.
    """
    largest = -np.inf
    largest_index_list = []

    for n in range(x.shape[0]):
        if x[n] > largest:
            largest = x[n]
            largest_index_list += [n]
        elif x[n] == largest:
            largest_index_list.append(n)

    return np.random.choice(largest_index_list)