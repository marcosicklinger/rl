import numpy as np

from utils.funcs.math import rnd_argmax


def greedy(q: np.ndarray[int | float]) -> int:
    """
    Takes a Q-value function for a specific observation and returns the greedy action.

    :param q: Array[int | float] - Q-value function for an observation.

    :return: int - index of the greedy action
    """
    return rnd_argmax(q)


def epsilon_greedy(q: np.ndarray[int | float], epsilon: float):
    """
    Takes a Q-value function for a specific observation and the parameter epsilon and returns an action sampled in
    the epsilon-greedy way.

    :param q: Array[int | float] - Q-value function for an observation.
    :param epsilon: float - epsilon parameter.

    :return: int - index of the greedy action
    """
    return rnd_argmax(q) if np.random.random() > epsilon else np.random.choice(np.arange(q.shape[0]))
