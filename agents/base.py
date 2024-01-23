from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def agent_reset(self, agent_setup: dict = {}) -> int:
        """Called at the beginning of the experiment, after
        the environment is set up.

        :param agent_setup: Dict - containing the information necessary to set up the agent for the experiment.

        :return: Int - the first action the agent takes.
        """

    @abstractmethod
    def agent_terminate(self) -> int | float:
        """Runs when the agent terminates.

        :return: Int | Float - the reward the agent received for entering the terminal state.
        """

    @abstractmethod
    def agent_step(self, reward: int | float, observation) -> int:
        """ Implements the learner step.

        :param reward: Int | Float - the reward the agent receives for taking the last action
        :param observation: agent's observation of the environment state resulting from the last step taken

        :return: Int - the current action selected by the agent
        """
