from abc import ABC, abstractmethod


class BaseEnvironment(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def env_reset(self, env_setup: dict = {}):
        """Called at the beginning of the experiment

        :param env_setup: Dict - containing the information necessary to set up the environment for the experiment.
        """

    @abstractmethod
    def env_terminate(self):
        """Called at the end of the experiment, when it terminates.
        """

    @abstractmethod
    def env_step(self):
        """Step taken by the environment

        :return: (Int | Float, State, Boolean) - tuple of reward, new environment state and a boolean to indicate if
        the state is terminal
        """
