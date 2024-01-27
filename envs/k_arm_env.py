import numpy as np

from RLGlue.environment import BaseEnvironment


class ModifiedKArmEnv(BaseEnvironment):
    def __init__(self):
        super().__init__()
        self.n_arms = None
        self.arms = []

    def env_init(self, env_info={}):
        self.n_arms = env_info.get('n_arms', 2)
        self.arms = [env_info.get('true_value', 0) for _ in range(self.n_arms)]

    def env_start(self):
        return None

    def env_step(self, action):
        reward = self.arms[action] + np.random.normal(0, 1)
        self.arms = [self.arms[_action] + np.random.normal(0, 0.01) for _action in range(self.n_arms)]

        return reward, None, False

    def env_cleanup(self):
        pass

    def env_message(self, message):
        pass
