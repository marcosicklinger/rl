from RLGlue.environment import BaseEnvironment


class MultiAgentActiveSearchEnv(BaseEnvironment):

    def __init__(self):
        super().__init__()
        self.n_agents = None
        self.n_targets = None
        self.boundary = None

        # define state space

    def env_init(self, env_info={}):
        pass

    def env_start(self):
        pass

    def env_step(self, action):
        pass

    def env_cleanup(self):
        pass

    def env_message(self, message):
        pass
