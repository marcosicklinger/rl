import numpy as np

from RLGlue.agent import BaseAgent
from utils.policy.greedy import epsilon_greedy


class EpsilonGreedyBandit(BaseAgent):

    def __init__(self):
        super().__init__()
        self.n_actions = None
        self.q_init = 0.
        self.size = 1
        self.Q = None
        self.policy = None
        self.epsilon = None
        self.arm_count = None
        self.last_action = None

    def agent_init(self, agent_info={}):
        self.n_actions = agent_info.get('n_actions')
        self.size = agent_info.get('size')
        self.q_init = agent_info.get('q_init')
        self.Q = np.ones(self.n_actions) * self.q_init
        self.policy = epsilon_greedy
        self.epsilon = agent_info.get('epsilon')
        self.arm_count = [0 for _ in range(self.n_actions)]
        self.last_action = 0

    def agent_start(self, observation):
        self.last_action = np.random.choice(self.n_actions)
        return self.last_action

    def agent_step(self, reward, observation=None):
        self.arm_count[self.last_action] += 1
        self.Q[self.last_action] += self.step_size() * [reward - self.Q[self.last_action]]

        self.last_action = current_action = self.policy(self.Q, self.epsilon)

        return current_action

    def step_size(self):
        return 1

    def agent_end(self, reward):
        pass

    def agent_cleanup(self):
        pass

    def agent_message(self, message):
        pass
