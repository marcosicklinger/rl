import numpy as np

from RLGlue.agent import BaseAgent
from utils.policy.greedy import epsilon_greedy


class EpsilonGreedyAgent(BaseAgent):

    def __init__(self):
        super().__init__()
        self.state_size = None
        self.n_actions = None
        self.q_init = 0.
        self.Q = None
        self.policy = epsilon_greedy
        self.alpha = None
        self.epsilon = None
        self.alpha_decay = None
        self.epsilon_decay = None
        self.arm_count = None
        self.last_action = None

    def agent_init(self, agent_info={}):
        self.state_size = agent_info.get("state_size", np.array([1]))
        self.n_actions = agent_info.get("n_actions", 2)
        self.q_init = agent_info.get("q_init", 0)
        self.Q = np.ones((*self.state_size, self.n_actions))*self.q_init
        self.alpha = agent_info.get("alpha", 0.1)
        self.epsilon = agent_info.get("epsilon", 0.1)
        self.arm_count = np.zeros((*self.state_size, self.n_actions))
        self.last_action = 0
        self.alpha_decay = agent_info.get("alpha_decay", lambda _: 1)
        self.epsilon_decay = agent_info.get("epsilon_decay", lambda _: 1)

    def agent_start(self, observation):
        self.last_action = self.policy(observation, self.epsilon)
        return self.last_action

    def agent_step(self, reward, observation):
        self.arm_count[*observation, self.last_action] += 1
        self.Q[*observation, self.last_action] += self.alpha*self.alpha_decay(self.arm_count)*(
                reward - self.Q[*observation, self.last_action]
        )

        self.last_action = current_action = self.policy(observation, self.epsilon)

        return current_action

    def agent_end(self, reward):
        pass

    def agent_cleanup(self):
        pass

    def agent_message(self, message):
        pass

