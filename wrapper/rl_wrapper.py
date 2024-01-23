class RLWrapper:

    def __init__(self, agent, environment):
        self.agent = agent()
        self.environment = environment()

        self.last_action = None
        self.n_steps = None
        self.n_episodes = None

    def rlw_reset(self, env_setup: dict = {}, agent_setup: dict = {}):
        self.environment.env_reset(env_setup)
        self.agent.agent_reset(agent_setup)

        self.n_steps = 0
        self.n_episodes = 0

    def rlw_terminate(self):
        self.agent.agent_terminate()
        self.environment.env_terminate()

    def rlw_step(self):
        (reward, last_state, is_terminal) = self.environment.env_step()

        if not is_terminal:
            self.n_steps += 1
            self.last_action = self.agent.agent_step(reward)
        else:
            self.n_episodes += 1
            self.last_action = None

        return reward, last_state, self.last_action, is_terminal

    def rlw_episode(self):
        pass

    def rlw_get_n_episodes(self):
        return self.n_episodes

    def rlw_get_n_steps(self):
        return self.n_steps
