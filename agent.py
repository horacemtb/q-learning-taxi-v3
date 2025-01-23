import numpy as np


class QLearningAgent():
    def __init__ (self, env, alpha=0.5, gamma=0.99, epsilon=1, noisy_episode_n=400):
        self.env = env
        self.state_n = env.observation_space.n
        self.action_n = env.action_space.n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.noisy_episode_n = noisy_episode_n
        self.Q = None
        self.init_Q()

    def init_Q(self):
        self.Q = np.zeros((self.state_n, self.action_n))

    def choose_action(self, state):
        policy = np.ones(self.action_n) * self.epsilon / self.action_n
        max_action = np.argmax(self.Q[state])
        policy[max_action] += 1 - self.epsilon

        return np.random.choice(np.arange(self.action_n), p=policy)

    def update_epsilon(self):
        self.epsilon = max(0, self.epsilon - 1 / self.noisy_episode_n)

    def learn(self, state, action, reward, next_state):
        self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])