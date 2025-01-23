import numpy as np
import pickle
from agent import QLearningAgent


def running_mean(x, N=100):
    x = np.array(x)
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y


def save_agent(agent, filename='agent.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump({
            'Q': agent.Q,
            'epsilon': agent.epsilon,
            'alpha': agent.alpha,
            'gamma': agent.gamma,
            'noisy_episode_n': agent.noisy_episode_n
        }, f)
    print(f'Agent saved to {filename}')


def load_agent(filename, env):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    agent = QLearningAgent(env, alpha=data['alpha'], gamma=data['gamma'], epsilon=data['epsilon'], noisy_episode_n=data['noisy_episode_n'])
    agent.Q = data['Q']
    print(f'Agent loaded from {filename}')
    return agent