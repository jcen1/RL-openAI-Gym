import numpy as np
import gymnasium as gym
import random

# Create Taxi environment
env = gym.make('Taxi-v3')

# Initialize Q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Set hyperparameters
learning_rate = 0.9
discount_factor = 0.8
exploration_rate = 1.0
exploration_decay_rate = 0.05
num_episodes = 1000
num_steps = 99

# Training the agent
for episode in range(num_episodes):
    state = env.reset()
    for step in range(num_steps):
        if np.random.uniform(0, 1) < exploration_rate:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        next_state, reward, done,_, _ = env.step(action)
        q_table[state, action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state]) - q_table[state, action])
        state = next_state
        if done:
            break
    exploration_rate -= exploration_decay_rate