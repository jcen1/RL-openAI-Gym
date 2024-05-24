import numpy as np
import gymnasium as gym

# need to install minigrid
from collections import defaultdict
# Create the environment
env = gym.make("MiniGrid-Dynamic-Obstacles-16x16-v0")

env = ImgObsWrapper(env) 
# Q-learning parameters
alpha = 0.1   # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1
epsilon_decay = 0.995
num_episodes = 1000
# Initialize Q-table
q_table = defaultdict(lambda: np.zeros(env.action_space.n))
def flatten_state(state):
   # Extract and flatten the image component of the state
   return tuple(state['image'].flatten())
def choose_action(state):
   if np.random.rand() < epsilon:
       return env.action_space.sample()  # Explore
   else:
       return np.argmax(q_table[state])  # Exploit
def update_q_table(state, action, reward, next_state):
   best_next_action = np.argmax(q_table[next_state])
   td_target = reward + gamma * q_table[next_state][best_next_action]
   td_error = td_target - q_table[state][action]
   q_table[state][action] += alpha * td_error
# Training loop
for episode in range(num_episodes):
   obs = env.reset()
   state = flatten_state(obs)
   total_reward = 0
   done = False
   while not done:
       action = choose_action(state)
       obs, reward, done, _ = env.step(action)
       next_state = flatten_state(obs)
       total_reward += reward
       update_q_table(state, action, reward, next_state)
       state = next_state
   if epsilon > epsilon_min:
       epsilon *= epsilon_decay
   print(f"Episode {episode+1}: Total Reward: {total_reward}")
# Testing the learned policy
obs = env.reset()
state = flatten_state(obs)
done = False
total_reward = 0
while not done:
   action = np.argmax(q_table[state])
   obs, reward, done, _ = env.step(action)
   next_state = flatten_state(obs)
   total_reward += reward
   env.render()  # Display the environment
   state = next_state
print(f"Test Episode: Total Reward: {total_reward}")
env.close()
