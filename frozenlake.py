import numpy as np
import gymnasium as gym

def initialize_q_table(state_space, action_space):
    return np.zeros((state_space, action_space))

def choose_action(state, q_table, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore action space
    else:
        return np.argmax(q_table[state, :])  # Exploit learned values

def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    best_next_action = np.argmax(q_table[next_state, :])
    td_target = reward + gamma * q_table[next_state, best_next_action]
    td_error = td_target - q_table[state, action]
    q_table[state, action] += alpha * td_error

# Initialize the FrozenLake environment
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False)
state_space = env.observation_space.n
action_space = env.action_space.n

# Initialize the Q-table
q_table = initialize_q_table(state_space, action_space)

# Hyperparameters
alpha = 0.9  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01  # Minimum exploration rate
decay_rate = 0.995  # Exponential decay rate for exploration prob
num_episodes = 10000
max_steps = 100  # Max steps per episode

# Training the agent
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_rewards = 0  # Track rewards per episode

    for _ in range(max_steps):
        action = choose_action(state, q_table, epsilon)
        next_state, reward, done, truncated, _ = env.step(action)
        update_q_table(q_table, state, action, reward, next_state, alpha, gamma)
        state = next_state

        total_rewards += reward  # Accumulate rewards

        if done:
            break

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * decay_rate)

    # Print the reward of every 1000th episode
    if (episode + 1) % 1000 == 0:
        print(f"Episode: {episode + 1}, Total Reward: {total_rewards}")

print("Training completed.")

# Display the final Q-table
print("\nFinal Q-Table Values")
print(q_table)

# Test the agent
state, _ = env.reset()
done = False
rewards = 0

print("\nTesting the trained agent:")
for step in range(max_steps):
    action = np.argmax(q_table[state, :])
    next_state, reward, done, truncated, _ = env.step(action)
    state = next_state
    rewards += reward
    env.render()
    print(f"Step: {step+1}, Action: {action}, State: {state}, Reward: {reward}, Total Rewards: {rewards}")
    if done:
        break

print(f"Total rewards during test episode: {rewards}")
env.close()
