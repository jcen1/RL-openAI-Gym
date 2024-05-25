import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt

def main():
    # Create Taxi environment
    env = gym.make('Taxi-v3', render_mode="rgb_array")

    # Initialize Q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))    

    # Hyperparameters
    learning_rate = 0.9
    discount_rate = 0.8
    epsilon = 1.0
    decay_rate = 0.005

    # Training variables
    num_episodes = 1000
    max_steps = 99  # per episode
    rewards = []

    # Training
    for episode in range(num_episodes):
        # Reset the environment
        state = env.reset()[0]  # Ensure state is an integer
        total_rewards = 0
        done = False

        for s in range(max_steps):
            # Exploration-exploitation tradeoff
            if random.uniform(0, 1) < epsilon:
                # Explore
                action = env.action_space.sample()
            else:
                # Exploit
                action = np.argmax(qtable[state, :])

            # Take action and observe reward
            new_state, reward, done, _, info = env.step(action)
            if isinstance(new_state, tuple):
                new_state = new_state[0]  # Ensure new_state is an integer

            # Q-learning algorithm
            qtable[state, action] = qtable[state, action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])

            # Update to our new state
            state = new_state
            total_rewards += reward

            # If done, finish episode
            if done:
                break

        # Track rewards
        rewards.append(total_rewards)

        # Decrease epsilon
        epsilon = np.exp(-decay_rate * episode)

    print(f"Training completed over {num_episodes} episodes")

    # Plotting the rewards
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning: Total Reward vs Episode')
    plt.show()

    input("Press Enter to watch trained agent...")

    # Watch trained agent
    state = env.reset()[0]  # Ensure state is an integer
    done = False
    rewards = 0

    for s in range(max_steps):
        print(f"TRAINED AGENT")
        print("Step {}".format(s + 1))

        action = np.argmax(qtable[state, :])
        new_state, reward, done, _, info = env.step(action)
        if isinstance(new_state, tuple):
            new_state = new_state[0]  # Ensure new_state is an integer
        rewards += reward
        env.render()
        print(f"score: {rewards}")
        state = new_state

        if done:
            break

    env.close()

if __name__ == "__main__":
    main()
