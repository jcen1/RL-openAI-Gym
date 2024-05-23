import numpy as np
import gymnasium as gym
import numpy as np
import gym
import random

def Q_learning_train(env, alpha, gamma, epsilon, episodes):
    all_epochs = []
    all_penalties = []

    # Initialize Q table of size (number of states x number of actions) with all zeroes
    q_table = np.zeros([env.observation_space.n, env.action_space.n])  

    for i in range(1, episodes + 1):
        state_tuple = env.reset()
        state = state_tuple if isinstance(state_tuple, int) else state_tuple[0]

        epochs, penalties, reward = 0, 0, 0
        done = False

        while not done:
            if not isinstance(state, int):
                print(f"State is not an integer: {state}")
            if state < 0 or state >= env.observation_space.n:
                print(f"State is out of bounds: {state}")

            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space randomly
            else:
                action = np.argmax(q_table[state])  # Exploit learned values by choosing optimal values

            next_state_tuple, reward, done, _,info = env.step(action) 
            next_state = next_state_tuple if isinstance(next_state_tuple, int) else next_state_tuple[0]

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1

        if i % 100 == 0:
            print(f"Episode: {i}")

    # Start with a random policy
    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n

    for state in range(env.observation_space.n):  # for each state
        best_act = np.argmax(q_table[state])  # find the best action
        policy[state] = np.eye(env.action_space.n)[best_act]  # update

    print("Training finished.\n")
    return policy, q_table

env = gym.make('Taxi-v3')
env.reset()
Q_learn_pol, q_table = Q_learning_train(env, 0.2, 0.95, 0.1, 100000)
