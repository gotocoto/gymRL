import gymnasium as gym
import numpy as np

def run_BJack_Q():
    # Start Q-learning code
    env = gym.make("Blackjack-v1", natural=False, sab=False)  # Initializing environments
    observation1, info = env.reset()
    terminated = False  # Will be true if we win Blackjack
    truncated = False  # Will be true when the "Actions" threshold is met

    #q_table = np.zeros((env.observation_space[0].n,  env.action_space[0].n))
    q_table = np.zeros((32, 32, 2))
    alpha = 0.1



    while not terminated and not truncated:
        for _ in range(1000):
            #implement q-learning
            action = env.action_space.sample() # Blackjack actions: Take a card, or no.
            #action = np.argmax(q_table[state])
            state, reward, terminated, truncated, info = env.step(action)
            #q_table[state, action] += (reward + np.max([q_table[state]]) - q_table[state, action])
            q_table[state, action] += (1-alpha)*(q_table[state,action]) + alpha*reward
            if terminated or truncated:
                observation2, info = env.reset()
    env.close()
    print(q_table)
    # End Q-learning code


if __name__ == "__main__":
    run_BJack_Q()
