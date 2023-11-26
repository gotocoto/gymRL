import gymnasium as gym
import numpy as np

'''
observation, info = env.reset()
for _ in range(50):
action = env.action_space.sample() # agent policy that uses the observation and info
observation, reward, terminated, truncated, info = env.step(action)
if terminated or truncated:
observation, info = env.reset()
env.close()
'''


def run_BJack_Q():
    # Start Q-learning code
    env = gym.make("Blackjack-v1", natural=False, sab=False)  # Initializing environments
    observation, info = env.reset()
    terminated = False  # Will be true if we win Blackjack
    truncated = False  # Will be true when the "Actions" threshold is met
    q_table = np.random.uniform(low=0, high=1)

    while (not terminated and not truncated):
        for _ in range(200):
            action = env.action_space.sample()  # Blackjack actions: Take a card, or no.
            observation, reward, terminated, truncated, info = env.step(action)
            state = observation
            print(state)
            env.render()
            if terminated or truncated:
                observation, info = env.reset()

    env.close()

    # End Q-learning code


if __name__ == "__main__":
    run_BJack_Q()
