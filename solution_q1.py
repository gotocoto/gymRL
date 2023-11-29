import gymnasium as gym
import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
def run_BJack_Q():
    # Start Q-learning code
    env = gym.make("Blackjack-v1", natural=False, sab=False)  # Initializing environments
    stateNEW, info = env.reset()
    terminated = False  # Will be true if we win Blackjack
    truncated = False  # Will be true when the "Actions" threshold is met
    q_table = np.zeros([32, 11, 2,2])
    alpha = 0.8 #will be used when adding values to the q-table
    eta = .4
    epilson = 10
    stateOLD = stateNEW
    for _ in range(1000000):
        #implement q-learning
        action = env.action_space.sample() #get a sample Blackjack action
        stateNEW, reward, terminated, truncated, info = env.step(action)
        q_table[stateOLD[0],stateOLD[1],stateOLD[2], action] = (1-alpha)*(q_table[stateOLD[0],stateOLD[1],stateOLD[2], action] ) + alpha*(reward+eta*np.max(q_table[stateNEW[0],stateNEW[1],stateNEW[2]] )) #adding new estimate to the q-table, creating a running average
        if terminated or truncated:
            observation, info = env.reset()
        epilson=max(0,epilson-0.00001) #making sure actions are taken randomly
        stateOLD = stateNEW
    env.close()
    print(q_table)
    # End Q-learning code


if __name__ == "__main__":
    run_BJack_Q()
