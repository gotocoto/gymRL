import gymnasium as gym
import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
def run_BJack_Q():
    # Start Q-learning code
    env = gym.make("Blackjack-v1", natural=False, sab=False)  # Initializing environments
    
    #env = gym.make("Blackjack-v1", natural=False, sab=False, render_mode="human")  # Initializing environments
    stateNEW, info = env.reset()
    terminated = False  # Will be true if we win Blackjack
    truncated = False  # Will be true when the "Actions" threshold is met
    #q_table = np.zeros([env.observation_space[0], env.action_space.n])
    q_table = np.zeros([32, 11, 2,2])
    alpha = 0.8
    eta = .4
    epilson = 10
    stateOLD = stateNEW
    for _ in range(1000000):
        #implement q-learning
        action = env.action_space.sample()
        '''
        if np.random.random() < epilson:
             # Blackjack actions: Take a card, or no. 0 or 1
        else:
            action = int(np.argmax(q_table[observation]))'''
        stateNEW, reward, terminated, truncated, info = env.step(action)
        #print(stateOLD)
        #print(q_table[stateOLD[0],stateOLD[1],stateOLD[2], action])
        #q_table[state, action] += (reward + np.max([q_table[state]]) - q_table[state, action])
        q_table[stateOLD[0],stateOLD[1],stateOLD[2], action]    = (1-alpha)*(q_table[stateOLD[0],stateOLD[1],stateOLD[2], action] ) + alpha*(reward+eta*np.max(q_table[stateNEW[0],stateNEW[1],stateNEW[2]] ))
        #q_table[stateOLD[0],stateOLD[1],stateOLD[2], action] +=1
        if terminated or truncated:
            observation, info = env.reset()
        epilson=max(0,epilson-0.00001)
        stateOLD = stateNEW
    env.close()
    print(q_table)
    # End Q-learning code


if __name__ == "__main__":
    run_BJack_Q()
