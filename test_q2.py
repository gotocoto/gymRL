import gymnasium as gym
import numpy as np
np.set_printoptions(threshold=np.inf)
env = gym.make("FrozenLake-v1", desc=None, map_name="4x4",is_slippery=True, ) #initialization
#env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", render_mode="human",is_slippery=True, ) #initialization
# Reference used https://www.oreilly.com/radar/introduction-to-reinforcement-learning-and-openai-gym/
# Part 2
T = np.zeros([env.observation_space.n,env.action_space.n,env.observation_space.n])
R = np.zeros([env.observation_space.n,env.action_space.n,env.observation_space.n])
observation1, info = env.reset()
observation2 = observation1
for _ in range(1000):
    action = env.action_space.sample() # agent policy that uses the observation and info
    observation2, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation2, info = env.reset()
    if(reward!=0):
        print(reward,observation1,action,observation2)
    
    T[observation1,action,observation2]+=1
    R[observation1,action,observation2]+=reward
    observation1 = observation2
#Part 3
Tl = np.zeros([env.observation_space.n,env.action_space.n,env.observation_space.n])
Rl = np.zeros([env.observation_space.n,env.action_space.n,env.observation_space.n])
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        sum = 0
        for sl in range(env.observation_space.n):
            sum+= T[s,a,sl]
        for sl in range(env.observation_space.n):
            if(sum!=0):
                Tl[s,a,sl]= T[s,a,sl]/sum
            else:
                Tl[s,a,sl]= 1/env.observation_space.n
            if(T[s,a,sl]!=0):
                Rl[s,a,sl]= R[s,a,sl]/T[s,a,sl]
print("The transition function")
print(Tl)
print("The reward function")
print(Rl)
eta = 0.3
#Part 4
V = np.zeros([env.observation_space.n])
Vnew = np.zeros([env.observation_space.n])
i =0
while not np.all(np.isclose(V,Vnew)) or np.all(V==0) or i>1000:
    i+=1
    for state in range(env.observation_space.n):
        actions = np.zeros([env.action_space.n])
        for action in range(env.action_space.n):
            for endstate in range(env.observation_space.n):
                actions[action]+= Tl[state,action,endstate]*(Rl[state,action,endstate]+eta*V[endstate])
        Vnew[state] = max(actions)
    V = Vnew
    #print(V[0])
print(V)
env.close()
