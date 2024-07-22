import gym
import numpy as np
import pickle as pkl

cliffEnv = gym.make("CliffWalking-v0",render_mode="ansi")
q_table = np.zeros(shape=(48, 4))

# Parameters
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
NUM_EPISODES = 500


def policy(state, explore=0.0):
    # explore refers to epsilon
    # the epsilon greedy algorithm states the following :
    #   1) take a random action with epsilon probability
    #   2) take the optimal action (argmax( Q(s,a) ) with 1 - epsilon probability

    if np.random.random() <= explore:
        action = np.random.randint(0, 4)  # select from all 4 actions and return a scalar
    else:
        action = (int(np.argmax(q_table[state])))
    return action


for episode in range(NUM_EPISODES):
    done = False
    total_reward = 0
    episode_length = 0
    state, _ = cliffEnv.reset()
    while not done:
        action = policy(state, EPSILON)
        next_state, reward, done, trunc, _ = cliffEnv.step(action)
        next_action = policy(next_state)  # finding the optimal action for next_state
        #                                   (since epsilon is not passed, optimal will be returned)
        q_table[state, action] += ALPHA*(reward+GAMMA*q_table[next_state, next_action]-q_table[state, action])
        state = next_state
        total_reward += reward
        episode_length += 1
    print(f"Episode {episode}: Total Reward = {total_reward}, Length = {episode_length}")
cliffEnv.close()
pkl.dump(q_table, open("q_learning_q_table.pkl", "wb"))
print("Training complete. Q-table saved! ")
