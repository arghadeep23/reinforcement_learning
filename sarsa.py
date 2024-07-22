import gym
import numpy as np
import pickle as pkl

cliffEnv = gym.make('CliffWalking-v0', render_mode='ansi')

# initializing the action value function Q(s,a)
q_table = np.zeros(shape=(48, 4))  # since there are 48 states and 4 actions (up, down, left, right for each state)


# defining the policy for the RL agent
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


# Setting the parameters
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
NUM_EPISODES = 500

# running the SARSA algorithm for a number of episodes
for episode in range(NUM_EPISODES):

    # done marks the end of the episode
    done = False
    total_reward = 0
    episode_length = 0
    # the following command initializes the environment and returns a new state
    state, _ = cliffEnv.reset()  # Unpack the returned tuple
    state = int(state)

    # Passing the state and epsilon to policy function to get a new action based on Epsilon greedy policy
    action = policy(state, EPSILON)

    while not done:
        next_state, reward, done, trunc, _ = cliffEnv.step(action)
        next_state = int(next_state)
        # in this code, we need to get the next action from the policy function first
        next_action = policy(next_state, EPSILON)

        # performing the update in the Q table
        q_table[state, action] += ALPHA * (reward + GAMMA * q_table[next_state, next_action] - q_table[state, action])

        state = next_state
        action = next_action
        total_reward += reward
        episode_length += 1

    print(f"Episode {episode}: Total Reward = {total_reward}, Length = {episode_length}")

cliffEnv.close()
pkl.dump(q_table, open("sarsa_q_table.pkl", "wb"))
print("Training complete. Q table saved")
