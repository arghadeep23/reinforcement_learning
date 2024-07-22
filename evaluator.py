# for the evaluation of the model created in q_learning.py
import pickle as pkl
import cv2
import gym
import numpy as np

cliffEnv = gym.make('CliffWalking-v0', render_mode='ansi')

# making use of the q_table created in sarsa.py
q_table = pkl.load(open("q_learning_q_table.pkl", "rb"))


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


def initialize_frame():
    width, height = 600, 200
    img = np.ones(shape=(height, width, 3)) * 255.0
    margin_horizontal = 6
    margin_vertical = 2

    # Vertical Lines
    for i in range(13):
        img = cv2.line(img, (49 * i + margin_horizontal, margin_vertical),
                       (49 * i + margin_horizontal, 200 - margin_vertical), color=(0, 0, 0), thickness=1)

    # Horizontal Lines
    for i in range(5):
        img = cv2.line(img, (margin_horizontal, 49 * i + margin_vertical),
                       (600 - margin_horizontal, 49 * i + margin_vertical), color=(0, 0, 0), thickness=1)

    # Cliff Box
    img = cv2.rectangle(img, (49 * 1 + margin_horizontal + 2, 49 * 3 + margin_vertical + 2),
                        (49 * 11 + margin_horizontal - 2, 49 * 4 + margin_vertical - 2), color=(255, 0, 255),
                        thickness=-1)
    img = cv2.putText(img, text="Cliff", org=(49 * 5 + margin_horizontal, 49 * 4 + margin_vertical - 10),
                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    # Goal
    frame = cv2.putText(img, text="G", org=(49 * 11 + margin_horizontal + 10, 49 * 4 + margin_vertical - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    # Start
    # frame = cv2.putText(img, text="S", org=(49 * 0 + margin_horizontal + 10, 49 * 4 + margin_vertical - 10),
    #                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return frame


# puts the agent at a state
def put_agent(img, state):
    margin_horizontal = 6
    margin_vertical = 2
    if isinstance(state, (tuple, list, np.ndarray)):
        state = state[0]  # Extract the first element if state is not a scalar
    row, column = np.unravel_index(indices=state, shape=(4, 12))
    cv2.putText(img, text="A", org=(49 * column + margin_horizontal + 10, 49 * (row + 1) + margin_vertical - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return img


NUM_EPISODES = 5
for episode in range(NUM_EPISODES):
    frame = initialize_frame()  # initializing a frame
    done = False
    total_reward = 0
    episode_length = 0
    state, _ = cliffEnv.reset()
    while not done:
        frame2 = put_agent(frame.copy(),state)
        cv2.imshow("Cliff Walking (Q Learning)", frame2)
        cv2.waitKey(250)  # wait for 250 ms
        # not passing any epsilon to make it the optimal policy
        action = policy(state)
        state, reward, done, trunc, _ = cliffEnv.step(action)
        total_reward += reward
        episode_length += 1

    print(f"The reward for episode {episode} is {total_reward}, the episode length is {episode_length}")

# The output showed a reward of -17 and a solution path which was
# suboptimal (as mentioned in Sutton and Barto )
