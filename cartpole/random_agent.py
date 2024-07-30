# Regarding the cart pole environment :
# It is a classic problem in control theory and reinforcement learning
# Dynamics : The system is controlled by applying a force +1 or -1 to the cart
# The state consists of 4 variables : Cart position, cart velocity, pole angle, pole angular velocity
# Goal : To keep the pole balanced upright on the cart for as long as possible
# Action Space: The agent can push the cart left (0) or right (1).
# Reward: +1 for every timestep the pole remains upright.
import gymnasium as gym
import cv2
import tensorflow as tf
import numpy as np

env = gym.make("CartPole-v1", render_mode="rgb_array")

for episode in range(5):
    terminated = truncated = False
    state, _ = env.reset()
    while not (terminated or truncated):
        frame = env.render()
        if isinstance(frame, list):
            frame = np.array(frame[0])
        cv2.imshow("CartPole", frame)
        cv2.waitKey(100)
        # we will select our action randomly between 0 and 1
        # numpy converts a tensor of multiple dimensions into an array and a scalar into an integer
        action = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32).numpy()
        state, reward, terminated, truncated, _ = env.step(action)
    cv2.destroyAllWindows()
env.close()
# In this code, we are not actually solving the problem optimally
# Rather, it's about taking random actions, and showing the different states, and running for a fixed number of episodes

