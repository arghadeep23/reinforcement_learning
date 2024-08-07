import gymnasium as gym
import cv2
import tensorflow as tf
import numpy as np
from keras.models import load_model

env = gym.make("CartPole-v1", render_mode="rgb_array")
q_net = load_model("improved_q_learning_q_net")


def policy(state, explore=0.0):
    if np.random.random() < explore:
        return env.action_space.sample()
    q_values = q_net.predict(state[np.newaxis], verbose=0)
    return np.argmax(q_values[0])


NUM_EPISODES = 10
for episode in range(NUM_EPISODES):
    done = False
    state, _ = env.reset()
    total_reward = 0
    steps = 0

    while not done:
        frame = env.render()
        cv2.imshow("CartPole", frame)
        cv2.waitKey(50)  # Increased delay for better visualization

        action = policy(state)
        next_state, reward, done, trunc, _ = env.step(action)

        state = next_state
        total_reward += reward
        steps += 1

    print(f"Episode {episode + 1}: Reward = {total_reward}, Steps = {steps}")

cv2.destroyAllWindows()
env.close()