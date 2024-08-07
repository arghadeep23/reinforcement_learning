import gymnasium as gym
import cv2
import tensorflow as tf
import numpy as np
from keras.models import load_model


def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * tf.square(quadratic) + delta * linear


env = gym.make("CartPole-v1", render_mode="rgb_array")
transformer_model = load_model("new_cartpole_transformer_model", custom_objects={"huber_loss": huber_loss})



def policy(state, explore=0.0):
    if np.random.random() < explore:
        return env.action_space.sample()
    state = np.expand_dims(state, axis=(0, 1))  # Reshape to (1, 1, 4)
    q_values = transformer_model(state, training=False)
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
        cv2.waitKey(50)  # Adjust delay for better visualization

        action = policy(state.squeeze())  # Remove extra dimensions for policy
        next_state, reward, done, trunc, _ = env.step(action)
        state = next_state
        total_reward += reward
        steps += 1

    print(f"Episode {episode + 1}: Reward = {total_reward}, Steps = {steps}")

cv2.destroyAllWindows()
env.close()
