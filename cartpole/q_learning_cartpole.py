import gymnasium as gym
import tensorflow as tf
from keras import Model, Input
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from collections import deque
import random

env = gym.make("CartPole-v1", render_mode="rgb_array")


# Q Network
def create_q_model(state_shape, action_shape):
    inputs = Input(shape=state_shape)
    x = Dense(64, activation="relu")(inputs)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(action_shape, activation="linear")(x)
    return Model(inputs=inputs, outputs=outputs)


state_shape = env.observation_space.shape
action_shape = env.action_space.n

q_net = create_q_model(state_shape, action_shape)
target_q_net = create_q_model(state_shape, action_shape)



# Parameters
ALPHA = 0.001
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
GAMMA = 0.99
NUM_EPISODES = 500
BATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5
MEMORY_SIZE = 10000

# Compile the models
q_net.compile(optimizer=Adam(learning_rate=ALPHA), loss='mse')
target_q_net.compile(optimizer=Adam(learning_rate=ALPHA), loss='mse')

optimizer = Adam(learning_rate=ALPHA)
memory = deque(maxlen=MEMORY_SIZE)


def policy(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    q_values = q_net.predict(state[np.newaxis], verbose=0)
    return np.argmax(q_values[0])


def train(batch_size):
    if len(memory) < batch_size:
        return

    minibatch = random.sample(memory, batch_size)
    states = np.array([transition[0] for transition in minibatch])
    actions = np.array([transition[1] for transition in minibatch])
    rewards = np.array([transition[2] for transition in minibatch])
    next_states = np.array([transition[3] for transition in minibatch])
    dones = np.array([transition[4] for transition in minibatch])

    target_q_values = target_q_net.predict(next_states, verbose=0)
    max_target_q_values = np.max(target_q_values, axis=1)
    targets = rewards + (1 - dones) * GAMMA * max_target_q_values

    q_values = q_net.predict(states, verbose=0)
    q_values[np.arange(batch_size), actions] = targets

    q_net.fit(states, q_values, verbose=0)


for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0

    while not done:
        action = policy(state, EPSILON)
        next_state, reward, done, truncated, _ = env.step(action)
        episode_reward += reward
        episode_length += 1

        memory.append((state, action, reward, next_state, done))
        state = next_state

        train(BATCH_SIZE)

    if episode % UPDATE_TARGET_EVERY == 0:
        target_q_net.set_weights(q_net.get_weights())

    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

    print(f"Episode: {episode}, Reward: {episode_reward}, Length: {episode_length}, Epsilon: {EPSILON:.4f}")

# Save the model
q_net.save("improved_q_learning_q_net")
env.close()
