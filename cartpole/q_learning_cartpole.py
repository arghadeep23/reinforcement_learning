import gymnasium as gym
import tensorflow as tf
from keras import Model, Input
from keras.layers import Dense

env = gym.make("CartPole-v1", render_mode="rgb_array")

# Q Network ( not table )
net_input = Input(shape=(4,))
x = Dense(64, activation="relu")(net_input)
x = Dense(32, activation="relu")(x)
output = Dense(2, activation="linear")(x)
q_net = Model(inputs=net_input, outputs=output)

# keras or tensorflow requires the input of this network to not be just a single
# tensor of shape 4 , but rather a batch of observations.
# so if we pass only 1 observation in the batch , then the shape will be [1,4]

# the output will be a batch of observations that we have given, like [1,2] , 2
# being the number of actions/q values

# Parameters
ALPHA = 0.001
EPSILON = 1.0
EPSILON_DECAY = 1.001
GAMMA = 0.99
NUM_EPISODES = 500


def policy(state, explore=0.0):
    action = tf.argmax(q_net(state)[0], output_type=tf.int32)  # extracting from the output , which has shape (1,2)
    if tf.random.uniform(shape=(), maxval=1) <= explore:
        action = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)
    return action


for episode in range(NUM_EPISODES):
    done = False
    total_reward = 0
    episode_length = 0
    state, _ = env.reset()
    state = tf.convert_to_tensor([state])

    while not done:
        action = policy(state, EPSILON)
        next_state, reward, done, trunc, _ = env.step(action.numpy())
        next_state = tf.convert_to_tensor([next_state])
        next_action = policy(next_state)
        target = reward + GAMMA * q_net(next_state)[0][next_action]

        if done:
            target = reward

        with tf.GradientTape() as tape:
            current = q_net(state)

        grads = tape.gradient(current,q_net.trainable_weights)
        delta = target - current[0][action]
        for j in range(len(grads)):
            # since we are dealing with tensors , normal plus operation won't work
            # hence , need to use assign_add
            q_net.trainable_weights[j].assign_add(ALPHA * delta * grads[j])

        state = next_state
        action = next_action
        total_reward += reward
        episode_length += 1
    print(f"The reward for episode {episode} is {total_reward}, the episode length is {episode_length}, the epsilon is {EPSILON}")
    EPSILON /= EPSILON_DECAY

# saving the q_network
q_net.save("q_learning_q_net")
env.close()

# initially the episode length was not increasing upto 500
# one problem can be that the fixed value of epsilon. To counter this problem,
# we can add some decay

