import gymnasium as gym
import tensorflow as tf
import random
from keras import Model, Input
from keras.layers import Dense, MultiHeadAttention, LayerNormalization, Add, GlobalAveragePooling1D, Layer
import numpy as np
import os


# Replay Memory Class
class ReplayMemory:
    def __init__(self, size):
        self.size = size
        self.memory = []

    def add(self, experience):
        self.memory.append(experience)
        if len(self.memory) > self.size:
            self.memory.pop(0)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


# Transformer Block Class
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.add1 = Add()
        self.add2 = Add()

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(self.add1([inputs, attn_output]))
        ffn_output = self.ffn(out1)
        return self.layernorm2(self.add2([out1, ffn_output]))


# Transformer Model Creation
def create_transformer_model(seq_length, state_dim, num_heads, ff_dim, num_actions):
    inputs = Input(shape=(seq_length, state_dim))
    transformer_block = TransformerBlock(state_dim, num_heads, ff_dim)
    x = transformer_block(inputs)
    x = GlobalAveragePooling1D()(x)
    x = Dense(20, activation="relu")(x)
    outputs = Dense(num_actions, activation="tanh")(x)
    return Model(inputs=inputs, outputs=outputs)

# Huber Loss Function
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * tf.square(quadratic) + delta * linear

ALPHA = 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
GAMMA = 0.99
NUM_EPISODES = 5000
BATCH_SIZE = 64
MEMORY_SIZE = 100000
UPDATE_TARGET_EVERY = 10
LR_DECAY = 0.9
LR_DECAY_EPISODES = 100

# Environment
env = gym.make("CartPole-v1")

# Transformer parameters
seq_length = 4
embed_dim = 128
num_heads = 8
ff_dim = 256
num_actions = env.action_space.n

# Main Model
state_dim = 4  # As we now know the state shape is (4,)
model = create_transformer_model(seq_length, state_dim, num_heads, ff_dim, num_actions)
target_model = create_transformer_model(seq_length, state_dim, num_heads, ff_dim, num_actions)
optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA)
model.compile(optimizer=optimizer, loss=huber_loss)

# Replay Memory
memory = ReplayMemory(MEMORY_SIZE)

# Target Network
target_model.set_weights(model.get_weights())

def policy(sequence, explore=0.0):
    if np.random.rand() <= explore:
        return np.random.choice(num_actions)
    sequence = np.expand_dims(sequence, axis=0)  # Reshape to (1, seq_length, state_dim)
    q_values = model(sequence, training=False)
    return np.argmax(q_values[0])

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    sequence = np.tile(state, (seq_length, 1))  # Initialize sequence with the first state repeated
    episode_reward = 0
    episode_length = 0

    while True:
        action = policy(sequence, explore=EPSILON)
        next_state, reward, done, trunc, _ = env.step(action)

        memory.add((sequence, action, reward, np.roll(sequence, shift=-1, axis=0), done))
        sequence = np.roll(sequence, shift=-1, axis=0)
        sequence[-1] = next_state

        episode_reward += reward
        episode_length += 1

        if done:
            break

        if len(memory.memory) >= BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
            sequences, actions, rewards, next_sequences, dones = zip(*batch)

            sequences = np.array(sequences)
            next_sequences = np.array(next_sequences)

            # Double DQN
            target_qs = model(sequences, training=False).numpy()
            next_actions = np.argmax(model(next_sequences, training=False).numpy(), axis=1)
            next_qs = target_model(next_sequences, training=False).numpy()

            for i in range(BATCH_SIZE):
                if dones[i]:
                    target_qs[i, actions[i]] = rewards[i]
                else:
                    target_qs[i, actions[i]] = rewards[i] + GAMMA * next_qs[i, next_actions[i]]

            model.train_on_batch(sequences, target_qs)

    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

    if episode % UPDATE_TARGET_EVERY == 0:
        target_model.set_weights(model.get_weights())

    # Learning rate decay
    if episode % LR_DECAY_EPISODES == 0 and episode > 0:
        current_lr = model.optimizer.learning_rate.numpy()
        new_lr = current_lr * LR_DECAY
        model.optimizer.learning_rate.assign(new_lr)
        print(f"Learning rate decayed to: {new_lr}")

    print(f"Episode: {episode}, Length: {episode_length}, Reward: {episode_reward}, Epsilon: {EPSILON:.4f}, LR: {model.optimizer.learning_rate.numpy():.6f}")

model_save_path = "new_cartpole_transformer_model"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
env.close()
