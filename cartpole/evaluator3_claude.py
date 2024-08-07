import gymnasium as gym
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
q_net = load_model("improved_q_learning_q_net")


def policy(state, explore=0.0):
    if np.random.random() < explore:
        return env.action_space.sample()
    q_values = q_net.predict(state[np.newaxis], verbose=0)
    return np.argmax(q_values[0])


def evaluate_model(num_episodes=100, render=False):
    env = gym.make("CartPole-v1", render_mode="rgb_array" if render else None)
    rewards = []
    steps = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        episode_steps = 0

        while not done:
            if render:
                env.render()

            action = policy(state)
            next_state, reward, done, trunc, _ = env.step(action)

            state = next_state
            total_reward += reward
            episode_steps += 1

        rewards.append(total_reward)
        steps.append(episode_steps)
        print(f"Episode {episode + 1}: Reward = {total_reward}, Steps = {episode_steps}")

    env.close()
    return rewards, steps


# Run evaluation
rewards, steps = evaluate_model(num_episodes=100)

# Calculate statistics
avg_reward = np.mean(rewards)
std_reward = np.std(rewards)
max_reward = np.max(rewards)
min_reward = np.min(rewards)

print(f"\nEvaluation Results:")
print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
print(f"Max Reward: {max_reward}")
print(f"Min Reward: {min_reward}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(rewards)
plt.title('Rewards over Episodes')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

plt.figure(figsize=(12, 6))
plt.hist(rewards, bins=20)
plt.title('Distribution of Rewards')
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.show()