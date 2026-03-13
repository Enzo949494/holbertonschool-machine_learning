import tensorflow as tf
tf.keras.__version__ = tf.__version__

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from train import NumpyObservation, build_model, FixWrappers


def build_play_env():
    env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)
    env = NumpyObservation(env)
    env = FixWrappers(env)
    return env


nb_actions = 4
input_shape = (84, 84)

model = build_model(input_shape, nb_actions)

memory = SequentialMemory(limit=500000, window_length=4)
policy = EpsGreedyQPolicy(eps=0.05)

dqn = DQNAgent(
    model=model,
    nb_actions=nb_actions,
    memory=memory,
    nb_steps_warmup=10,
    target_model_update=10000,
    policy=policy,
    enable_double_dqn=True,
    gamma=0.99,
    batch_size=32,
    train_interval=4,
)

dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
dqn.load_weights("policy.h5")

env = build_play_env()

# Utiliser une boucle manuelle au lieu de dqn.test() qui peut hang
nb_episodes = 100

for episode in range(nb_episodes):
    observation = env.reset()
    episode_reward = 0
    episode_steps = 0
    done = False
    
    while not done:
        # Utiliser la policy directement pour l'action
        state = dqn.memory.get_recent_state(observation)
        state = np.array([state])  # Ajouter dimension batch
        q_values = dqn.model.predict(state, verbose=0)
        action = dqn.policy.select_action(q_values=q_values[0])  # Prendre le premier élément du batch
        
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        episode_steps += 1
        env.render()
    
    print(f"Episode {episode + 1}: reward={episode_reward}, steps={episode_steps}")

print(f"\nCompleted {nb_episodes} episodes!")
env.close()
