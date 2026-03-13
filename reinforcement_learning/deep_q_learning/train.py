import tensorflow as tf
tf.keras.__version__ = tf.__version__

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Permute
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.callbacks import Callback


class FixWrappers(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def render(self, mode=None, **kwargs):
        return self.env.render()



class NumpyObservation(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs)


class SaveBestModel(Callback):
    def __init__(self, filepath):
        self.filepath = filepath
        self.best_mean_reward = -np.inf
        self.recent_rewards = []

    def on_episode_end(self, episode, logs={}):
        reward = logs.get('episode_reward', 0)
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0)
        mean_reward = np.mean(self.recent_rewards)
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model.save_weights(self.filepath, overwrite=True)
            print(f"\n✅ Meilleur modèle sauvegardé — mean_reward: {mean_reward:.3f}")


def build_env():
    env = gym.make("BreakoutNoFrameskip-v4", render_mode=None)
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)
    env = NumpyObservation(env)
    env = FixWrappers(env)
    return env


def build_model(input_shape, nb_actions):
    model = Sequential([
        Permute((2, 3, 1), input_shape=(4,) + input_shape),
        Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),
        Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
        Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(nb_actions, activation='linear'),
    ])
    return model


if __name__ == "__main__":
    env = build_env()
    nb_actions = env.action_space.n
    input_shape = (84, 84)

    model = build_model(input_shape, nb_actions)
    model.summary()

    memory = SequentialMemory(limit=200000, window_length=4)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.0,
        value_min=0.1,
        value_test=0.05,
        nb_steps=500000
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=50000,
        target_model_update=10000,
        policy=policy,
        enable_double_dqn=True,
        gamma=0.99,
        batch_size=32,
        train_interval=4,
    )

    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    callbacks = [SaveBestModel("policy.h5")]

    dqn.fit(
        env,
        nb_steps=1000000,
        visualize=False,
        verbose=2,
        log_interval=10000,
        callbacks=callbacks,
    )

    env.close()
