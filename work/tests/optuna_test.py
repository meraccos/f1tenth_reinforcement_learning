import optuna
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from optuna import Trial
from optuna.study import Study

from stable_baselines3 import PPO
from utils import TensorboardCallback, create_env
import random

maps = list(range(1, 250))

# random.seed(5)

# env = create_env(maps=maps, seed=5)

class ProgressCallback:
    def __init__(self):
        self.trial_count = 0

    def __call__(self, study: Study, trial: Trial) -> None:
        self.trial_count += 1
        print(f"Trial {self.trial_count}: reward={trial.value}, params={trial.params}")


def objective(trial):
    # Create the custom gym environment
    # env = make_vec_env('YourCustomGymEnvironment-v0', n_envs=4)
    # env = DummyVecEnv([lambda: create_env(maps=[0], seed=5) for _ in range(4)])
    env = DummyVecEnv([lambda: create_env(maps=maps, seed=5)])


    # Define hyperparameters to optimize
    n_steps = trial.suggest_int('n_steps', 16, 2048)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 1, log=True)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
    n_epochs = trial.suggest_int('n_epochs', 1, 10)

    # Create the PPO agent
    model = PPO("MultiInputPolicy", env, verbose=0, n_steps=n_steps, gamma=gamma, learning_rate=learning_rate,
                ent_coef=ent_coef, clip_range=clip_range, n_epochs=n_epochs)
    # model = PPO("MultiInputPolicy", env, verbose=0, learning_rate=learning_rate)

    # Train the agent
    # eval_env = gym.make('YourCustomGymEnvironment-v0')
    eval_env = create_env(maps=maps, seed=5)
    eval_callback = EvalCallback(eval_env, best_model_save_path='./models/', log_path='./logs/',
                                 eval_freq=500, deterministic=True, render=False)
    model.learn(total_timesteps=10000, callback=eval_callback)

    # Evaluate the agent
    num_episodes = 5
    rewards = []
    for _ in range(num_episodes):
        obs = eval_env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    mean_reward = np.mean(rewards)

    return mean_reward


study = optuna.create_study(direction='maximize')

progress_callback = ProgressCallback()
study.optimize(objective, n_trials=10, callbacks=[progress_callback])
# study.optimize(objective, n_trials=100)

print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))