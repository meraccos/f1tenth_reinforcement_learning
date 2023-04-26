from stable_baselines3 import PPO
from utils import create_env
from stable_baselines3.common.evaluation import evaluate_policy

maps = list(range(1, 450))
env = create_env(maps=maps, seed=5)

model = "models/trial_600k"

model = PPO.load(path=model, env=env)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

