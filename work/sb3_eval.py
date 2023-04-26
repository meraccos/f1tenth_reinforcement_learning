from stable_baselines3 import PPO
from utils import create_env
from stable_baselines3.common.evaluation import evaluate_policy

maps = list(range(1, 450))

# random.seed(72)

env = create_env(maps=maps, seed=5)

model = "/Users/meraj/workspace/f1tenth_gym/work/models/1lap_b2048_g9999_600k"

model = PPO.load(path=model, env=env)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)

print(mean_reward, std_reward)
