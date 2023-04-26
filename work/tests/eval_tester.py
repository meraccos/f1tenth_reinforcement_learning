from stable_baselines3 import PPO
from utils import create_env
import random

maps = list(range(1, 450))

# random.seed(72)

env = create_env(maps=maps, seed=5)
env.training = False

model = "/Users/meraj/workspace/f1tenth_gym/work/models/1lap_b2048_g9999_600k"

model = PPO.load(path=model, env=env)

obs = env.reset()
done = False

while not done:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render(mode="human_fast")
