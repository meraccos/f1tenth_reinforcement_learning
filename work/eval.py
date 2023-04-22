from stable_baselines3 import PPO
from utils import create_env

maps = list(range(1, 200))

env = create_env(maps=maps)
env.training = False

model = "/Users/meraj/workspace/f1tenth_gym/work/models/norm_rew3_batch256_10000000"

model = PPO.load(path=model, env=env)

obs = env.reset()
done = False

while not done:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render(mode="human_fast")
