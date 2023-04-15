from stable_baselines3 import PPO
from utils import *
from reward import *

maps = list(range(1,60))

env = create_env(maps=maps)

model = "models/ppo_model_100000.zip"
model = PPO.load(path=model, env=env)

# model = PPO.load("/Users/meraj/workspace/f1tenth_gym/work/models/base/01_100423.zip", env=env)

obs = env.reset()
done = False

while not done:
    action, _state = model.predict(obs, deterministic=False)
    # action = np.array([[0.0, 0.0]])
    obs, reward, done, info = env.step(action)
    env.render(mode='human_fast')
    # time.sleep(0.5)
