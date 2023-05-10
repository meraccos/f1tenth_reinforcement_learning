from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from utils import create_env, linear_schedule
from callbacks import TensorboardCallback, CustomEvalCallback

import random
from torch.nn import Mish


if __name__ == "__main__":
    save_interval = 5e4
    eva_freq = 5e4
    n_eval_episodes = 20
    learn_steps = 1e7
    log_name = "dr_01"

    save_path = f"./models/{log_name}"
    log_dir = "./metrics/"
    maps = list(range(1, 450))

    random.seed(8)

    initial_learning_rate = 0.00015
    lr_schedule = linear_schedule(initial_learning_rate)

    env = create_env(maps=maps, seed=8)
    eval_env = create_env(maps=maps, seed=8)

    policy_kwargs = dict(activation_fn=Mish,
                         net_arch=dict(pi=[32, 32], vf=[32, 32]))

    model = PPO(
        "MlpPolicy",
        env,
        verbose=2,
        n_steps=1024,
        ent_coef=0.01,
        # learning_rate=0.0001,
        learning_rate=lr_schedule,
        batch_size=128,
        # max_grad_norm=0.5,
        # gae_lambda=0.95,
        gamma=0.993,
        n_epochs=10,
        clip_range=0.25,
        tensorboard_log=log_dir,
        device="cpu",
        policy_kwargs=policy_kwargs
    )
        
    callbacks = CallbackList([TensorboardCallback(save_interval, save_path), 
                              CustomEvalCallback(eval_env,
                                                best_model_save_path="./best_models/",
                                                log_path=log_dir,
                                                n_eval_episodes=n_eval_episodes,
                                                eval_freq=eva_freq)])
    
    model.learn(
        total_timesteps=learn_steps,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=log_name,
    )

    env.close()
