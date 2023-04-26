import random

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from utils import create_env, linear_schedule
from callbacks import TensorboardCallback, CustomEvalCallback


# Define the custom learning rate schedule function

if __name__ == "__main__":
    save_interval = 100_000
    eva_freq = 500
    n_eval_episodes = 1
    log_name = "two_cbs"

    save_path = f"./models/{log_name}"
    log_dir = "./metrics/"
    maps = list(range(1, 450))

    random.seed(8)

    initial_learning_rate = 0.0003
    lr_schedule = linear_schedule(initial_learning_rate)

    env = create_env(maps=maps, seed=8)
    eval_env = create_env(maps=maps, seed=8)

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        # n_steps=1865,
        # ent_coef=0.032233448682464166,
        learning_rate=0.0001,
        # learning_rate=lr_schedule,
        batch_size=2048,
        # vf_coef=0.5,
        # max_grad_norm=0.5,
        # gae_lambda=0.95,
        gamma=0.99,
        # n_epochs=10,
        # clip_range=0.14766513397642733,  # Adjust this value as needed
        tensorboard_log=log_dir,
        device="cpu",
    )
    
    callbacks = CallbackList([TensorboardCallback(save_interval, save_path), 
                              CustomEvalCallback(eval_env,
                                                best_model_save_path="./best_models/",
                                                log_path=log_dir,
                                                n_eval_episodes=n_eval_episodes,
                                                eval_freq=eva_freq)])
    
    model.learn(
        total_timesteps=10000_000,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=log_name,
    )

    env.close()
