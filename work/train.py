import random

from stable_baselines3 import PPO
from utils import TensorboardCallback, create_env


# Define the custom learning rate schedule function
def linear_schedule(initial_learning_rate: float):
    def schedule(progress_remaining: float):
        return initial_learning_rate * progress_remaining

    return schedule


if __name__ == "__main__":
    save_interval = 100_000
    log_name = "obstacle_v_newaction2_b2048_s3"

    save_path = f"./models/{log_name}"
    log_dir = "./metrics/"
    maps = list(range(1, 450))

    random.seed(8)

    initial_learning_rate = 0.0001
    lr_schedule = linear_schedule(initial_learning_rate)

    env = create_env(maps=maps, seed=8)

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
        # gamma=0.9151909702089307,
        # n_epochs=10,
        # clip_range=0.14766513397642733,  # Adjust this value as needed
        tensorboard_log=log_dir,
        device="cpu",
    )

    combined_callback = TensorboardCallback(save_interval, save_path, verbose=1)
    model.learn(
        total_timesteps=10000_000,
        callback=combined_callback,
        progress_bar=True,
        tb_log_name=log_name,
    )

    env.close()
