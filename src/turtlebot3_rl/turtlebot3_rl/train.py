import argparse
import os

import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from turtlebot3_rl.gym_env import TurtleBot3GazeboEnv

DEFAULT_LOG_DIR = os.path.expanduser("~/turtlebot3_rl_ws/logs")
DEFAULT_CHECKPOINT_DIR = os.path.expanduser("~/turtlebot3_rl_ws/checkpoints")
DEFAULT_MODEL_PATH = os.path.expanduser("~/turtlebot3_rl_ws/models/ppo_turtlebot3")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a TurtleBot3 goal-reaching policy in Gazebo")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--checkpoint-freq", type=int, default=5_000)
    parser.add_argument("--resume", type=str, default=None, help="Path to a .zip checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
    os.makedirs(DEFAULT_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DEFAULT_MODEL_PATH), exist_ok=True)

    env = TurtleBot3GazeboEnv()
    env = Monitor(env, filename=os.path.join(DEFAULT_LOG_DIR, "monitor.csv"))

    if args.resume:
        model = PPO.load(args.resume, env=env, device="cpu")
    else:
        model = PPO("MlpPolicy", env,device="cpu" ,verbose=1, tensorboard_log=DEFAULT_LOG_DIR)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=DEFAULT_CHECKPOINT_DIR,
        name_prefix="ppo_turtlebot3",
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=args.resume is None,
        )
    finally:
        model.save(DEFAULT_MODEL_PATH)
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
