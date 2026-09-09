import argparse
import os

import rclpy
from stable_baselines3 import PPO

from turtlebot3_rl.gym_env import TurtleBot3GazeboEnv

DEFAULT_MODEL_PATH = os.path.expanduser("~/turtlebot3_rl_ws/models/ppo_turtlebot3")


def parse_args():
    parser = argparse.ArgumentParser(description="Run a trained TurtleBot3 policy in Gazebo")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()

    env = TurtleBot3GazeboEnv()
    model = PPO.load(args.model, env=env, device="cpu")

    try:
        for episode in range(1, args.episodes + 1):
            obs, info = env.reset()
            terminated = truncated = False
            episode_reward = 0.0
            steps = 0

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1

            outcome = "goal reached" if info.get("goal_reached") else (
                "collision" if info.get("collision") else "timed out"
            )
            print(f"Episode {episode}: {outcome}, reward={episode_reward:.2f}, steps={steps}")
    finally:
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
