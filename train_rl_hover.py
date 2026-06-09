import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--iters", type=int, default=1000)
parser.add_argument("--steps", type=int, default=64)
parser.add_argument("--save_path", type=str, default="checkpoints/rl_hover.pt")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import torch.optim as optim

from environments.drone_hover_env import DroneHoverEnv, DroneHoverEnvCfg
from controllers.rl_hover_controller import RLHoverPolicy


def main():
    env_cfg = DroneHoverEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.hover_height = 1.0

    env = DroneHoverEnv(cfg=env_cfg, render_mode=None)
    device = env.device

    policy = RLHoverPolicy(obs_dim=14, action_dim=4).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    gamma = 0.99
    hover_motor = 1.0 / 2.2
    residual_scale = 0.3
    exploration_std = 0.10

    obs, _ = env.reset()

    for it in range(args.iters):
        log_probs = []
        rewards = []

        for _ in range(args.steps):
            policy_obs = obs["policy"]

            residual_mean = policy(policy_obs)

            dist = torch.distributions.Normal(
                residual_mean,
                exploration_std,
            )

            residual_sample = torch.clamp(dist.sample(), 0.0, 1.0)
            log_prob = dist.log_prob(residual_sample).sum(dim=-1)

            residual = (residual_sample - 0.5) * residual_scale
            actions = hover_motor + residual
            actions = torch.clamp(actions, 0.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(actions)

            pos = policy_obs[:, 0:3]
            z_error = policy_obs[:, 3]
            lin_vel = policy_obs[:, 8:11]
            ang_vel = policy_obs[:, 11:14]

            shaped_reward = (
                -2.0 * torch.abs(z_error)
                -0.25 * torch.linalg.norm(pos[:, 0:2], dim=-1)
                -0.10 * torch.linalg.norm(lin_vel, dim=-1)
                -0.05 * torch.linalg.norm(ang_vel, dim=-1)
            )

            log_probs.append(log_prob)
            rewards.append(shaped_reward)

            if torch.any(terminated | truncated):
                obs, _ = env.reset()

        returns = []
        running_return = torch.zeros(args.num_envs, device=device)

        for r in reversed(rewards):
            running_return = r + gamma * running_return
            returns.insert(0, running_return)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns)

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = -(log_probs * returns).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_reward = torch.stack(rewards).mean().item()

        print(
            f"iter={it:04d} "
            f"loss={loss.item():.4f} "
            f"reward={mean_reward:.4f}"
        )

        if it % 25 == 0:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(policy.state_dict(), args.save_path)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(policy.state_dict(), args.save_path)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()