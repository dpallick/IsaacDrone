import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--iters", type=int, default=2000)
parser.add_argument("--steps", type=int, default=128)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--minibatch_size", type=int, default=1024)
parser.add_argument("--save_path", type=str, default="checkpoints/rl_hover.pt")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import torch.optim as optim

from environments.drone_hover_env import DroneHoverEnv, DroneHoverEnvCfg
from controllers.rl_hover_controller import RLHoverActor


class Critic(nn.Module):
    def __init__(self, obs_dim=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


def compute_reward(obs, actions, hover_motor):
    pos = obs[:, 0:3]
    z_error = obs[:, 3]
    lin_vel = obs[:, 8:11]
    ang_vel = obs[:, 11:14]

    return (
        1.0
        - 2.0 * torch.abs(z_error)
        - 0.25 * torch.linalg.norm(pos[:, 0:2], dim=-1)
        - 0.10 * torch.linalg.norm(lin_vel, dim=-1)
        - 0.05 * torch.linalg.norm(ang_vel, dim=-1)
        - 0.02 * torch.linalg.norm(actions - hover_motor, dim=-1)
    )


def main():
    env_cfg = DroneHoverEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.hover_height = 1.0

    env = DroneHoverEnv(cfg=env_cfg, render_mode=None)
    device = env.device

    actor = RLHoverActor(obs_dim=14, action_dim=4).to(device)
    critic = Critic(obs_dim=14).to(device)

    log_std = nn.Parameter(torch.ones(4, device=device) * -2.5)

    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()) + [log_std],
        lr=3e-4,
    )

    gamma = 0.99
    gae_lambda = 0.95
    clip_eps = 0.2
    value_coef = 0.5
    entropy_coef = 0.01

    hover_motor = 1.0 / 2.2
    residual_scale = 0.15

    obs, _ = env.reset()
    best_reward = -1e9

    for it in range(args.iters):
        obs_buf = []
        action_raw_buf = []
        logprob_buf = []
        reward_buf = []
        done_buf = []
        value_buf = []

        for _ in range(args.steps):
            policy_obs = obs["policy"]

            with torch.no_grad():
                mean = actor(policy_obs)
                std = torch.exp(log_std).expand_as(mean)
                dist = torch.distributions.Normal(mean, std)

                action_raw = dist.sample()
                action_raw_clamped = torch.clamp(action_raw, 0.0, 1.0)
                logprob = dist.log_prob(action_raw).sum(dim=-1)
                value = critic(policy_obs)

                residual = (action_raw_clamped - 0.5) * residual_scale
                actions = torch.clamp(hover_motor + residual, 0.0, 1.0)

            obs, env_reward, terminated, truncated, info = env.step(actions)
            next_obs = obs["policy"]

            reward = compute_reward(next_obs, actions, hover_motor)
            done = terminated | truncated

            obs_buf.append(policy_obs)
            action_raw_buf.append(action_raw)
            logprob_buf.append(logprob)
            reward_buf.append(reward)
            done_buf.append(done.float())
            value_buf.append(value)

            if torch.any(done):
                obs, _ = env.reset()

        with torch.no_grad():
            next_value = critic(obs["policy"])

        obs_buf = torch.stack(obs_buf)
        action_raw_buf = torch.stack(action_raw_buf)
        logprob_buf = torch.stack(logprob_buf)
        reward_buf = torch.stack(reward_buf)
        done_buf = torch.stack(done_buf)
        value_buf = torch.stack(value_buf)

        advantages = torch.zeros_like(reward_buf)
        last_gae = torch.zeros(args.num_envs, device=device)

        for t in reversed(range(args.steps)):
            if t == args.steps - 1:
                next_nonterminal = 1.0 - done_buf[t]
                next_values = next_value
            else:
                next_nonterminal = 1.0 - done_buf[t + 1]
                next_values = value_buf[t + 1]

            delta = reward_buf[t] + gamma * next_values * next_nonterminal - value_buf[t]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + value_buf

        b_obs = obs_buf.reshape(-1, 14)
        b_actions_raw = action_raw_buf.reshape(-1, 4)
        b_old_logprob = logprob_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = value_buf.reshape(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        batch_size = b_obs.shape[0]
        indices = torch.arange(batch_size, device=device)

        for _ in range(args.epochs):
            perm = indices[torch.randperm(batch_size, device=device)]

            for start in range(0, batch_size, args.minibatch_size):
                mb_idx = perm[start:start + args.minibatch_size]

                mean = actor(b_obs[mb_idx])
                std = torch.exp(log_std).expand_as(mean)
                dist = torch.distributions.Normal(mean, std)

                new_logprob = dist.log_prob(b_actions_raw[mb_idx]).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_logprob - b_old_logprob[mb_idx])

                pg_loss_1 = -b_advantages[mb_idx] * ratio
                pg_loss_2 = -b_advantages[mb_idx] * torch.clamp(
                    ratio,
                    1.0 - clip_eps,
                    1.0 + clip_eps,
                )

                policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                new_value = critic(b_obs[mb_idx])
                value_loss = 0.5 * ((new_value - b_returns[mb_idx]) ** 2).mean()

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()) + [log_std],
                    1.0,
                )
                optimizer.step()

        mean_reward = reward_buf.mean().item()

        print(
            f"iter={it:04d} "
            f"reward={mean_reward:.4f} "
            f"log_std={log_std.mean().item():.3f}"
        )

        if mean_reward > best_reward:
            best_reward = mean_reward
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(
                {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "log_std": log_std.detach().cpu(),
                    "best_reward": best_reward,
                },
                args.save_path,
            )
            print(f"saved best reward {best_reward:.4f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()