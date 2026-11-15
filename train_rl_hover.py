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
parser.add_argument("--airframe", choices=["crazyflie", "dd24"], default="crazyflie")
parser.add_argument("--residual_frac", type=float, default=0.33,
                    help="Policy thrust authority around hover, as a fraction of the hover command.")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import torch.optim as optim

from environments.drone_hover_env import DroneHoverEnv, DroneHoverEnvCfg, DroneHoverEnvCfgDD24
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


def main():
    env_cfg = DroneHoverEnvCfgDD24() if args.airframe == "dd24" else DroneHoverEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.hover_height = 1.0

    env = DroneHoverEnv(cfg=env_cfg, render_mode=None)
    device = env.device
    obs_dim = env_cfg.observation_space

    actor = RLHoverActor(obs_dim=obs_dim, action_dim=4).to(device)
    critic = Critic(obs_dim=obs_dim).to(device)

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

    # Hover command comes from the airframe's thrust-to-weight (was hard-coded to
    # the Crazyflie's 1/2.2, which makes a DD24 permanently over-thrust).
    hover_motor = env.hover_motor
    residual_scale = 2.0 * args.residual_frac * hover_motor
    print(f"hover_motor={hover_motor:.3f} residual_scale={residual_scale:.3f}")

    obs, _ = env.reset()
    best_reward = -1e9

    for it in range(args.iters):
        obs_buf = []
        action_raw_buf = []
        logprob_buf = []
        reward_buf = []
        done_buf = []
        value_buf = []
        ep_done_buf = []

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

            # Reward is computed inside the env from ground-truth state, and
            # includes the net/geofence and crash penalties.
            reward = env_reward
            done = terminated | truncated

            obs_buf.append(policy_obs)
            action_raw_buf.append(action_raw)
            logprob_buf.append(logprob)
            reward_buf.append(reward)
            done_buf.append(done.float())
            value_buf.append(value)

            # DirectRLEnv already auto-resets only the finished envs inside step().
            # (The old global env.reset() here restarted ALL envs whenever ANY
            # one crashed, so episodes almost never got long.)
            ep_done_buf.append(done.float())

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
            # done_buf[t] is the outcome of the step taken at t, so it gates the
            # bootstrap from t+1 (the old code used done_buf[t+1], off by one).
            next_nonterminal = 1.0 - done_buf[t]
            next_values = next_value if t == args.steps - 1 else value_buf[t + 1]

            delta = reward_buf[t] + gamma * next_values * next_nonterminal - value_buf[t]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + value_buf

        b_obs = obs_buf.reshape(-1, obs_dim)
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
        n_eps = torch.stack(ep_done_buf).sum().item()
        avg_ep_len = args.steps * args.num_envs / max(n_eps, 1.0)

        print(
            f"iter={it:04d} "
            f"reward={mean_reward:.4f} "
            f"ep_len~{avg_ep_len:.0f} "
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
                    "obs_dim": obs_dim,
                    "hover_motor": hover_motor,
                    "residual_scale": residual_scale,
                    "airframe": args.airframe,
                },
                args.save_path,
            )
            print(f"saved best reward {best_reward:.4f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()