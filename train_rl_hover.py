import argparse
import csv
import os
import random
import shutil
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--iters", type=int, default=2000)
parser.add_argument("--steps", type=int, default=128)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--minibatch_size", type=int, default=1024)
parser.add_argument("--save_path", type=str, default=None,
                    help="Optional extra copy of the best checkpoint (old behaviour).")
parser.add_argument("--run_dir", type=str, default="runs/hover",
                    help="Folder for best.pt, latest.pt, progress.csv, DONE marker.")
parser.add_argument("--resume", action="store_true",
                    help="Continue from <run_dir>/latest.pt if it exists.")
parser.add_argument("--save_every", type=int, default=25, help="Write latest.pt every N iters.")
parser.add_argument("--time_limit_min", type=float, default=0.0,
                    help="Stop cleanly (saving latest.pt) after this many minutes; 0 = no limit.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--obs_mode", choices=["onboard", "privileged"], default="onboard",
                    help="onboard = only real-drone sensors (default); privileged = ground truth")
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
    t_start = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.run_dir, exist_ok=True)

    env_cfg = DroneHoverEnvCfgDD24() if args.airframe == "dd24" else DroneHoverEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.obs_mode = args.obs_mode
    env_cfg.__post_init__()   # recompute observation sizes for the chosen mode
    env_cfg.hover_height = 1.0

    env = DroneHoverEnv(cfg=env_cfg, render_mode=None)
    device = env.device
    obs_dim = env_cfg.observation_space
    # Asymmetric actor-critic: the critic sees ground truth (training only).
    critic_dim = env_cfg.state_space or obs_dim
    print(f"obs_mode={args.obs_mode} actor inputs={obs_dim} critic inputs={critic_dim}")

    def critic_in(o):
        return o["critic"] if "critic" in o else o["policy"]

    actor = RLHoverActor(obs_dim=obs_dim, action_dim=4).to(device)
    critic = Critic(obs_dim=critic_dim).to(device)

    log_std = nn.Parameter(torch.ones(4, device=device) * -2.5)

    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()) + [log_std],
        lr=args.lr,
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

    def checkpoint(it, best_reward):
        return {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "log_std": log_std.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "iter": it,
            "best_reward": best_reward,
            "obs_dim": obs_dim,
            "critic_dim": critic_dim,
            "obs_mode": args.obs_mode,
            "obs_history": env_cfg.obs_history,
            "hover_motor": hover_motor,
            "residual_scale": residual_scale,
            "airframe": args.airframe,
            "seed": args.seed,
        }

    def save_atomic(obj, path):
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)   # never leaves a half-written checkpoint if the job is killed

    latest_path = os.path.join(args.run_dir, "latest.pt")
    best_path = os.path.join(args.run_dir, "best.pt")
    csv_path = os.path.join(args.run_dir, "progress.csv")
    done_path = os.path.join(args.run_dir, "DONE")

    start_it = 0
    best_reward = -1e9
    if args.resume and os.path.isfile(latest_path):
        ck = torch.load(latest_path, map_location=device)
        actor.load_state_dict(ck["actor"])
        critic.load_state_dict(ck["critic"])
        with torch.no_grad():
            log_std.copy_(ck["log_std"].to(device))
        optimizer.load_state_dict(ck["optimizer"])
        start_it = ck["iter"] + 1
        best_reward = ck["best_reward"]
        print(f"Resumed from {latest_path} at iter {start_it} (best reward {best_reward:.4f})")

    new_csv = not os.path.isfile(csv_path) or start_it == 0
    csv_file = open(csv_path, "w" if new_csv else "a", newline="")
    writer = csv.writer(csv_file)
    if new_csv:
        writer.writerow(["iter", "reward", "ep_len", "log_std", "crash_frac", "net_frac", "minutes"])

    obs, _ = env.reset()
    it = start_it - 1
    stopped_early = False

    for it in range(start_it, args.iters):
        obs_buf = []
        crit_buf = []
        action_raw_buf = []
        logprob_buf = []
        reward_buf = []
        done_buf = []
        value_buf = []
        ep_done_buf = []
        n_crash = torch.zeros((), device=device)
        n_net = torch.zeros((), device=device)

        for _ in range(args.steps):
            policy_obs = obs["policy"]
            obs_prev = obs

            with torch.no_grad():
                mean = actor(policy_obs)
                std = torch.exp(log_std).expand_as(mean)
                dist = torch.distributions.Normal(mean, std)

                action_raw = dist.sample()
                action_raw_clamped = torch.clamp(action_raw, 0.0, 1.0)
                logprob = dist.log_prob(action_raw).sum(dim=-1)
                value = critic(critic_in(obs))

                residual = (action_raw_clamped - 0.5) * residual_scale
                actions = torch.clamp(hover_motor + residual, 0.0, 1.0)

            obs, env_reward, terminated, truncated, info = env.step(actions)

            # Reward is computed inside the env from ground-truth state, and
            # includes the net/geofence and crash penalties.
            reward = env_reward
            done = terminated | truncated

            obs_buf.append(policy_obs)
            crit_buf.append(critic_in(obs_prev))
            action_raw_buf.append(action_raw)
            logprob_buf.append(logprob)
            reward_buf.append(reward)
            done_buf.append(done.float())
            value_buf.append(value)

            # DirectRLEnv already auto-resets only the finished envs inside step().
            # (The old global env.reset() here restarted ALL envs whenever ANY
            # one crashed, so episodes almost never got long.)
            ep_done_buf.append(done.float())
            n_crash += (env.term_crashed | env.term_flipped).float().sum()
            n_net += env.term_net.float().sum()

        with torch.no_grad():
            next_value = critic(critic_in(obs))

        obs_buf = torch.stack(obs_buf)
        crit_buf = torch.stack(crit_buf)
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
        b_crit = crit_buf.reshape(-1, critic_dim)
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

                new_value = critic(b_crit[mb_idx])
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

        minutes = (time.time() - t_start) / 60.0
        crash_frac = n_crash.item() / max(n_eps, 1.0)
        net_frac = n_net.item() / max(n_eps, 1.0)

        print(
            f"iter={it:04d} "
            f"reward={mean_reward:.4f} "
            f"ep_len~{avg_ep_len:.0f} "
            f"crash={crash_frac:.2f} net={net_frac:.2f} "
            f"log_std={log_std.mean().item():.3f} "
            f"t={minutes:.1f}min",
            flush=True,
        )
        writer.writerow([it, f"{mean_reward:.5f}", f"{avg_ep_len:.1f}",
                         f"{log_std.mean().item():.4f}", f"{crash_frac:.4f}",
                         f"{net_frac:.4f}", f"{minutes:.2f}"])
        csv_file.flush()

        if mean_reward > best_reward:
            best_reward = mean_reward
            save_atomic(checkpoint(it, best_reward), best_path)
            if args.save_path:
                os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
                shutil.copy(best_path, args.save_path)

        if (it + 1) % args.save_every == 0:
            save_atomic(checkpoint(it, best_reward), latest_path)

        if args.time_limit_min > 0 and minutes >= args.time_limit_min:
            print(f"Time limit reached ({minutes:.1f} min) - saving and stopping at iter {it}.")
            stopped_early = True
            break

    save_atomic(checkpoint(it, best_reward), latest_path)
    csv_file.close()
    if not stopped_early:
        open(done_path, "w").write(f"finished {args.iters} iters, best reward {best_reward:.4f}\n")
        print(f"Training complete -> {best_path}")
    env.close()


if __name__ == "__main__":
    # Isaac Sim's simulation_app.close() can hang forever in headless containers
    # (it did on Oscar: 6+ hours after training had finished). Everything is already
    # written to disk by now, so exit the process directly instead.
    import os as _os, sys as _sys, traceback as _tb
    _code = 0
    try:
        main()
    except Exception:
        _tb.print_exc()
        _code = 1
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(_code)