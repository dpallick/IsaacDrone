import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

from environments.drone_hover_env import DroneHoverEnv, DroneHoverEnvCfg
from controllers.pid_controller import DroneWaypointPID

def sample_waypoints(device):
    waypoints = torch.zeros(4, 3, device=device)

    waypoints[:, 0] = 0.0
    waypoints[:, 1] = 0.0
    waypoints[:, 2] = torch.empty(4, device=device).uniform_(0.3, 1.0)

    return waypoints


def main():
    env_cfg = DroneHoverEnvCfg()

    env = DroneHoverEnv(
        cfg=env_cfg,
        render_mode="human",
    )

    dt = env.cfg.sim.dt * env.cfg.decimation

    controller = DroneWaypointPID(
        dt=dt,
        device=env.device,
    )

    obs, _ = env.reset()
    controller.reset(env.num_envs)

    waypoints = sample_waypoints(env.device)
    waypoint_id = 0

    print("\nRandom waypoints:")
    for i, wp in enumerate(waypoints):
        print(f"{i}: {wp.detach().cpu().numpy()}")

    target = waypoints[waypoint_id].unsqueeze(0)
    controller.set_target(target)

    while simulation_app.is_running():
        with torch.inference_mode():
            policy_obs = obs["policy"]

            pos = policy_obs[:, 0:3]
            dist = torch.linalg.norm(pos - target, dim=-1)

            if dist.item() < 0.08:
                waypoint_id += 1

                if waypoint_id >= 4:
                    print("Completed all 4 waypoints.")
                    waypoint_id = 0
                    waypoints = sample_waypoints(env.device)

                    print("\nNew random waypoints:")
                    for i, wp in enumerate(waypoints):
                        print(f"{i}: {wp.detach().cpu().numpy()}")

                target = waypoints[waypoint_id].unsqueeze(0)
                controller.set_target(target)
                print(f"New target {waypoint_id}: {target.detach().cpu().numpy()}")

            actions = controller.act(policy_obs)

            obs, reward, terminated, truncated, info = env.step(actions)

            if torch.any(terminated | truncated):
                obs, _ = env.reset()
                controller.reset(env.num_envs)

                waypoint_id = 0
                waypoints = sample_waypoints(env.device)
                target = waypoints[waypoint_id].unsqueeze(0)
                controller.set_target(target)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()