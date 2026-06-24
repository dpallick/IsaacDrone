import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()

controller_group = parser.add_mutually_exclusive_group(required=True)
controller_group.add_argument("--PID", action="store_true")
controller_group.add_argument("--RLHover", action="store_true")

parser.add_argument(
    "--rl_checkpoint",
    type=str,
    default=None,
    help="Path to RL hover policy checkpoint.",
)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import torch

from environments.drone_hover_env import DroneHoverEnv, DroneHoverEnvCfg
from controllers.pid_controller import DroneWaypointPID
from controllers.rl_hover_controller import RLHoverController


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

    if args.PID:
        controller = DroneWaypointPID(
            dt=dt,
            device=env.device,
        )
        print("Using PID controller")

    elif args.RLHover:
        controller = RLHoverController(
            device=env.device,
            checkpoint=args.rl_checkpoint,
        )
        print("Using RL hover controller")

        if args.rl_checkpoint is None:
            print("WARNING: No RL checkpoint provided. Running untrained policy.")

    obs, _ = env.reset()
    controller.reset(env.num_envs)

    waypoints = sample_waypoints(env.device)
    waypoint_id = 0
    waypoint_timer = 0.0
    target = waypoints[waypoint_id].unsqueeze(0)

    if hasattr(controller, "set_target"):
        controller.set_target(target)

    while simulation_app.is_running():
        with torch.inference_mode():
            policy_obs = obs["policy"]

            # Time-based waypoint switching (position is not available with acceleration observations)
            waypoint_timer += dt
            if waypoint_timer > 5.0:
                waypoint_id += 1
                waypoint_timer = 0.0

                if waypoint_id >= 4:
                    waypoint_id = 0
                    waypoints = sample_waypoints(env.device)

                target = waypoints[waypoint_id].unsqueeze(0)
                if hasattr(controller, "set_target"):
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

                if hasattr(controller, "set_target"):
                    controller.set_target(target)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()