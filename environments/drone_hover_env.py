from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets import CRAZYFLIE_CFG
from isaaclab.sensors.imu import Imu, ImuCfg

from pxr import UsdGeom

from .dd24_params import (DD24, DD24_TOF_SENSORS, TOF_MAX_RANGE_M, TOF_MIN_RANGE_M,
                          TOF_NOISE_STD_M, TOF_NOISE_STD_FRAC)
from . import arena as A


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4) in (w, x, y, z) order."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def tof_ranges(pos_local, quat, sensor_pos_b, sensor_dir_b, half_xy, height):
    """True range from each ToF sensor to the floor / net walls / net ceiling.

    pos_local: (N, 3) drone CG position relative to the arena centre (floor at z=0)
    quat:      (N, 4) drone attitude (w, x, y, z)
    sensor_pos_b, sensor_dir_b: (S, 3) mounting point and pointing direction in the body frame
    half_xy:   (2,) half-size of the net; height: ceiling height (<= 0 = no ceiling)
    Returns (range (N, S), hit_net (N, S) bool). From inside the box the nearest
    positive plane crossing is where the ray exits, i.e. what the sensor sees.
    """
    N, S = pos_local.shape[0], sensor_pos_b.shape[0]
    q = quat[:, None, :].expand(N, S, 4)
    origin = pos_local[:, None, :] + quat_rotate(q, sensor_pos_b[None].expand(N, S, 3))
    d = quat_rotate(q, sensor_dir_b[None].expand(N, S, 3))
    inf = torch.full((N, S), float("inf"), device=pos_local.device)

    def plane(axis, value):
        dk = d[..., axis]
        t = (value - origin[..., axis]) / torch.where(dk.abs() < 1e-9, torch.full_like(dk, 1e-9), dk)
        return torch.where(t > 0, t, inf)

    t_floor = plane(2, 0.0)
    t_net = torch.minimum(torch.minimum(plane(0, half_xy[0]), plane(0, -half_xy[0])),
                          torch.minimum(plane(1, half_xy[1]), plane(1, -half_xy[1])))
    if height > 0:
        t_net = torch.minimum(t_net, plane(2, height))
    rng = torch.minimum(t_floor, t_net)
    return rng, t_net < t_floor


@configclass
class DroneHoverEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 2
    action_space = 4
    # [imu_acc(3), z_err(1), quat(4), lin_vel(3), gyro_body(3), xy_err(2), tof(S)]
    # New fields are appended LAST so obs[:, 0:14] keeps the old layout (PID still works).
    # tof = range / tof_max_range in [0, 1], one per sensor in `tof_sensors`.
    observation_space = 16 + 1
    state_space = 0
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
    )
    robot = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Drone"
    )
    imu: ImuCfg = ImuCfg(prim_path="/World/envs/env_.*/Drone/body")

    hover_height = 1.0
    crash_height = A.CRASH_HEIGHT
    # Perpendicular lever arm used by the mixer below (NOT center-to-motor for an
    # X frame: lever = center_to_motor / sqrt(2)).
    arm_length = 0.046
    max_thrust_to_weight = 2.2
    yaw_coeff = 0.00001
    max_prop_visual_speed = 50000.0
    # ---- Time-of-flight range sensors -----------------------------------------
    # (name, position in body frame (m), pointing direction). Each is ray-cast from
    # its own mounting point, so tilting the drone changes the readings as it would
    # for real. The Crazyflie default is a single downward sensor at the centre.
    tof_sensors = (("down", (0.0, 0.0, -0.01), (0.0, 0.0, -1.0)),)
    tof_max_range = TOF_MAX_RANGE_M
    tof_min_range = TOF_MIN_RANGE_M
    tof_noise_std = TOF_NOISE_STD_M
    tof_noise_frac = TOF_NOISE_STD_FRAC
    # [TODO] How often an IR ToF fails to see the NET (sparse mesh) and reports
    # "out of range" instead. 0 = net always detected like a solid wall.
    tof_net_dropout = 0.0

    # Prop prims to spin for visuals. Missing prims just disable the effect.
    prop_visual_paths = (
        "m1_prop/ccw_prop",
        "m2_prop/cw_prop",
        "m3_prop/ccw_prop",
        "m4_prop/cw_prop",
    )

    # ---- Flight arena (the safety net) --------------------------------------
    # The net is modelled as a GEOFENCE, not as a physical/cloth object:
    #   * get within `net_margin` of any net face -> episode terminates + penalty
    #   * within `net_warn_band` of the margin   -> smooth shaping penalty
    # The arena is centred on each env origin. [TODO] measure the INSIDE of the net.
    arena_size_xy = A.ARENA_SIZE_XY   # set in environments/arena.py (shared with RotorPy)
    arena_height = A.ARENA_HEIGHT
    net_margin = A.NET_MARGIN
    net_warn_band = A.NET_WARN_BAND
    max_tilt_rad = 1.2           # terminate if tilted beyond this (~70 deg)

    # Spawn distribution (relative to env origin). Randomised so the policy has to
    # actively correct xy drift instead of only ever seeing a perfect start.
    spawn_xy_range = 0.5
    spawn_z_range = (0.3, 1.2)
    spawn_lin_vel_range = 0.3
    spawn_ang_vel_range = 0.5

    # ---- Reward weights ------------------------------------------------------
    # Computed in the env from ground-truth state (privileged); the policy itself
    # only sees the observation vector.
    rew_alive = 1.0
    rew_z = 2.0
    rew_xy = 1.0
    rew_lin_vel = 0.10
    rew_ang_vel = 0.05
    rew_action = 0.02
    rew_net = 2.0
    rew_crash = 10.0


@configclass
class DroneHoverEnvCfgDD24(DroneHoverEnvCfg):
    """DD24 airframe config.

    `robot` points at the DD24 USD produced by assets/dd24/build_dd24_asset.py.
    Mass and inertia come from that USD (PhysX), so they are only correct once the
    asset has been built from YOUR measured values (see dd24_params.py).
    """
    # Lever arm for an X frame = center-to-motor / sqrt(2).
    arm_length = DD24.arm_length_m / math.sqrt(2.0)
    max_thrust_to_weight = DD24.thrust_to_weight
    yaw_coeff = DD24.yaw_coeff
    # URDF-imported props are links named m{i}_prop; spin the link's visuals.
    prop_visual_paths = ("m1_prop", "m2_prop", "m3_prop", "m4_prop")
    # Five ToFs at their CAD positions (see dd24_params.DD24_TOF_SENSORS).
    tof_sensors = DD24_TOF_SENSORS
    observation_space = 16 + len(DD24_TOF_SENSORS)

    def __post_init__(self):
        super().__post_init__()
        # Imported lazily so the Crazyflie config works before the DD24 USD exists.
        from assets.dd24.dd24_cfg import DD24_CFG
        self.robot = DD24_CFG.replace(prim_path="/World/envs/env_.*/Drone")


class DroneHoverEnv(DirectRLEnv):
    cfg: DroneHoverEnvCfg

    def __init__(self, cfg: DroneHoverEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actions = torch.zeros(self.num_envs, 4, device=self.device)
        self.force_b = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.torque_b = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self.body_id = self.drone.find_bodies("body")[0]
        self.prop_joint_ids = [
            self.drone.find_joints(f"m{i}_joint")[0][0] for i in range(1, 5)
        ]

        masses = self.drone.root_physx_view.get_masses()
        self.mass = masses[0].sum().item()
        self.g = abs(self.cfg.sim.gravity[2])
        self.hover_thrust = self.mass * self.g
        self.max_total_thrust = self.hover_thrust * self.cfg.max_thrust_to_weight
        self.max_motor_thrust = self.max_total_thrust / 4.0
        # Per-motor command (0..1) that exactly cancels gravity.
        self.hover_motor = 1.0 / self.cfg.max_thrust_to_weight

        self._target_local = torch.tensor(
            [0.0, 0.0, self.cfg.hover_height], device=self.device
        )
        self._tof_pos_b = torch.tensor([t[1] for t in self.cfg.tof_sensors], device=self.device)
        self._tof_dir_b = torch.tensor([t[2] for t in self.cfg.tof_sensors], device=self.device)
        self._tof_dir_b = self._tof_dir_b / torch.linalg.norm(self._tof_dir_b, dim=-1, keepdim=True)
        assert self.cfg.observation_space == 16 + len(self.cfg.tof_sensors), \
            "observation_space must be 16 + number of ToF sensors"
        self._half_xy = torch.tensor(
            [self.cfg.arena_size_xy[0] / 2.0, self.cfg.arena_size_xy[1] / 2.0],
            device=self.device,
        )

        inertia = self.drone.root_physx_view.get_inertias()[0, self.body_id[0]]
        print(f"Drone mass: {self.mass:.4f} kg  | body inertia diag: "
              f"{inertia.view(3, 3).diagonal().tolist()}")
        print(f"Hover thrust: {self.hover_thrust:.3f} N | max motor thrust: "
              f"{self.max_motor_thrust:.3f} N | hover cmd: {self.hover_motor:.3f}")
        print("Joint names:", self.drone.joint_names)
        print(f"Arena (net) inner size: {self.cfg.arena_size_xy} x {self.cfg.arena_height} m, "
              f"margin {self.cfg.net_margin} m")
        self._prop_xforms = None

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.drone = Articulation(self.cfg.robot)
        self.scene.articulations["drone"] = self.drone

        # One IMU sensor over all envs, registered with the scene so it is
        # actually updated every step (the old per-env sensors never were, so
        # the accel obs was always zero).
        self.imu = Imu(self.cfg.imu)
        self.scene.sensors["imu"] = self.imu

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        # Drones in different envs must not collide with each other.
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        light_cfg.func("/World/Light", light_cfg)

    # ---------------------------------------------------------------- actions
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = torch.clamp(actions.clone(), 0.0, 1.0)
        f = self.actions * self.max_motor_thrust
        f1, f2, f3, f4 = f[:, 0], f[:, 1], f[:, 2], f[:, 3]

        self.force_b.zero_()
        self.torque_b.zero_()
        self.force_b[:, 0, 2] = f.sum(dim=-1)
        self.torque_b[:, 0, 0] = self.cfg.arm_length * ((f3 + f4) - (f1 + f2))
        self.torque_b[:, 0, 1] = self.cfg.arm_length * ((f2 + f3) - (f1 + f4))
        self.torque_b[:, 0, 2] = self.cfg.yaw_coeff * ((f1 + f3) - (f2 + f4))

    def _apply_action(self):
        self.drone.set_external_force_and_torque(
            self.force_b, self.torque_b, body_ids=self.body_id
        )
        if self.sim.has_gui():
            self._spin_props_visual()

    def _spin_props_visual(self):
        if self._prop_xforms is None:
            stage = self.sim.stage
            self._prop_xforms = []
            for rel in self.cfg.prop_visual_paths:
                prim = stage.GetPrimAtPath(f"/World/envs/env_0/Drone/{rel}")
                if not prim.IsValid():
                    print(f"[viz] prop prim not found ({rel}); prop spin disabled")
                    self._prop_xforms = []
                    break
                self._prop_xforms.append(UsdGeom.XformCommonAPI(prim))
            self._prop_angles = [0.0] * len(self._prop_xforms)
        for i, xform in enumerate(self._prop_xforms):
            if self.actions[0, i] > 0.05:
                speed = self.cfg.max_prop_visual_speed * (-1.0 if i in (1, 3) else 1.0)
                self._prop_angles[i] += speed * self.cfg.sim.dt
            xform.SetRotate(
                (0.0, 0.0, math.degrees(self._prop_angles[i])),
                UsdGeom.XformCommonAPI.RotationOrderXYZ,
            )

    # ---------------------------------------------------------------- helpers
    def _local_pos(self) -> torch.Tensor:
        """Position relative to this env's arena centre (env origin)."""
        return self.drone.data.root_pos_w - self.scene.env_origins

    def _dist_to_net(self, pos_local: torch.Tensor) -> torch.Tensor:
        """Distance from the drone to the nearest net face (walls, and ceiling if any)."""
        d_xy = self._half_xy - pos_local[:, 0:2].abs()          # (N, 2)
        d = d_xy.min(dim=-1).values
        if self.cfg.arena_height > 0:
            d = torch.minimum(d, self.cfg.arena_height - pos_local[:, 2])
        return d

    def _tof_readings(self, pos_local: torch.Tensor) -> torch.Tensor:
        """Simulated ToF measurements (m): true range + noise, clipped to the sensor's
        limits, with optional dropouts when the target is the net."""
        rng, hit_net = tof_ranges(pos_local, self.drone.data.root_quat_w, self._tof_pos_b,
                                  self._tof_dir_b, self._half_xy, self.cfg.arena_height)
        noise = torch.randn_like(rng) * (self.cfg.tof_noise_std + self.cfg.tof_noise_frac * rng.clamp(max=10.0))
        meas = rng + noise
        if self.cfg.tof_net_dropout > 0:
            drop = hit_net & (torch.rand_like(rng) < self.cfg.tof_net_dropout)
            meas = torch.where(drop, torch.full_like(meas, self.cfg.tof_max_range), meas)
        # Beyond max range a real ToF reports "no target"; encode that as max range.
        meas = torch.where(rng > self.cfg.tof_max_range, torch.full_like(meas, self.cfg.tof_max_range), meas)
        return meas.clamp(self.cfg.tof_min_range, self.cfg.tof_max_range)

    # ------------------------------------------------------------ observations
    def _get_observations(self):
        pos = self._local_pos()
        data = self.drone.data
        lin_acc_b = self.imu.data.lin_acc_b  # includes gravity, like a real IMU
        z_error = self.cfg.hover_height - pos[:, 2:3]
        xy_error = -pos[:, 0:2]               # target (0,0) minus position

        # Rotation rates as a real gyro reports them: in the drone's own (body) axes.
        gyro_b = self.imu.data.ang_vel_b
        tof = self._tof_readings(pos) / self.cfg.tof_max_range

        obs = torch.cat(
            [
                lin_acc_b,
                z_error,
                data.root_quat_w,
                data.root_lin_vel_w,
                gyro_b,
                xy_error,
                tof,
            ],
            dim=-1,
        )
        return {"policy": obs}

    # ----------------------------------------------------------------- reward
    def _get_rewards(self):
        pos = self._local_pos()
        data = self.drone.data
        err = pos - self._target_local

        # Soft "don't go near the net" term: 0 far away, ramps to 1 at the hard margin.
        dist = self._dist_to_net(pos)
        net_pen = torch.clamp(
            (self.cfg.net_margin + self.cfg.net_warn_band - dist) / self.cfg.net_warn_band,
            0.0,
            1.0,
        ) ** 2

        reward = (
            self.cfg.rew_alive
            - self.cfg.rew_z * err[:, 2].abs()
            - self.cfg.rew_xy * torch.linalg.norm(err[:, 0:2], dim=-1)
            - self.cfg.rew_lin_vel * torch.linalg.norm(data.root_lin_vel_w, dim=-1)
            - self.cfg.rew_ang_vel * torch.linalg.norm(data.root_ang_vel_b, dim=-1)
            - self.cfg.rew_action * torch.linalg.norm(self.actions - self.hover_motor, dim=-1)
            - self.cfg.rew_net * net_pen
            # reset_terminated is filled by _get_dones(), which DirectRLEnv calls first.
            - self.cfg.rew_crash * self.reset_terminated.float()
        )
        return reward

    # ------------------------------------------------------------------ dones
    def _get_dones(self):
        pos = self._local_pos()
        crashed = pos[:, 2] < self.cfg.crash_height
        hit_net = self._dist_to_net(pos) < self.cfg.net_margin
        # projected_gravity_b z is -1 when level, +1 upside down.
        tilt = torch.acos(torch.clamp(-self.drone.data.projected_gravity_b[:, 2], -1.0, 1.0))
        flipped = tilt > self.cfg.max_tilt_rad
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        # Kept for logging/evaluation (why did each episode end?).
        self.term_crashed, self.term_net, self.term_flipped = crashed, hit_net, flipped
        return crashed | hit_net | flipped, timed_out

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.drone._ALL_INDICES
        super()._reset_idx(env_ids)

        n = len(env_ids)
        dev = self.device
        root_state = self.drone.data.default_root_state[env_ids].clone()

        def u(lo, hi, *shape):
            return torch.empty(*shape, device=dev).uniform_(lo, hi)

        r = self.cfg.spawn_xy_range
        local = torch.stack(
            [u(-r, r, n), u(-r, r, n), u(*self.cfg.spawn_z_range, n)], dim=-1
        )
        # Spawn relative to each env's own origin (previously every drone was
        # placed at world (0, 0, 0.5), i.e. all stacked on top of each other).
        root_state[:, 0:3] = self.scene.env_origins[env_ids] + local
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
        root_state[:, 7:10] = u(-self.cfg.spawn_lin_vel_range, self.cfg.spawn_lin_vel_range, n, 3)
        root_state[:, 10:13] = u(-self.cfg.spawn_ang_vel_range, self.cfg.spawn_ang_vel_range, n, 3)

        self.drone.write_root_pose_to_sim(root_state[:, 0:7], env_ids)
        self.drone.write_root_velocity_to_sim(root_state[:, 7:13], env_ids)
        self.drone.reset(env_ids)
        self.actions[env_ids] = self.hover_motor if hasattr(self, "hover_motor") else 0.0
