import torch
import torch.nn as nn


class RLHoverActor(nn.Module):
    def __init__(self, obs_dim=14, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
            nn.Sigmoid(),
        )

    def forward(self, obs):
        return self.net(obs)


RLHoverPolicy = RLHoverActor


class RLHoverController:
    def __init__(self, device="cuda:0", checkpoint=None):
        self.device = device
        # Defaults match the old Crazyflie setup; new checkpoints carry their own.
        self.hover_motor = 1.0 / 2.2
        self.residual_scale = 0.15
        obs_dim = 16
        state = None

        if checkpoint is not None:
            state = torch.load(checkpoint, map_location=device)
            if "actor" in state:
                self.hover_motor = state.get("hover_motor", self.hover_motor)
                self.residual_scale = state.get("residual_scale", self.residual_scale)
                obs_dim = state.get("obs_dim", 14)
                state = state["actor"]

        self.obs_dim = obs_dim
        self.policy = RLHoverActor(obs_dim=obs_dim).to(device)
        self.policy.eval()
        if state is not None:
            self.policy.load_state_dict(state)

    def reset(self, num_envs):
        pass

    def act(self, obs):
        with torch.no_grad():
            residual_raw = self.policy(obs[:, : self.obs_dim])

        residual = (residual_raw - 0.5) * self.residual_scale
        actions = self.hover_motor + residual

        return torch.clamp(actions, 0.0, 1.0)