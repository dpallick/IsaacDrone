import torch
import torch.nn as nn

class RLHoverPolicy(nn.Module):
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

class RLHoverController:
    def __init__(self, device="cuda:0", checkpoint=None):
        self.device = device
        self.hover_motor = 1.0 / 2.2
        self.residual_scale = 0.3
        self.policy = RLHoverPolicy().to(device)
        self.policy.eval()
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location=device)
            self.policy.load_state_dict(state)

    def reset(self, num_envs):
        pass

    def act(self, obs):
        with torch.no_grad():
            residual_raw = self.policy(obs)
            residual = (residual_raw - 0.5) * self.residual_scale
            actions = self.hover_motor + residual
            return torch.clamp(actions, 0.0, 1.0)