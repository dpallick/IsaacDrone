"""Sensor timing as the policy will see it on the drone (pure torch).

The policy runs at a fixed rate (50 Hz, like a ROS timer in the MAVROS node) and
always uses the LATEST message of each topic. A sensor that publishes slower than
that repeats its last value; a sensor with latency shows the world as it was a
few tens of ms ago. HeldSensor reproduces both:

    raw (true reading now) --> delay line (latency) --> sample at the sensor's
    own rate (phase clock, random start) --> hold until the next sample

Rate and latency are drawn per env per episode from ranges, so the policy can't
rely on exact timing.
"""
from __future__ import annotations

import torch


class HeldSensor:
    def __init__(self, n: int, dim: int, max_delay_steps: int, device):
        self.n, self.dim, self.L, self.device = n, dim, max_delay_steps + 1, device
        self.buf = torch.zeros(n, self.L, dim, device=device)
        self.hold = torch.zeros(n, dim, device=device)
        self.phase = torch.zeros(n, device=device)
        self.rate = torch.ones(n, device=device)
        self.lat = torch.zeros(n, dtype=torch.long, device=device)
        self.fill = torch.ones(n, dtype=torch.bool, device=device)

    def reset(self, ids, rate_hz, latency_steps):
        self.rate[ids] = rate_hz
        self.lat[ids] = latency_steps.clamp(0, self.L - 1)
        self.phase[ids] = torch.rand(len(ids) if not isinstance(ids, slice) else self.n, device=self.device)
        self.fill[ids] = True

    def __call__(self, raw: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance one policy step with this step's true reading; return what the
        policy would read from the topic now."""
        if bool(self.fill.any()):          # fresh episode: pretend it's been steady
            f = self.fill
            self.buf[f] = raw[f][:, None].expand(-1, self.L, -1)
            self.hold[f] = raw[f]
            self.fill[f] = False
        self.buf = torch.cat([self.buf[:, 1:], raw[:, None]], dim=1)
        delayed = self.buf[torch.arange(self.n, device=self.device), self.L - 1 - self.lat]
        self.phase = self.phase + self.rate * dt
        fresh = self.phase >= 1.0
        self.phase = torch.where(fresh, self.phase - 1.0, self.phase)
        self.hold = torch.where(fresh[:, None], delayed, self.hold)
        return self.hold.clone()
