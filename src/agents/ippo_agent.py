"""Independent PPO (IPPO) Baseline for Decentralized Traffic Light Control.

In IPPO, each intersection agent acts independently without a centralized critic
or explicit inter-agent communication, serving as a primary baseline against MAPPO.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class IPPOAgent(nn.Module):
    """Independent Actor-Critic network for an individual traffic light."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs):
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return Categorical(logits=logits), value


class IPPOController:
    """Manages independent PPO baseline updates across intersection agents."""
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        device: str = "cpu"
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.device = torch.device(device)

        self.network = IPPOAgent(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def select_actions(self, obs_dict: dict, deterministic: bool = False):
        actions = {}
        log_probs = {}
        values = {}

        with torch.no_grad():
            for tid, obs in obs_dict.items():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
                dist, value = self.network(obs_t)
                action = torch.argmax(dist.probs).item() if deterministic else dist.sample().item()
                actions[tid] = action
                log_probs[tid] = dist.log_prob(torch.tensor(action, device=self.device)).item()
                values[tid] = value.item()

        return actions, log_probs, values

    def save(self, filepath: str):
        torch.save(self.network.state_dict(), filepath)

    def load(self, filepath: str):
        self.network.load_state_dict(torch.load(filepath, map_location=self.device))

