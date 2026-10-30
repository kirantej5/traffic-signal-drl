"""Multi-Agent PPO (MAPPO) with Centralized Critic for Cooperative Traffic Signal Control.

Implements Centralized Training with Decentralized Execution (CTDE):
- Decentralized Actor: Each intersection agent selects green phase actions using its
  local + 1-hop neighbor observations.
- Centralized Critic: The value function estimates the expected global return from
  the concatenated observations of all N neighborhood intersections, resolving the
  non-stationarity dilemma in multi-agent reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class ActorNetwork(nn.Module):
    """Decentralized policy network for an individual intersection agent."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs):
        logits = self.net(obs)
        return Categorical(logits=logits)


class CentralizedCriticNetwork(nn.Module):
    """Centralized value network that assesses global neighborhood traffic states."""
    def __init__(self, global_state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, global_state):
        return self.net(global_state).squeeze(-1)


class MAPPOTrafficController:
    """Coordinates Multi-Agent PPO policy updates across all neighborhood intersection agents."""
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        device: str = "cpu"
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.global_state_dim = obs_dim * n_agents
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.device = torch.device(device)

        # Networks
        self.actor = ActorNetwork(obs_dim, action_dim).to(self.device)
        self.critic = CentralizedCriticNetwork(self.global_state_dim).to(self.device)

        # Optimizers
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_actions(self, obs_dict: dict, deterministic: bool = False):
        """Samples phase actions for all agents given their local+neighbor observations."""
        actions = {}
        log_probs = {}
        
        # Build global state tensor
        ordered_tids = sorted(list(obs_dict.keys()))
        obs_list = [obs_dict[tid] for tid in ordered_tids]
        global_state = np.concatenate(obs_list).astype(np.float32)
        global_state_t = torch.tensor(global_state, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            state_value = self.critic(global_state_t).item()

            for tid in ordered_tids:
                obs_t = torch.tensor(obs_dict[tid], dtype=torch.float32, device=self.device)
                dist = self.actor(obs_t)
                
                if deterministic:
                    action = torch.argmax(dist.probs).item()
                else:
                    action = dist.sample().item()
                    
                actions[tid] = action
                log_probs[tid] = dist.log_prob(torch.tensor(action, device=self.device)).item()

        return actions, log_probs, state_value, global_state

    def update(self, trajectory_buffer: list, epochs: int = 4, batch_size: int = 64):
        """Performs PPO actor and centralized critic updates using collected trajectory rollouts."""
        if not trajectory_buffer:
            return {"actor_loss": 0.0, "critic_loss": 0.0}

        # Unpack trajectories
        # Each entry: (obs_dict, actions_dict, log_probs_dict, rewards_dict, global_state, value, done)
        T = len(trajectory_buffer)
        
        # 1. Compute Generalized Advantage Estimations (GAE)
        values = [t["value"] for t in trajectory_buffer] + [0.0]
        rewards = [sum(t["rewards"].values()) / self.n_agents for t in trajectory_buffer]
        dones = [t["done"] for t in trajectory_buffer]

        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
            
        returns = advantages + np.array(values[:-1])
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Prepare PyTorch Tensors
        global_states_t = torch.tensor(np.array([t["global_state"] for t in trajectory_buffer]), dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        # Flat agent observations and actions
        all_obs = []
        all_actions = []
        all_old_log_probs = []
        all_adv_expanded = []

        for t_idx, step_data in enumerate(trajectory_buffer):
            adv_val = advantages_t[t_idx]
            for tid, action in step_data["actions"].items():
                all_obs.append(step_data["obs"][tid])
                all_actions.append(action)
                all_old_log_probs.append(step_data["log_probs"][tid])
                all_adv_expanded.append(adv_val)

        all_obs_t = torch.tensor(np.array(all_obs), dtype=torch.float32, device=self.device)
        all_actions_t = torch.tensor(np.array(all_actions), dtype=torch.int64, device=self.device)
        all_old_log_probs_t = torch.tensor(np.array(all_old_log_probs), dtype=torch.float32, device=self.device)
        all_adv_expanded_t = torch.stack(all_adv_expanded).to(self.device)

        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for _ in range(epochs):
            # Critic Update
            current_values = self.critic(global_states_t)
            critic_loss = nn.MSELoss()(current_values, returns_t)

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer_critic.step()
            total_critic_loss += critic_loss.item()

            # Actor Update
            dist = self.actor(all_obs_t)
            new_log_probs = dist.log_prob(all_actions_t)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_log_probs - all_old_log_probs_t)
            surr1 = ratios * all_adv_expanded_t
            surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * all_adv_expanded_t
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.optimizer_actor.step()
            total_actor_loss += actor_loss.item()

        return {
            "actor_loss": total_actor_loss / epochs,
            "critic_loss": total_critic_loss / epochs
        }

    def save(self, filepath: str):
        """Saves actor and critic weights."""
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        """Loads actor and critic weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])

