"""Graph Neural Network (GNN) Policy Head for Topology-Aware Traffic Signal Control.

Models the neighborhood intersections as a graph G = (V, E):
- Nodes V: Traffic light intersections.
- Edges E: Arterial roads connecting adjacent intersections.
- Spatial Multi-Head Graph Attention: Intersections learn dynamic attention weights
  over neighbors, allowing green waves to naturally form along high-density corridors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class GraphAttentionLayer(nn.Module):
    """Single-head Graph Attention Layer (Veličković et al., ICLR 2018)."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h: [N, in_features], adj: [N, N] adjacency matrix
        Wh = self.W(h)  # [N, out_features]
        N = Wh.size(0)

        # Broadcast addition for pairwise attention
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), Wh.repeat(N, 1)], dim=1).view(N, N, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)


class GNNTrafficPolicy(nn.Module):
    """Graph Attention Network that coordinates multi-intersection phase control."""
    def __init__(self, node_feature_dim: int = 26, hidden_dim: int = 64, action_dim: int = 2):
        super().__init__()
        self.gat1 = GraphAttentionLayer(node_feature_dim, hidden_dim)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim)
        
        # Policy head per node
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value head (for Centralized GNN Critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_features, adj_matrix):
        """
        node_features: [N, node_feature_dim]
        adj_matrix: [N, N] binary adjacency matrix with self-loops
        """
        x = self.gat1(node_features, adj_matrix)
        x = self.gat2(x, adj_matrix)
        
        logits = self.policy_head(x)  # [N, action_dim]
        values = self.value_head(x).squeeze(-1)  # [N]
        
        return logits, values

