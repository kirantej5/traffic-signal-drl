"""Neighborhood Multi-Agent Traffic Signal Control with Dynamic Routing."""

try:
    from .multi_env import MultiIntersectionEnv
    from .routing import DynamicRouter
    from .agents.mappo_agent import MAPPOTrafficController
    from .agents.ippo_agent import IPPOController
    from .agents.gnn_policy import GNNTrafficPolicy
    from .metrics import TrafficMetricsCalculator
except ImportError:
    from multi_env import MultiIntersectionEnv
    from routing import DynamicRouter
    from agents.mappo_agent import MAPPOTrafficController
    from agents.ippo_agent import IPPOController
    from agents.gnn_policy import GNNTrafficPolicy
    from metrics import TrafficMetricsCalculator

__all__ = [
    "MultiIntersectionEnv",
    "DynamicRouter",
    "MAPPOTrafficController",
    "IPPOController",
    "GNNTrafficPolicy",
    "TrafficMetricsCalculator",
]
