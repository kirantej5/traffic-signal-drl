import torch

from src.agents.gnn_policy import GNNTrafficPolicy
from src.agents.mappo_agent import MAPPOTrafficController


def dummy_observations(n_agents=4, obs_dim=38):
    return {f"TLS_{index}": torch.zeros(obs_dim).numpy() for index in range(n_agents)}


def test_mappo_action_shapes_and_checkpoint_round_trip(tmp_path):
    torch.manual_seed(7)
    controller = MAPPOTrafficController(n_agents=4, obs_dim=38, action_dim=2, device="cpu")
    observations = dummy_observations()
    actions, log_probs, value, global_state = controller.select_actions(observations, deterministic=True)

    assert set(actions) == set(observations)
    assert all(action in (0, 1) for action in actions.values())
    assert set(log_probs) == set(observations)
    assert isinstance(value, float)
    assert global_state.shape == (152,)

    checkpoint = tmp_path / "controller.pt"
    controller.save(str(checkpoint))
    restored = MAPPOTrafficController(n_agents=4, obs_dim=38, action_dim=2, device="cpu")
    restored.load(str(checkpoint))
    restored_actions, _, _, _ = restored.select_actions(observations, deterministic=True)
    assert restored_actions == actions


def test_gnn_policy_output_shapes():
    torch.manual_seed(7)
    policy = GNNTrafficPolicy(node_feature_dim=26, hidden_dim=32, action_dim=2)
    logits, values = policy(torch.randn(4, 26), torch.eye(4))
    assert logits.shape == (4, 2)
    assert values.shape == (4,)
