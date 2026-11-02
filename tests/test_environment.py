import shutil

import pytest

from src.multi_env import MultiIntersectionEnv


@pytest.mark.skipif(shutil.which("sumo") is None, reason="SUMO is required for integration test")
def test_environment_reset_and_step():
    environment = MultiIntersectionEnv(
        net_file="data/neighborhood.net.xml",
        route_file="data/neighborhood.rou.xml",
        sumo_cfg_file="data/neighborhood.sumocfg",
        max_steps=10,
        delta_time=5,
    )
    try:
        observations, info = environment.reset(seed=11)
        assert info["n_agents"] == 16
        assert len(observations) == 16
        assert all(observation.shape == (38,) for observation in observations.values())

        actions = {agent_id: 0 for agent_id in observations}
        next_observations, rewards, terminated, truncated, step_info = environment.step(actions)
        assert set(next_observations) == set(observations)
        assert set(rewards) == set(observations)
        assert step_info["step"] == 5
        assert all(value is False for value in truncated.values())
        assert all(value is False for value in terminated.values())
    finally:
        environment.close()
