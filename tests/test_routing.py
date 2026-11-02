from src.routing import DynamicRouter


def test_router_reset_restores_cooldown_state():
    router = DynamicRouter(reroute_interval=15, cooldown_period=30)
    router.last_update_step = 90
    router.vehicle_last_reroute["vehicle-1"] = 75
    router.total_reroutes = 4

    router.reset()

    assert router.last_update_step == -15
    assert router.vehicle_last_reroute == {}
    assert router.total_reroutes == 0


def test_router_does_not_update_before_interval_without_traci():
    router = DynamicRouter(reroute_interval=15)
    assert router.step_reroute(0) == 0
    assert router.step_reroute(10) == 0
