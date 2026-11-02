from src.metrics import TrafficMetricsCalculator


def test_tripinfo_metrics_and_time_saved(tmp_path):
    tripinfo = tmp_path / "tripinfo.xml"
    tripinfo.write_text(
        """<root>
        <tripinfo duration="30" waitingTime="10" timeLoss="12"><emissions CO2_abs="1000000" fuel_abs="2000" /></tripinfo>
        <tripinfo duration="20" waitingTime="4" timeLoss="5"><emissions CO2_abs="500000" fuel_abs="1000" /></tripinfo>
        </root>""",
        encoding="utf-8",
    )

    metrics = TrafficMetricsCalculator.parse_tripinfo(str(tripinfo))
    saved = TrafficMetricsCalculator.calculate_time_saved(
        {"mean_waiting_time": 10.0}, {"mean_waiting_time": metrics["mean_waiting_time"], "arrived": metrics["arrived"]}
    )

    assert metrics["arrived"] == 2
    assert metrics["mean_waiting_time"] == 7.0
    assert metrics["total_co2_kg"] == 1.5
    assert metrics["total_fuel_liters"] == 3.0
    assert saved["time_saved_per_veh_sec"] == 3.0


def test_fairness_and_throughput_edge_cases():
    assert TrafficMetricsCalculator.calculate_jains_fairness_index([]) == 1.0
    assert TrafficMetricsCalculator.calculate_jains_fairness_index([10, 10]) == 1.0
    assert TrafficMetricsCalculator.calculate_throughput(100, 1800) == 200.0
