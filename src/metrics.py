"""Quantitative Metrics and Statistical Analysis Engine for Traffic Signal Research.

Computes:
1. Average Vehicle Travel Time and Waiting Time (from tripinfo XML).
2. Time Saved per Vehicle and Total Network Hours Saved vs. Baselines.
3. System Throughput (vehicles completed per hour).
4. Environmental Abatement (CO2 emissions in kg and fuel in liters).
5. Green Wave Progression Score (% non-stop corridor vehicles).
6. Jain's Fairness Index across all neighborhood intersections.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


class TrafficMetricsCalculator:
    """Computes academic-grade ITS performance indicators from simulation outputs."""

    @staticmethod
    def parse_tripinfo(tripinfo_xml_path: str):
        """Extracts individual vehicle trip metrics from SUMO tripinfo output."""
        if not os.path.exists(tripinfo_xml_path):
            return {
                "arrived": 0,
                "mean_duration": 0.0,
                "mean_waiting_time": 0.0,
                "mean_time_loss": 0.0,
                "total_co2_kg": 0.0,
                "total_fuel_liters": 0.0
            }

        tree = ET.parse(tripinfo_xml_path)
        root = tree.getroot()

        durations = []
        waiting_times = []
        time_losses = []
        co2_mg_list = []
        fuel_ml_list = []

        for trip in root.findall("tripinfo"):
            durations.append(float(trip.get("duration", 0.0)))
            waiting_times.append(float(trip.get("waitingTime", 0.0)))
            time_losses.append(float(trip.get("timeLoss", 0.0)))

            # Check emissions subelement if present
            emissions = trip.find("emissions")
            if emissions is not None:
                co2_mg_list.append(float(emissions.get("CO2_abs", 0.0)))
                fuel_ml_list.append(float(emissions.get("fuel_abs", 0.0)))

        arrived = len(durations)
        return {
            "arrived": arrived,
            "mean_duration": float(np.mean(durations)) if arrived > 0 else 0.0,
            "mean_waiting_time": float(np.mean(waiting_times)) if arrived > 0 else 0.0,
            "mean_time_loss": float(np.mean(time_losses)) if arrived > 0 else 0.0,
            "total_co2_kg": float(np.sum(co2_mg_list) / 1e6) if co2_mg_list else 0.0,
            "total_fuel_liters": float(np.sum(fuel_ml_list) / 1e3) if fuel_ml_list else 0.0
        }

    @staticmethod
    def calculate_time_saved(baseline_metrics: dict, proposed_metrics: dict):
        """Calculates travel time and delay savings per vehicle and network-wide."""
        w_base = baseline_metrics.get("mean_waiting_time", 0.0)
        w_prop = proposed_metrics.get("mean_waiting_time", 0.0)
        
        delta_wait_per_veh = max(0.0, w_base - w_prop)
        arrived = proposed_metrics.get("arrived", 0)
        
        # Total person-hours saved across the simulation period
        total_hours_saved = (delta_wait_per_veh * arrived) / 3600.0
        
        # Percentage efficiency improvement
        efficiency_gain_pct = ((w_base - w_prop) / max(0.1, w_base)) * 100.0
        
        return {
            "time_saved_per_veh_sec": delta_wait_per_veh,
            "total_network_hours_saved": total_hours_saved,
            "efficiency_gain_pct": efficiency_gain_pct
        }

    @staticmethod
    def calculate_throughput(arrived_count: int, duration_sec: float = 3600.0):
        """Calculates completed vehicles per hour."""
        return (arrived_count / max(1.0, duration_sec)) * 3600.0

    @staticmethod
    def calculate_jains_fairness_index(delays_per_intersection: list):
        """Calculates Jain's Fairness Index: F = (sum x_i)^2 / (N * sum x_i^2)."""
        x = np.array(delays_per_intersection, dtype=np.float64)
        if len(x) == 0 or np.sum(x**2) == 0:
            return 1.0
        return float((np.sum(x) ** 2) / (len(x) * np.sum(x ** 2)))

    @staticmethod
    def calculate_green_wave_score(tripinfo_xml_path: str, zero_stop_threshold: float = 1.0):
        """Calculates percentage of vehicles that completed trips with near-zero waiting time."""
        if not os.path.exists(tripinfo_xml_path):
            return 0.0
        tree = ET.parse(tripinfo_xml_path)
        root = tree.getroot()
        trips = root.findall("tripinfo")
        if not trips:
            return 0.0
        non_stop = sum(1 for t in trips if float(t.get("waitingTime", 0.0)) <= zero_stop_threshold)
        return (non_stop / len(trips)) * 100.0

