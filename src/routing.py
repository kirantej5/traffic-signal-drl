"""Dynamic Vehicle Routing and Traffic Rerouting Engine for Neighborhood MARL.

Implements:
1. Continuous edge impedance monitor: Computes dynamic edge travel times based on
   real-time speed drops and intersection queue spillbacks.
2. TraCI Travel-Time Adaptation: Updates SUMO's internal routing weights using `adaptTraveltime`.
3. Anti-Oscillation Vehicle Rerouter: Selectively reroutes en-route vehicles with
   configurable cooldown periods to prevent route thrashing.
"""

import time
import numpy as np

try:
    import traci
except ImportError:
    traci = None


class DynamicRouter:
    """Manages real-time vehicle rerouting across the neighborhood road network."""

    def __init__(
        self,
        net_edges: list = None,
        reroute_interval: int = 15,    # Simulation seconds between routing updates
        cooldown_period: int = 30,     # Min seconds before a single vehicle can be rerouted again
        congestion_threshold: float = 0.5  # Re-route when speed drops below 50% of limit
    ):
        self.net_edges = net_edges or []
        self.reroute_interval = reroute_interval
        self.cooldown_period = cooldown_period
        self.congestion_threshold = congestion_threshold
        
        self.last_update_step = -reroute_interval
        self.vehicle_last_reroute = {}  # {veh_id: last_reroute_step}
        self.total_reroutes = 0

    def update_edge_travel_times(self, step: int):
        """Calculates dynamic travel time per edge and feeds it into SUMO's routing engine."""
        if traci is None:
            return {}

        edge_costs = {}
        if not self.net_edges:
            try:
                self.net_edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
            except Exception:
                return {}

        for edge_id in self.net_edges:
            try:
                # Free-flow metrics
                edge_len = max(10.0, traci.lane.getLength(f"{edge_id}_0") if f"{edge_id}_0" in traci.lane.getIDList() else 200.0)
                speed_limit = 13.89
                
                # Real-time state
                mean_speed = traci.edge.getLastStepMeanSpeed(edge_id)
                halting_count = traci.edge.getLastStepHaltingNumber(edge_id)
                
                # Dynamic travel time: Travel time + Queue Delay penalty
                effective_speed = max(1.0, mean_speed)
                travel_time = edge_len / effective_speed
                queue_delay = halting_count * 2.0  # Approx 2s delay per halted vehicle
                
                dynamic_cost = travel_time + queue_delay
                edge_costs[edge_id] = dynamic_cost
                
                # Adapt TraCI travel time model for Dijkstra calculations
                traci.edge.adaptTraveltime(edge_id, dynamic_cost)
            except Exception:
                pass

        return edge_costs

    def step_reroute(self, step: int):
        """Executes dynamic vehicle rerouting for vehicles stuck in congestion."""
        if traci is None:
            return 0

        # Run routing pass at designated interval
        if step - self.last_update_step < self.reroute_interval:
            return 0

        self.last_update_step = step
        
        # 1. Update edge travel times across all arterial roads
        self.update_edge_travel_times(step)
        
        # 2. Query running vehicles
        try:
            running_vehs = traci.vehicle.getIDList()
        except Exception:
            return 0

        rerouted_this_step = 0
        
        for veh_id in running_vehs:
            # Check cooldown to prevent routing oscillation
            last_reroute = self.vehicle_last_reroute.get(veh_id, -self.cooldown_period)
            if step - last_reroute < self.cooldown_period:
                continue

            try:
                curr_speed = traci.vehicle.getSpeed(veh_id)
                max_speed = traci.vehicle.getMaxSpeed(veh_id)
                
                # Trigger reroute if vehicle is experiencing significant delay
                if curr_speed < (max_speed * self.congestion_threshold):
                    # Call TraCI online dynamic rerouting based on updated edge travel times
                    traci.vehicle.rerouteTraveltime(veh_id, currentTravelTimes=True)
                    self.vehicle_last_reroute[veh_id] = step
                    rerouted_this_step += 1
                    self.total_reroutes += 1
            except Exception:
                pass

        # Cleanup departed vehicles from memory
        active_set = set(running_vehs)
        self.vehicle_last_reroute = {vid: s for vid, s in self.vehicle_last_reroute.items() if vid in active_set}

        return rerouted_this_step

    def reset(self):
        """Resets tracking cache for new episode."""
        self.last_update_step = -self.reroute_interval
        self.vehicle_last_reroute.clear()
        self.total_reroutes = 0

