"""Multi-Agent Gymnasium Environment for Neighborhood-Scale Traffic Signal Control.

Manages N communicating intersection controllers over a single shared SUMO/TraCI instance.
Supports:
- Topology graph discovery (road network adjacency).
- Multi-agent observation space (local lane states + 1-hop neighbor metrics).
- Synchronous phase transitions with independent yellow duration enforcement.
- Cooperative shared reward formulation: R_i = alpha * R_local + (1 - alpha) * R_neighbor.
"""

import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Ensure SUMO_HOME and tools are in system path
sumo_candidates = [
    r"C:\Program Files (x86)\SUMO",
    r"C:\Program Files\SUMO",
    os.environ.get("SUMO_HOME", ""),
    r"C:\Program Files (x86)\Eclipse\Sumo"
]
for candidate in sumo_candidates:
    if candidate and os.path.exists(candidate):
        os.environ["SUMO_HOME"] = candidate
        tools_path = os.path.join(candidate, "tools")
        if tools_path not in sys.path:
            sys.path.append(tools_path)
        break

try:
    import traci
    import sumolib
except ImportError:
    traci = None
    sumolib = None
    print("[Warning] TraCI/sumolib could not be loaded. MultiIntersectionEnv will run in mock mode.")


class MultiIntersectionEnv(gym.Env):
    """Multi-Agent Gymnasium Environment for Coordinated Traffic Light Control.
    
    Exposes a dictionary-based multi-agent interface:
      step(action_dict) -> (obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        net_file: str,
        route_file: str,
        sumo_cfg_file: str,
        tls_ids: list = None,
        use_gui: bool = False,
        delta_time: int = 5,
        yellow_duration: int = 4,
        max_steps: int = 3600,
        alpha: float = 0.5,           # Reward cooperation factor (1.0 = purely local, 0.0 = purely global)
        max_neighbors: int = 4,       # Standard 4-way grid connectivity
        seed: int = 42
    ):
        super().__init__()
        
        self.net_file = net_file
        self.route_file = route_file
        self.sumo_cfg_file = sumo_cfg_file
        self.use_gui = use_gui
        self.delta_time = delta_time
        self.yellow_duration = yellow_duration
        self.max_steps = max_steps
        self.alpha = float(alpha)
        self.max_neighbors = max_neighbors
        self.seed_val = seed
        
        self.is_closed = True
        self.step_counter = 0
        
        # Resolve SUMO binary
        if sumolib is not None:
            self.sumo_binary = sumolib.checkBinary("sumo-gui") if self.use_gui else sumolib.checkBinary("sumo")
        else:
            self.sumo_binary = "sumo-gui" if self.use_gui else "sumo"

        # Discover or assign TLS agents
        self.tls_ids = tls_ids or []
        self.neighbors = {}            # {tls_id: [neighbor_tls_ids]}
        self.controlled_lanes = {}     # {tls_id: [incoming_lanes]}
        self.green_phases = {}         # {tls_id: [green_phase_indices]}
        self.yellow_phases = {}        # {tls_id: {green_idx: yellow_idx}}
        self.current_phases = {}       # {tls_id: current_phase_index}
        
        # Build network topology graph if SUMO network exists
        self._build_topology()
        
        # Define uniform action and observation spaces per agent
        self.num_actions = 2  # Standard NS-Green (0) and EW-Green (1)
        self.action_spaces = {tid: spaces.Discrete(self.num_actions) for tid in self.tls_ids}
        
        # Observation dimension per agent:
        # Local: 8 incoming lanes * 3 features (queue, density, speed) + 2 (current phase one-hot) = 26
        # Neighbor: 4 neighbors * 3 features (mean queue, mean density, active phase) = 12
        # Total per agent = 38 features
        self.local_obs_dim = 8 * 3 + self.num_actions
        self.neighbor_obs_dim = self.max_neighbors * 3
        self.obs_dim = self.local_obs_dim + self.neighbor_obs_dim
        
        self.observation_spaces = {
            tid: spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
            for tid in self.tls_ids
        }
        
        # Fallback spaces for standard gym compatibility
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

    def _build_topology(self):
        """Analyzes the SUMO network to discover road graph connections between traffic lights."""
        if sumolib is None or not os.path.exists(self.net_file):
            if not self.tls_ids:
                self.tls_ids = ["TLS_0", "TLS_1"]
            self.neighbors = {tid: [] for tid in self.tls_ids}
            return

        try:
            net = sumolib.net.readNet(self.net_file)
            all_tls = [tls.getID() for tls in net.getTrafficLights()]
            if not self.tls_ids:
                self.tls_ids = sorted(all_tls)
                
            self.neighbors = {tid: [] for tid in self.tls_ids}
            tls_set = set(self.tls_ids)
            
            # Map node connections via direct edges
            for edge in net.getEdges():
                if edge.isSpecial():
                    continue
                from_node = edge.getFromNode().getID()
                to_node = edge.getToNode().getID()
                if from_node in tls_set and to_node in tls_set and from_node != to_node:
                    if to_node not in self.neighbors[from_node]:
                        self.neighbors[from_node].append(to_node)
                    if from_node not in self.neighbors[to_node]:
                        self.neighbors[to_node].append(from_node)
                        
            print(f"[MultiIntersectionEnv] Network topology mapped: {len(self.tls_ids)} intersections.")
        except Exception as e:
            print(f"[Warning] Failed to build topology graph: {e}")
            if not self.tls_ids:
                self.tls_ids = ["A0", "A1"]
            self.neighbors = {tid: [] for tid in self.tls_ids}

    def _start_sumo(self):
        """Initializes the SUMO simulation and queries traffic light logics."""
        if traci is None:
            return

        sumo_cmd = [
            self.sumo_binary,
            "-c", self.sumo_cfg_file,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1",
            "--no-warnings", "true"
        ]
        
        traci.start(sumo_cmd)
        self.is_closed = False
        
        # Verify discovered traffic lights
        active_tls = traci.trafficlight.getIDList()
        if not self.tls_ids:
            self.tls_ids = sorted(list(active_tls))
            
        for tid in self.tls_ids:
            if tid in active_tls:
                # Discover controlled incoming lanes
                all_lanes = traci.trafficlight.getControlledLanes(tid)
                unique_lanes = sorted(list(set(all_lanes)))
                self.controlled_lanes[tid] = unique_lanes
                
                # Discover green and yellow phases from program logic
                logics = traci.trafficlight.getAllProgramLogics(tid)
                if logics:
                    phases = logics[0].phases
                    green_indices = []
                    yellow_map = {}
                    for i, p in enumerate(phases):
                        state = p.state
                        if ("G" in state or "g" in state) and ("y" not in state):
                            green_indices.append(i)
                            # The yellow phase is typically the next phase
                            yellow_idx = (i + 1) % len(phases)
                            yellow_map[len(green_indices) - 1] = yellow_idx
                            
                    # Fallback if detection didn't find at least 2 green phases
                    if len(green_indices) < 2:
                        green_indices = [0, 2] if len(phases) > 2 else [0]
                        yellow_map = {0: 1, 1: 3} if len(phases) > 3 else {0: 0}
                        
                    self.green_phases[tid] = green_indices[:self.num_actions]
                    self.yellow_phases[tid] = yellow_map
                else:
                    self.green_phases[tid] = [0, 2]
                    self.yellow_phases[tid] = {0: 1, 1: 3}
                    
                self.current_phases[tid] = 0

    def _get_agent_local_obs(self, tid: str):
        """Extracts normalized queue, density, speed, and phase vector for a single intersection."""
        if traci is None or self.is_closed or tid not in self.controlled_lanes:
            return np.zeros(self.local_obs_dim, dtype=np.float32)

        lane_features = []
        lanes = self.controlled_lanes[tid][:8]  # Bound to max 8 approach lanes for fixed dimension
        
        for lane in lanes:
            try:
                lane_len = traci.lane.getLength(lane)
                capacity = max(1.0, lane_len / 7.5)
                speed_limit = max(0.1, traci.lane.getMaxSpeed(lane))
                
                halting = traci.lane.getLastStepHaltingNumber(lane)
                total_veh = traci.lane.getLastStepVehicleNumber(lane)
                mean_speed = traci.lane.getLastStepMeanSpeed(lane)
                
                norm_queue = min(1.0, halting / capacity)
                norm_density = min(1.0, total_veh / capacity)
                norm_speed = min(1.0, mean_speed / speed_limit)
                
                lane_features.extend([norm_queue, norm_density, norm_speed])
            except Exception:
                lane_features.extend([0.0, 0.0, 0.0])
                
        # Pad with zeros if intersection has fewer than 8 incoming lanes
        while len(lane_features) < 8 * 3:
            lane_features.extend([0.0, 0.0, 0.0])
            
        # One-hot encoding of current active green phase
        current_p = self.current_phases.get(tid, 0)
        phase_one_hot = [1.0 if current_p == a else 0.0 for a in range(self.num_actions)]
        
        return np.array(lane_features + phase_one_hot, dtype=np.float32)

    def _get_agent_neighbor_obs(self, tid: str):
        """Gathers summary state features from 1-hop neighbor intersections."""
        if traci is None or self.is_closed:
            return np.zeros(self.neighbor_obs_dim, dtype=np.float32)

        neighbor_features = []
        nbr_list = self.neighbors.get(tid, [])[:self.max_neighbors]
        
        for nbr_id in nbr_list:
            if nbr_id in self.controlled_lanes:
                lanes = self.controlled_lanes[nbr_id]
                try:
                    queues = [traci.lane.getLastStepHaltingNumber(l) for l in lanes]
                    densities = [traci.lane.getLastStepVehicleNumber(l) for l in lanes]
                    mean_q = min(1.0, (sum(queues) / max(1, len(lanes))) / 10.0)
                    mean_d = min(1.0, (sum(densities) / max(1, len(lanes))) / 10.0)
                    active_p = float(self.current_phases.get(nbr_id, 0)) / max(1, self.num_actions - 1)
                    neighbor_features.extend([mean_q, mean_d, active_p])
                except Exception:
                    neighbor_features.extend([0.0, 0.0, 0.0])
            else:
                neighbor_features.extend([0.0, 0.0, 0.0])
                
        # Zero-pad for boundary intersections with fewer neighbors
        while len(neighbor_features) < self.max_neighbors * 3:
            neighbor_features.extend([0.0, 0.0, 0.0])
            
        return np.array(neighbor_features, dtype=np.float32)

    def _get_all_observations(self):
        """Constructs full observation dictionary for all intersection agents."""
        obs_dict = {}
        for tid in self.tls_ids:
            local = self._get_agent_local_obs(tid)
            nbr = self._get_agent_neighbor_obs(tid)
            obs_dict[tid] = np.concatenate([local, nbr]).astype(np.float32)
        return obs_dict

    def _get_local_reward(self, tid: str):
        """Computes local penalty based on queues and cumulative delays."""
        if traci is None or self.is_closed or tid not in self.controlled_lanes:
            return 0.0
            
        total_queue = 0
        total_wait = 0
        for lane in self.controlled_lanes[tid]:
            try:
                total_queue += traci.lane.getLastStepHaltingNumber(lane)
                total_wait += traci.lane.getWaitingTime(lane)
            except Exception:
                pass
                
        # Local penalty formula
        return -float(total_queue + 0.05 * total_wait)

    def _get_cooperative_rewards(self):
        """Computes shared cooperative rewards for each agent using alpha mixing."""
        local_rewards = {tid: self._get_local_reward(tid) for tid in self.tls_ids}
        rewards = {}
        
        for tid in self.tls_ids:
            nbr_list = self.neighbors.get(tid, [])
            if nbr_list:
                nbr_reward = sum([local_rewards.get(nbr, 0.0) for nbr in nbr_list]) / len(nbr_list)
            else:
                nbr_reward = local_rewards[tid]
                
            # Mixed reward: alpha * local + (1 - alpha) * neighbor_mean
            rewards[tid] = self.alpha * local_rewards[tid] + (1.0 - self.alpha) * nbr_reward
            
        return rewards

    def step(self, actions: dict):
        """Advances the simulation by delta_time with coordinated yellow transitions."""
        self.step_counter += self.delta_time
        
        if traci is None or self.is_closed:
            obs = {tid: np.zeros(self.obs_dim, dtype=np.float32) for tid in self.tls_ids}
            rewards = {tid: 0.0 for tid in self.tls_ids}
            terminated = self.step_counter >= self.max_steps
            return obs, rewards, {tid: terminated for tid in self.tls_ids}, {tid: False for tid in self.tls_ids}, {}

        # 1. Identify agents switching phases and execute Yellow transitions
        switching_tls = {}
        for tid, action in actions.items():
            if tid in self.green_phases:
                target_green = self.green_phases[tid][action]
                current_green = traci.trafficlight.getPhase(tid)
                if current_green != target_green:
                    yellow_phase = self.yellow_phases[tid].get(action, current_green + 1)
                    traci.trafficlight.setPhase(tid, yellow_phase)
                    switching_tls[tid] = target_green
                    
        # 2. Advance Yellow simulation steps (4 seconds)
        if switching_tls:
            for _ in range(self.yellow_duration):
                traci.simulationStep()
                
            # 3. Transition switching agents to their target Green phases
            for tid, target_green in switching_tls.items():
                traci.trafficlight.setPhase(tid, target_green)
                self.current_phases[tid] = actions[tid]
                
        # 4. Advance remaining delta_time seconds
        remaining_steps = max(1, self.delta_time - (self.yellow_duration if switching_tls else 0))
        for _ in range(remaining_steps):
            traci.simulationStep()

        # Update current phases for non-switching agents
        for tid, action in actions.items():
            self.current_phases[tid] = action

        # 5. Retrieve post-step observations, rewards, and simulation metrics
        obs_dict = self._get_all_observations()
        reward_dict = self._get_cooperative_rewards()
        terminated = self.step_counter >= self.max_steps
        terminated_dict = {tid: terminated for tid in self.tls_ids}
        truncated_dict = {tid: False for tid in self.tls_ids}
        
        # System-level metrics in info dictionary
        total_network_queue = sum([
            sum([traci.lane.getLastStepHaltingNumber(l) for l in self.controlled_lanes.get(tid, [])])
            for tid in self.tls_ids
        ])
        
        info_dict = {
            "step": self.step_counter,
            "total_network_queue": total_network_queue,
            "arrived_vehicles": traci.simulation.getArrivedNumber()
        }
        
        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

    def reset(self, seed=None, options=None):
        """Restarts the neighborhood simulation for a new episode."""
        super().reset(seed=seed)
        self.step_counter = 0
        self.current_phases = {tid: 0 for tid in self.tls_ids}
        
        self.close()
        self._start_sumo()
        
        obs_dict = self._get_all_observations()
        info_dict = {"step": 0, "n_agents": len(self.tls_ids)}
        
        return obs_dict, info_dict

    def close(self):
        """Safely shuts down TraCI connection."""
        if not self.is_closed and traci is not None:
            try:
                traci.close()
            except Exception:
                pass
            self.is_closed = True

    def render(self):
        pass

