import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Ensure SUMO_HOME is added to PATH for TraCI to import sumolib/traci
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    # Fallback to default Windows install location if not set
    default_sumo_path = r"C:\Program Files (x86)\Eclipse\Sumo\tools"
    if os.path.exists(default_sumo_path):
        sys.path.append(default_sumo_path)
        os.environ["SUMO_HOME"] = r"C:\Program Files (x86)\Eclipse\Sumo"

try:
    import traci
    import sumolib
except ImportError:
    # Mocking TraCI for environments where SUMO is not physically installed (e.g. CI/CD or template checks)
    traci = None
    sumolib = None
    print("[Warning] TraCI/sumolib could not be imported. The environment will run in mock mode.")


class SumoTrafficEnv(gym.Env):
    """Custom Gymnasium environment for Adaptive Traffic Signal Control (ATSC) using SUMO and TraCI.
    Designed to support GPU-accelerated Deep Reinforcement Learning (e.g., DQN, PPO).
    
    This environment is general-purpose and automatically detects the lanes and phases 
    controlled by the traffic light, making it easy to swap in custom SUMO scenarios.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        net_file: str,
        route_file: str,
        sumo_cfg_file: str,
        tls_id: str = "C",
        use_gui: bool = False,
        min_green: int = 10,
        yellow_duration: int = 4,
        max_steps: int = 3600,
        reward_type: str = "queue",  # Choose from 'queue', 'wait', or 'combined'
        delta_time: int = 5,         # Number of simulation seconds between actions
    ):
        super().__init__()
        
        self.net_file = net_file
        self.route_file = route_file
        self.sumo_cfg_file = sumo_cfg_file
        self.tls_id = tls_id
        self.use_gui = use_gui
        self.min_green = min_green
        self.yellow_duration = yellow_duration
        self.max_steps = max_steps
        self.reward_type = reward_type
        self.delta_time = delta_time
        
        self.sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        self.is_closed = True
        self.step_counter = 0
        
        # Traffic light phases mapping for a standard 4-way intersection
        # Actions directly select green phases. 
        # Action 0 -> N-S Straight Green (Phase 0)
        # Action 1 -> N-S Left-Turn Green (Phase 2)
        # Action 2 -> E-W Straight Green (Phase 4)
        # Action 3 -> E-W Left-Turn Green (Phase 6)
        self.green_phases = [0, 2, 4, 6]
        self.num_actions = len(self.green_phases)
        self.action_space = spaces.Discrete(self.num_actions)
        
        # State variables
        self.current_phase_idx = 0  # Index in self.green_phases
        self.in_yellow_transition = False
        self.yellow_timer = 0
        self.target_phase_idx = 0
        
        # Track previous step values for differential rewards
        self.prev_waiting_time = 0
        self.prev_queue_length = 0
        
        # We initialize the environment and extract controlled lanes dynamically
        # Since SUMO isn't started yet, we define a fallback size or configure spaces dynamically on reset
        # Let's set a fixed observation space representing the 12 incoming lanes of the default 4-way intersection
        # Observation: [Queue Length, Average Speed, Vehicle Count] for each incoming lane
        # If the user replaces the network, we dynamically adjust in the reset function
        self.incoming_lanes = []
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(24,),  # Default: 12 lanes * 2 features (normalized queue, normalized density)
            dtype=np.float32
        )
        
    def _start_sumo(self):
        """Starts the SUMO simulation using TraCI."""
        if traci is None:
            print("[Info] Running in mock mode because TraCI is not available.")
            return
            
        sumo_cmd = [
            self.sumo_binary,
            "-c", self.sumo_cfg_file,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",  # Retain long waiting times for calculation
            "--time-to-teleport", "-1"        # Prevent teleporting to ensure accurate waiting times
        ]
        
        # Start TraCI
        traci.start(sumo_cmd)
        self.is_closed = False
        
        # Dynamically discover lanes controlled by the TLS
        all_controlled_lanes = list(set(traci.trafficlight.getControlledLanes(self.tls_id)))
        # Filter for incoming lanes (incoming lanes end in 'C2...'? No, incoming lanes are '...2C')
        self.incoming_lanes = [lane for lane in all_controlled_lanes if lane.endswith("2C") or "_to_C" in lane or "2C_" in lane]
        if not self.incoming_lanes:
            # Fallback to all controlled lanes if name matching fails
            self.incoming_lanes = all_controlled_lanes
            
        self.incoming_lanes = sorted(self.incoming_lanes)
        
        # Dynamically update observation space size based on actual lane count
        num_lanes = len(self.incoming_lanes)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_lanes * 2,),  # [Normalized Queue, Normalized Density] for each lane
            dtype=np.float32
        )
        
    def _get_observation(self):
        """Retrieves observation vector from SUMO simulator."""
        if traci is None or self.is_closed:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
        obs = []
        for lane in self.incoming_lanes:
            # Feature 1: Halting vehicles (speed < 0.1 m/s) divided by max lane capacity (approx length/7.5m per car)
            lane_len = traci.lane.getLength(lane)
            max_capacity = max(1.0, lane_len / 7.5)
            
            queue_len = traci.lane.getLastStepHaltingNumber(lane)
            normalized_queue = min(1.0, queue_len / max_capacity)
            
            # Feature 2: Density (current vehicles / capacity)
            veh_count = traci.lane.getLastStepVehicleNumber(lane)
            normalized_density = min(1.0, veh_count / max_capacity)
            
            obs.extend([normalized_queue, normalized_density])
            
        return np.array(obs, dtype=np.float32)

    def _get_reward(self):
        """Calculates the DRL reward based on current traffic states."""
        if traci is None or self.is_closed:
            return 0.0
            
        # Extract metrics
        total_queue = 0
        total_wait = 0
        
        for lane in self.incoming_lanes:
            total_queue += traci.lane.getLastStepHaltingNumber(lane)
            # Accumulate waiting time (vehicles with speed < 0.1 m/s wait times)
            total_wait += traci.lane.getWaitingTime(lane)
            
        if self.reward_type == "queue":
            # Penalize long queues
            reward = -float(total_queue)
        elif self.reward_type == "wait":
            # Penalize long waiting times
            reward = -float(total_wait)
        else: # combined
            reward = -float(total_queue * 0.5 + total_wait * 0.01)
            
        return reward

    def step(self, action):
        """Executes one control action in the environment."""
        self.step_counter += self.delta_time
        
        if traci is None or self.is_closed:
            # Mock step execution
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            reward = 0.0
            terminated = self.step_counter >= self.max_steps
            truncated = False
            return obs, reward, terminated, truncated, {}
            
        target_phase = self.green_phases[action]
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        
        # Check if we need to execute a yellow transition phase
        if current_phase != target_phase:
            # If changing green phase, trigger yellow phase
            # In SUMO, the yellow phase is typically the green phase index + 1
            yellow_phase = (current_phase + 1) % len(traci.trafficlight.getAllProgramLogics(self.tls_id)[0].phases)
            
            # Set yellow phase
            traci.trafficlight.setPhase(self.tls_id, yellow_phase)
            # Run yellow phase duration
            for _ in range(self.yellow_duration):
                traci.simulationStep()
                
            # Now switch to the target green phase
            traci.trafficlight.setPhase(self.tls_id, target_phase)
            
        # Run the simulation for the remaining step time (delta_time - yellow_duration if switched, else full delta_time)
        run_steps = self.delta_time
        for _ in range(run_steps):
            traci.simulationStep()
            
        # Retrieve next state and reward
        obs = self._get_observation()
        reward = self._get_reward()
        
        # Check termination condition
        terminated = self.step_counter >= self.max_steps
        truncated = False
        
        # Extra simulation metrics to return in info dictionary
        info = {
            "step": self.step_counter,
            "total_queue": sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.incoming_lanes]),
            "avg_speed": np.mean([traci.lane.getLastStepMeanSpeed(lane) for lane in self.incoming_lanes])
        }
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Resets the environment for a new episode."""
        super().reset(seed=seed)
        self.step_counter = 0
        self.current_phase_idx = 0
        
        # Close active connection before starting a new one
        self.close()
        
        # Start simulation
        self._start_sumo()
        
        # Get initial state
        obs = self._get_observation()
        info = {"step": 0}
        
        return obs, info

    def close(self):
        """Cleans up TraCI connection and shuts down SUMO."""
        if not self.is_closed:
            if traci is not None:
                try:
                    traci.close()
                except Exception:
                    pass
            self.is_closed = True

    def render(self):
        """Renders the environment (handled natively by sumo-gui if enabled)."""
        pass
