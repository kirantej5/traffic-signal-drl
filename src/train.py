import os
import argparse
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from environment import SumoTrafficEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN agent for Adaptive Traffic Signal Control using PyTorch and SUMO.")
    parser.add_argument("--net-file", type=str, default="data/intersection.net.xml", help="Path to SUMO network file")
    parser.add_argument("--route-file", type=str, default="data/intersection.rou.xml", help="Path to SUMO routes file")
    parser.add_argument("--sumo-cfg", type=str, default="data/intersection.sumocfg", help="Path to SUMO configuration file")
    parser.add_argument("--tls-id", type=str, default="C", help="ID of traffic light to control")
    parser.add_argument("--reward-type", type=str, default="queue", choices=["queue", "wait", "combined"], help="Reward function type")
    parser.add_argument("--tb-log", type=str, default="results/tensorboard", help="TensorBoard log directory")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save trained models")
    parser.add_argument("--gui", action="store_true", help="Run training with SUMO GUI enabled (recommended for debugging only)")
    parser.add_argument("--total-timesteps", type=int, default=100000, help="Total timesteps to train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device for neural networks (PyTorch)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Establish output directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.tb_log, exist_ok=True)
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Resolve GPU/CPU device execution
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        
    print(f"============================================================")
    print(f"[Training Mode] Starting Deep Q-Network Training")
    print(f"Device: {device.upper()}")
    print(f"Target Timesteps: {args.total_timesteps}")
    print(f"Reward function: {args.reward_type}")
    print(f"============================================================")
    
    # Initialize Gymnasium environment wrapper
    env = SumoTrafficEnv(
        net_file=args.net_file,
        route_file=args.route_file,
        sumo_cfg_file=args.sumo_cfg,
        tls_id=args.tls_id,
        use_gui=args.gui,
        reward_type=args.reward_type,
        max_steps=3600,
        delta_time=5
    )
    
    # Wrap environment with SB3 Monitor for logging episode rewards
    env = Monitor(env, filename=os.path.join(args.tb_log, "monitor.csv"))
    
    # Set up Deep Q-Network hyperparameters (MlpPolicy for feature vectors)
    # Using PyTorch GPU acceleration internally during backprop updates
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=args.tb_log,
        device=device,
        policy_kwargs=dict(net_arch=[128, 128])
    )
    
    # Define Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(args.model_dir, "checkpoints"),
        name_prefix="dqn_traffic"
    )
    
    print("[Info] Commencing DRL agent training loop...")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # Save final model
        final_model_path = os.path.join(args.model_dir, "dqn_traffic_final")
        model.save(final_model_path)
        print(f"[Success] Training completed. Saved final model weights to: {final_model_path}")
        
    except KeyboardInterrupt:
        print("[Warning] Training interrupted by user. Saving checkpoint...")
        model.save(os.path.join(args.model_dir, "dqn_traffic_interrupted"))
        
    finally:
        env.close()

if __name__ == "__main__":
    main()
