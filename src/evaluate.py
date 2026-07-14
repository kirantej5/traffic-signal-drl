import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

from environment import SumoTrafficEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent on a SUMO traffic light intersection.")
    parser.add_argument("--model-path", type=str, default="models/dqn_traffic_final", help="Path to trained model zip file")
    parser.add_argument("--net-file", type=str, default="data/intersection.net.xml", help="Path to SUMO network file")
    parser.add_argument("--route-file", type=str, default="data/intersection.rou.xml", help="Path to SUMO routes file")
    parser.add_argument("--sumo-cfg", type=str, default="data/intersection.sumocfg", help="Path to SUMO configuration file")
    parser.add_argument("--tls-id", type=str, default="C", help="ID of traffic light to control")
    parser.add_argument("--gui", action="store_true", help="Run simulation with SUMO GUI rendering enabled")
    parser.add_argument("--out-dir", type=str, default="results", help="Directory to save evaluation results and plots")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run evaluation")
    return parser.parse_args()

def plot_metrics(df, out_dir):
    """Generates and saves visual plots comparing DQN metrics during evaluation."""
    plt.figure(figsize=(12, 8))
    
    # 1. Plot Halting Queue Length over Time
    plt.subplot(2, 2, 1)
    plt.plot(df["step"], df["total_queue"], color="crimson", label="Queue Length")
    plt.xlabel("Simulation Steps (seconds)")
    plt.ylabel("Halting Vehicles (count)")
    plt.title("Queue Length over Time")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # 2. Plot Average Speed over Time
    plt.subplot(2, 2, 2)
    plt.plot(df["step"], df["avg_speed"], color="dodgerblue", label="Avg Speed")
    plt.xlabel("Simulation Steps (seconds)")
    plt.ylabel("Speed (m/s)")
    plt.title("Average Vehicle Speed")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    # 3. Plot Instantaneous Rewards
    plt.subplot(2, 2, 3)
    plt.plot(df["step"], df["reward"], color="forestgreen", label="Step Reward")
    plt.xlabel("Simulation Steps (seconds)")
    plt.ylabel("Reward")
    plt.title("Instantaneous Rewards")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # 4. Plot Cumulative Rewards
    plt.subplot(2, 2, 4)
    plt.plot(df["step"], np.cumsum(df["reward"]), color="purple", label="Cumulative Reward")
    plt.xlabel("Simulation Steps (seconds)")
    plt.ylabel("Total Reward")
    plt.title("Cumulative Reward over Episode")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "evaluation_metrics.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[Success] Metric plot successfully generated at: {plot_path}")

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"============================================================")
    print(f"[Evaluation Mode] Loading Trained DRL Policy")
    print(f"Model path: {args.model_path}")
    print(f"Running for: {args.episodes} episode(s)")
    print(f"SUMO GUI: {'Enabled' if args.gui else 'Disabled (Headless)'}")
    print(f"============================================================")
    
    # Initialize environment
    env = SumoTrafficEnv(
        net_file=args.net_file,
        route_file=args.route_file,
        sumo_cfg_file=args.sumo_cfg,
        tls_id=args.tls_id,
        use_gui=args.gui,
        max_steps=3600,
        delta_time=5
    )
    
    # Check if trained model weights exist; if not, we run random baseline policy
    model_exists = os.path.exists(args.model_path + ".zip") or os.path.exists(args.model_path)
    
    if model_exists:
        print("[Info] Model weights found. Loading trained policy...")
        model = DQN.load(args.model_path, env=env)
    else:
        print("[Warning] Model weights NOT found. Executing random baseline control policy...")
        model = None

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        
        # Log lists
        steps = []
        queues = []
        speeds = []
        rewards = []
        
        print(f"[Info] Running Episode {ep+1}...")
        while not done:
            # Predict action
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()  # Random baseline
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            steps.append(info.get("step", 0))
            queues.append(info.get("total_queue", 0))
            speeds.append(info.get("avg_speed", 0.0))
            rewards.append(reward)
            
        print(f"[Info] Episode {ep+1} completed.")
        
        # Compile dataframe and save csv
        df = pd.DataFrame({
            "step": steps,
            "total_queue": queues,
            "avg_speed": speeds,
            "reward": rewards
        })
        
        csv_path = os.path.join(args.out_dir, f"evaluation_results_ep_{ep+1}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[Success] Simulation logs saved to: {csv_path}")
        
        # Generate plot
        plot_metrics(df, args.out_dir)
        
    env.close()

if __name__ == "__main__":
    main()
