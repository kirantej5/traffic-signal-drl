import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import torch

def setup_sumo_paths():
    """Validates and sets SUMO_HOME path variables to ensure TraCI loads properly."""
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)
        return True
    else:
        # Standard system default search paths
        search_paths = [
            r"C:\Program Files (x86)\Eclipse\Sumo\tools",
            r"C:\Program Files\Eclipse\Sumo\tools",
            "/usr/share/sumo/tools",
            "/usr/local/share/sumo/tools"
        ]
        for path in search_paths:
            if os.path.exists(path):
                sys.path.append(path)
                os.environ["SUMO_HOME"] = os.path.dirname(path)
                print(f"[Info] Dynamically configured SUMO_HOME to: {os.environ['SUMO_HOME']}")
                return True
    return False

def check_gpu_status():
    """Queries and returns PyTorch CUDA GPU availability details."""
    gpu_available = torch.cuda.is_available()
    device_details = {}
    if gpu_available:
        device_details["device_name"] = torch.cuda.get_device_name(0)
        device_details["device_count"] = torch.cuda.device_count()
        device_details["cuda_version"] = torch.version.cuda
    return gpu_available, device_details

def plot_training_results(monitor_csv_path: str, save_plot_path: str = "results/training_curve.png"):
    """Reads a Stable-Baselines3 monitor.csv log and plots the rolling reward convergence."""
    if not os.path.exists(monitor_csv_path):
        print(f"[Error] Monitor log file not found at: {monitor_csv_path}")
        return
        
    try:
        # SB3 Monitor file has 2 lines of metadata header
        df = pd.read_csv(monitor_csv_path, skiprows=1)
        
        # Calculate moving average rewards (window of 10 episodes)
        df["rolling_reward"] = df["r"].rolling(window=min(10, len(df)), min_periods=1).mean()
        
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df["r"], alpha=0.3, color="teal", label="Episode Reward")
        plt.plot(df.index, df["rolling_reward"], color="darkcyan", linewidth=2, label="10-Ep Moving Avg")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title("DQN Traffic Controller Training Convergence")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        
        os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
        plt.savefig(save_plot_path, dpi=300)
        plt.close()
        print(f"[Success] Training curve plotted successfully at: {save_plot_path}")
    except Exception as e:
        print(f"[Error] Failed to generate training curve. Details: {e}")

if __name__ == "__main__":
    # Test script output
    print("SUMO Path Check:", setup_sumo_paths())
    gpu, details = check_gpu_status()
    print("GPU Acceleration Support:", gpu)
    if gpu:
        print(f"Device: {details['device_name']} (CUDA v{details['cuda_version']})")
