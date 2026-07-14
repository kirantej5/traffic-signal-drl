# GPU-Accelerated Deep Reinforcement Learning for Adaptive Traffic Signal Control

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.0+-green.svg)](https://stable-baselines3.readthedocs.io/)
[![SUMO](https://img.shields.io/badge/SUMO-1.18+-lightgrey.svg)](https://eclipse.dev/sumo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A professional, GPU-accelerated deep reinforcement learning (DRL) framework for Adaptive Traffic Signal Control (ATSC). This repository contains the complete implementation for a final-year project utilizing **PyTorch**, **Stable-Baselines3**, and the **SUMO** micro-traffic simulator connected via **TraCI**.

```
                           +--------------------------------------+
                           |  DRL Agent (Stable-Baselines3/DQN)   |
                           |  [PyTorch Neural Network on GPU/CUDA]|
                           +--------------------------------------+
                                     |                ^
                             Action: |                | Reward,
                        Green Phase  |                | Traffic State
                                     v                |
                           +--------------------------------------+
                           |        Gymnasium Environment         |
                           +--------------------------------------+
                                     |                ^
                           TraCI API |                | Simulation
                                     v                | Data
                           +--------------------------------------+
                           |   Micro-Traffic Simulator (SUMO CPU) |
                           +--------------------------------------+
```

---

## 🚀 Key Features

* **Custom Gymnasium Environment**: A highly configurable Gym environment wrapping SUMO through TraCI, exposing vehicle density and halting queue states.
* **Dynamic Lane Discovery**: Automatically detects traffic light controlled lanes, allowing you to easily swap in custom SUMO scenarios.
* **GPU-Accelerated Training**: Integrated with Stable-Baselines3 (DQN) with full PyTorch CUDA support for rapid policy updates.
* **Realistic Traffic Transition**: Implements mandatory yellow signal transition phases (safety logic) during green signal phase switches.
* **Robust Evaluation & Plotting**: Built-in script that runs evaluation rollouts, logs metrics to CSV, and auto-generates comparative plots (Queue lengths, speeds, rewards).

---

## 📂 Project Structure

```
├── data/                    # Traffic network and scenario configurations
│   ├── generate_scenario.py # Script to create standard 4-way intersection files
│   ├── intersection.net.xml # Compiled road network layout
│   ├── intersection.rou.xml # Vehicle route flows definition
│   └── intersection.sumocfg # Main SUMO configuration file
├── docs/                    # Architectural and methodology documentation
│   └── methodology.md       # State/Action MDP formulation and architecture details
├── models/                  # Saved models and checkpoints
├── results/                 # Tensorboard logs, CSV metrics, and figures
├── src/                     # Core python codebase
│   ├── __init__.py          # Package marker
│   ├── environment.py       # Custom Gym/TraCI interface environment
│   ├── train.py             # Stable-Baselines3 training script
│   ├── evaluate.py          # Evaluation and metric plotting script
│   └── utils.py             # Helper tools (GPU checks, paths, learning curves)
├── requirements.txt         # Package dependencies
├── .gitignore               # Ignored cache, models, and log files
├── LICENSE                  # MIT License details
└── README.md                # Project documentation overview
```

---

## 🛠️ Installation & Setup

### 1. Install SUMO (Simulation of Urban MObility)
Ensure SUMO is installed on your local machine:
* **Windows**: Download the installer from the [Eclipse SUMO website](https://eclipse.dev/sumo/) and run it.
* **Linux (Ubuntu)**: Run:
  ```bash
  sudo apt-get update
  sudo apt-get install sumo sumo-tools sumo-doc
  ```

### 2. Configure Environment Variables
Verify that the `SUMO_HOME` environment variable is configured to point to your SUMO installation folder:
* **Windows (PowerShell)**:
  ```powershell
  [Environment]::SetEnvironmentVariable("SUMO_HOME", "C:\Program Files (x86)\Eclipse\Sumo", "User")
  ```
* **Linux**: Add this line to your `~/.bashrc`:
  ```bash
  export SUMO_HOME="/usr/share/sumo"
  ```

### 3. Install Python Dependencies
Create a virtual environment and install the required library dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚦 Usage Guide

### 1. Generate/Rebuild Scenario Files
To recreate or modify the standard 4-way intersection road network and realistic flow definitions, run:
```bash
python data/generate_scenario.py
```

### 2. Train the DRL Agent (DQN)
Train the agent utilizing PyTorch on your GPU. The training script automatically uses CUDA if an NVIDIA GPU is available:
```bash
python src/train.py --total-timesteps 100000 --reward-type queue
```
*Specify `--gui` to visually monitor vehicles while training, or `--device cpu` to force CPU training.*

### 3. Monitor Training with TensorBoard
View live training convergence metrics (loss, rewards, episode length):
```bash
tensorboard --logdir results/tensorboard
```

### 4. Evaluate the Agent & Generate Metrics
Evaluate the trained agent, outputting step logs to CSV and rendering performance plots in the `results/` folder:
```bash
python src/evaluate.py --model-path models/dqn_traffic_final --gui
```
This command generates the evaluation results, including:
* `results/evaluation_results_ep_1.csv` (step-by-step halting counts and vehicle speeds)
* `results/evaluation_metrics.png` (visual plots of queue lengths and speed trajectories)

---

## 🔧 Replacing with a Custom SUMO Scenario

This project is built to support user-customized maps and traffic junctions. To run the DQN agent on your own scenario:
1. Copy your `.net.xml`, `.rou.xml`, and `.sumocfg` files into the `data/` directory.
2. In the training execution, reference your files:
   ```bash
   python src/train.py --net-file data/my_map.net.xml --route-file data/my_map.rou.xml --sumo-cfg data/my_map.sumocfg --tls-id my_traffic_light_node
   ```
3. The environment dynamically inspects `my_traffic_light_node` to determine lane layouts, queue measurements, and action indexes, requiring no changes to the core code.

---

## 📝 Project Details & Methodology

A complete overview of the state representations, reward functions, DQN neural network architecture, and traffic light safety mechanisms can be found in the [Methodology Documentation](docs/methodology.md).

---

## 🎓 License & Credits

* **License**: Open-source under the [MIT License](LICENSE).
* **Developer**: kirantej5
* **Frameworks Used**: [SUMO](https://github.com/eclipse-sumo/sumo), [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [PyTorch](https://pytorch.org/).
