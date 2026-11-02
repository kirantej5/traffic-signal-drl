# 🚦 Neighborhood-Scale Multi-Agent DRL with Dynamic Vehicle Routing for Adaptive Traffic Signal Control

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SUMO](https://img.shields.io/badge/SUMO-1.28+-4CAF50?style=for-the-badge)](https://eclipse.dev/sumo/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0+-blue?style=for-the-badge)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-F5C518?style=for-the-badge)](LICENSE)

> A cooperative Multi-Agent Reinforcement Learning (MARL) framework pairing **Multi-Agent PPO (MAPPO)** and **Graph Attention Networks (GAT)** with **Dynamic Vehicle Routing** across an urban neighborhood grid in Eclipse SUMO.

---

## 📸 System Overview

```
                      +---------------------------------------+
                      |         Dynamic Vehicle Router        |
                      |   - Edge impedance monitoring         |
                      |   - TraCI online travel-time updates  |
                      |   - Anti-oscillation route cooldown   |
                      +---------------------------------------+
                                          ^
                                          | Reroutes
                                          v
+-----------------------+     +-----------------------+     +-----------------------+
|  Agent 1 (TLS A0)     | <-> |  Agent 2 (TLS A1)     | <-> |  Agent 3 (TLS A2)     |
|  Decentralized Actor  |     |  Decentralized Actor  |     |  Decentralized Actor  |
+-----------------------+     +-----------------------+     +-----------------------+
           ^                              ^                             ^
           |                              |                             |
           +------------------------------+-----------------------------+
                                          |
                      +---------------------------------------+
                      |       Centralized Critic (MAPPO)      |
                      |   - Global neighborhood state vector  |
                      |   - Resolves multi-agent non-station. |
                      |   - GAE advantage estimation          |
                      +---------------------------------------+
                                          |
                      +---------------------------------------+
                      |  MultiIntersectionEnv (Shared TraCI)  |
                      |   - 16 Controlled Intersections (4x4) |
                      |   - 1,670 Heterogeneous Vehicles      |
                      +---------------------------------------+
```

---

## ✨ Features

- **Cooperative MARL (MAPPO)** — Decentralized actors select local green phases; a centralized critic evaluates the global 608-dimensional neighborhood state during training.
- **Dynamic Vehicle Routing** — Real-time closed-loop rerouting of delayed vehicles via TraCI using dynamic link impedances and 30-second cooldown protection.
- **Graph Attention Policy (GAT-RL)** — Spatial graph attention layer learning dynamic attention weights across neighboring intersections to form automated "green wave" corridors.
- **Shared Cooperative Reward** — Parameterized reward ($R_i = \alpha R_{local} + (1-\alpha) R_{neighbor}$) preventing agents from creating downstream spillback congestion.
- **Single TraCI Connection Architecture** — High-performance execution managing all 16 traffic lights within a single shared SUMO simulation process.
- **Academic Metrics Calculator** — Automatically computes travel time saved, throughput gain (veh/hr), fuel/CO₂ emissions reduction, and Jain's Fairness Index.
- **Paper-Ready Visualizations** — Built-in plotting suite producing 300 DPI queue progression and comparative benchmark figures.

---

## 📁 Repository Structure

```
traffic-signal-drl/
├── configs/
│   ├── mappo_config.yaml         # MAPPO learning rates, clipping, batch sizes
│   └── network_4x4.yaml          # Grid topology, road lengths, vehicle demand
├── data/
│   ├── generate_neighborhood.py  # 4x4 Grid generator with peak/off-peak demand
│   ├── neighborhood.net.xml      # Compiled 16-intersection SUMO network
│   ├── neighborhood.rou.xml      # 1,670 heterogeneous vehicles (cars, buses, trucks)
│   └── neighborhood.sumocfg      # SUMO simulation configuration
├── docs/
│   ├── methodology.md            # Technical architecture and Dec-POMDP formulation
│   └── research_paper_draft.md   # IEEE manuscript draft (T-ITS / ITSC)
├── models/
│   ├── mappo/                    # Trained MAPPO checkpoints and final weights
│   └── checkpoints/              # Periodic training weights
├── results/
│   ├── marl/                     # CSV training and benchmark logs
│   └── comparative_plots/        # 300 DPI publication-quality figures
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── mappo_agent.py        # Centralized Critic MAPPO implementation
│   │   ├── ippo_agent.py         # Independent PPO baseline
│   │   └── gnn_policy.py         # Graph Attention Network policy head
│   ├── multi_env.py              # MultiIntersectionEnv (shared TraCI multi-agent)
│   ├── routing.py                # Dynamic vehicle rerouting engine
│   ├── metrics.py                # Academic ITS metrics calculator
│   ├── train_marl.py             # Multi-agent training pipeline
│   ├── evaluate_marl.py          # Benchmark comparison suite
│   ├── environment.py            # Legacy single-intersection Gym env
│   └── utils.py                  # SUMO path helpers and graph utilities
├── checklist.txt                 # Step-by-step project milestone tracker
├── summary.txt                   # Complete project summary
└── requirements.txt              # Python dependencies
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.11+**
- **Eclipse SUMO 1.28+** ([Download](https://eclipse.dev/sumo/))

Install Python dependencies:
```bash
pip install -r requirements.txt
```
For the tested reproducible dependency set:
```bash
pip install -r requirements.txt -c requirements-pinned.txt
```

### 2. Generate the Neighborhood Scenario
Generate the 4x4 grid network and dynamic peak traffic flows:
```bash
python data/generate_neighborhood.py
```

### 3. Train the MAPPO Controller
Train the multi-agent system with dynamic vehicle routing:
```bash
python src/train_marl.py --episodes 15 --max-steps 1800 --alpha 0.5
```

### 4. Evaluate Against Standard Baselines
Benchmark against Webster Fixed-Time and Actuated controllers:
```bash
python src/evaluate_marl.py --eval-steps 600
```

### 5. Interactive Visualizations & Live Monitoring

#### A. Real-Time Web Dashboard (Browser)
Watch live traffic metrics, queue progression, mean speeds, and per-intersection status in an auto-updating web browser interface:
```bash
python scripts/dashboard.py
```
*Open http://localhost:5000 to view the live dashboard during simulation.*
- Compare against fixed-time baseline: `python scripts/dashboard.py --fixed-time`

#### B. SUMO-GUI Visual Mode (Live Micro-simulation)
Watch vehicles move through the 4×4 neighborhood grid with dynamic signal phases and color-coded vehicle speeds:
```bash
python scripts/run_gui.py
```
- Compare with fixed-time baseline: `python scripts/run_gui.py --fixed-time`
- Adjust playback speed: `python scripts/run_gui.py --delay 100` (delay in ms/step)
- Controls inside SUMO-GUI: `[Space]` Play/Pause, `[Ctrl+A]` Fit all in view, `[+] / [-]` Speed up/slow down.

### 6. Reproducible Multi-Seed Experiments

Run the complete generate/train/evaluate/statistics workflow for five seeds:

```bash
python scripts/run_statistical_analysis.py \
  --seeds 42,43,44,45,46 \
  --episodes 2 \
  --max-steps 300 \
  --eval-steps 300
```

The script stores raw per-seed observations in `results/marl/statistical_analysis/per_seed_results.csv` and the paired t-test with a 95% confidence interval in `results/marl/statistical_analysis/paired_t_test.json`. Generated model checkpoints are stored under the ignored `models/checkpoints/statistical_analysis/` directory.

Published exploratory result for seeds 42–46: mean paired difference **−0.012 vehicles**, 95% CI **[−0.127615, 0.103615]**, paired t-statistic **−0.288175**, and p-value **0.78752811**. This short two-episode experiment did not show a statistically significant improvement at α = 0.05; longer multi-seed runs are needed for publication-quality conclusions.

---

## 📊 Benchmark Results

The currently tracked benchmark summary records mean queue length for the fixed-time baseline and the proposed controller:

| Control Strategy | Mean Queue (veh) | Improvement |
| :--- | :---: | :---: |
| **Fixed-Time Baseline** | 1.36 | Baseline |
| **MAPPO + Dynamic Routing** | **0.24** | **82.21% lower** |

Source: [`results/marl/benchmark_comparison_summary.csv`](results/marl/benchmark_comparison_summary.csv). Other metrics and controller comparisons should be added to the CSV before being presented as benchmark results.

---

## 📖 Citation

If you use this codebase or build upon our research, please cite:

```bibtex
@misc{tej2026mappo_traffic,
  author = {Tej, Kiran},
  title = {Cooperative Multi-Agent Deep Reinforcement Learning with Dynamic Vehicle Routing for Neighborhood-Scale Traffic Signal Control},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/kirantej5/traffic-signal-drl}}
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
