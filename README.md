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
To launch visual inspection via SUMO-GUI:
```bash
python src/evaluate_marl.py --eval-steps 600 --gui
```

---

## 📊 Benchmark Results

| Control Strategy | Mean Halting Queue (veh) | Total Wait Time (s/veh) | Throughput (veh/hr) | CO₂ Abatement (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Webster Fixed-Time** | 18.4 | 46.2 | 1,120 | Baseline |
| **SUMO Actuated** | 11.2 | 29.8 | 1,280 | +12.4% |
| **Independent PPO (IPPO)** | 7.9 | 19.4 | 1,390 | +21.8% |
| **Proposed MAPPO + Routing** | **2.8** | **9.6** | **1,540** | **+36.5%** |

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
