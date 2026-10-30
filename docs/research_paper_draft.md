# Cooperative Multi-Agent Deep Reinforcement Learning with Dynamic Vehicle Routing for Neighborhood-Scale Traffic Signal Control

**Authors**: *Kiran et al.*  
**Target Venue**: IEEE Transactions on Intelligent Transportation Systems (T-ITS) / IEEE ITSC  

---

## Abstract
Urban traffic congestion leads to substantial economic losses, fuel waste, and elevated carbon emissions. While single-intersection Deep Reinforcement Learning (DRL) controllers have shown promise, they fail to coordinate across adjacent intersections, frequently causing vehicle queue spillbacks and congestion bottlenecks. In this paper, we propose a scalable, cooperative multi-agent traffic control architecture that jointly optimizes traffic signal timings and vehicle routing at a neighborhood scale. We formulate the multi-intersection coordination problem as a Decentralized Partially Observable Markov Decision Process (Dec-POMDP) and solve it using Multi-Agent Proximal Policy Optimization (MAPPO) with Centralized Training and Decentralized Execution (CTDE). Furthermore, we integrate a closed-loop dynamic vehicle routing layer via microscopic simulation (SUMO/TraCI) that recalculates optimal travel paths in response to evolving signal timings. Experimental results across a 16-intersection urban grid demonstrate that our proposed cooperative MAPPO system achieves significant reductions in average vehicle queue lengths, waiting times, and vehicular emissions compared to standardized Webster Fixed-Time and Actuated control baselines, while maintaining high corridor throughput and network equity.

---

## 1. Introduction
Modern urban transportation systems increasingly face acute congestion during peak hours. Traditional traffic light controllers rely primarily on fixed-time cycle schedules (e.g., Webster's method) or local sensor-actuated systems. While simple and reliable, these legacy systems cannot adapt to rapid, non-linear shifts in traffic demand.

In recent years, Deep Reinforcement Learning (DRL) has emerged as an effective paradigm for Adaptive Traffic Signal Control (ATSC). However, the vast majority of existing DRL implementations focus on isolated, single-intersection setups. In a dense urban neighborhood, isolated control policies cause the "spillback" phenomenon: clearing an intersection aggressively may overwhelm downstream arterial roads, paralyzing the wider network.

To resolve this, we present an end-to-end framework featuring:
1. **Cooperative Multi-Agent Control (MAPPO)**: Local actors select green phases using both local lane states and 1-hop neighbor traffic information, while a centralized critic observes the global neighborhood state during training to resolve multi-agent non-stationarity.
2. **Dynamic Vehicle Rerouting**: Vehicles dynamically reroute based on real-time edge travel times and intersection delays, diffusing congestion before severe queues materialize.
3. **Cooperative Reward Mixing**: A parameterized reward structure ($R_i = \alpha R_{local} + (1-\alpha) R_{neighbor}$) that aligns local intersection throughput with overall neighborhood efficiency.

---

## 2. Problem Formulation (Dec-POMDP)
We model the $N$-intersection network as a Decentralized Partially Observable Markov Decision Process defined by the tuple:
$$\langle \mathcal{N}, \mathcal{S}, \{\mathcal{A}_i\}, \mathcal{P}, \{\mathcal{R}_i\}, \Omega, \{\mathcal{O}_i\}, \gamma \rangle$$

- **Agents $\mathcal{N}$**: Set of traffic light controllers $\{1, 2, \dots, N\}$.
- **Local Observation $\mathcal{O}_i$**: For intersection $i$, the state vector captures normalized queue length, vehicle density, and mean velocity across controlled incoming lanes, augmented with the one-hot encoding of the active signal phase.
- **Neighbor Communication**: Each agent's observation is augmented with summary features from 1-hop neighbors $\mathcal{N}(i)$:
  $$s_i(t) = s_i^{local}(t) \oplus \bigoplus_{j \in \mathcal{N}(i)} [\bar{q}_j(t), \bar{d}_j(t), \phi_j(t)]$$
- **Action Space $\mathcal{A}_i$**: Discrete selection among non-conflicting green phase configurations (e.g., North-South Green vs. East-West Green). When an agent alters its green phase, a mandatory 4-second yellow transition phase is automatically executed to ensure deceleration safety.
- **Cooperative Reward $\mathcal{R}_i$**:
  $$R_i(t) = \alpha R_{local, i}(t) + (1 - \alpha) \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} R_{local, j}(t)$$
  where $R_{local, i}(t) = -\sum_{l \in \mathcal{L}_i} (q_l(t) + 0.05 \cdot w_l(t))$.

---

## 3. Methodology & System Architecture

### 3.1 Multi-Agent PPO with Centralized Critic (MAPPO)
To counter environmental non-stationarity where multiple agents update policies simultaneously, we employ Centralized Training with Decentralized Execution (CTDE):
- **Decentralized Actor $\pi_\theta(a_i \mid o_i)$**: Maps each agent's 38-dimensional local+neighbor observation to phase action probabilities.
- **Centralized Critic $V_\phi(s_{global})$**: Evaluates the global 608-dimensional state vector encompassing all 16 intersections during training, computing Generalized Advantage Estimation (GAE):
  $$\hat{A}_t = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$$

### 3.2 Dynamic Vehicle Routing Layer
The dynamic routing engine queries link speeds and queue occupancies every $\Delta t_{route}$ seconds. The dynamic impedance for edge $e$ is:
$$\text{Cost}(e) = \frac{\text{Length}(e)}{\max(1.0, \bar{v}_e)} + \beta \cdot N_{halt}(e)$$
En-route vehicles experiencing delay below a velocity threshold are selectively rerouted via TraCI with an individual 30-second cooldown period to prevent oscillatory route flapping.

---

## 4. Experimental Setup & Results

### 4.1 Simulation Environment
- **Simulator**: Eclipse SUMO 1.28+ via TraCI.
- **Network**: 4x4 Grid Neighborhood (16 controlled junctions, 200m link lengths, 2 lanes per approach).
- **Traffic Demand**: 1,670 heterogeneous vehicles (85% passenger cars, 10% transit buses, 5% heavy trucks) simulated over 3,600 seconds with morning and evening peak surges.
- **Emissions Standard**: HBEFA3 vehicle emission models.

### 4.2 Benchmark Comparisons
| Control Strategy | Mean Halting Queue (veh) | Total Wait Time (s/veh) | Throughput (veh/hr) | CO₂ Abatement (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Webster Fixed-Time** | 18.4 | 46.2 | 1,120 | Baseline |
| **SUMO Actuated** | 11.2 | 29.8 | 1,280 | +12.4% |
| **Independent PPO (IPPO)** | 7.9 | 19.4 | 1,390 | +21.8% |
| **Proposed MAPPO + Routing** | **2.8** | **9.6** | **1,540** | **+36.5%** |

---

## 5. Ablation Studies
1. **Communication Value**: Disabling 1-hop neighbor message passing increases average queue lengths by 34%, confirming the necessity of spatial awareness.
2. **Dynamic Routing Synergy**: Signal control without dynamic routing results in isolated arterial bottlenecks; enabling closed-loop vehicle rerouting distributes vehicular load across parallel grid corridors.
3. **Reward Factor ($\alpha$)**: An intermediate mixing factor ($\alpha = 0.5$) outperforms both purely selfish ($\alpha = 1.0$) and purely global ($\alpha = 0.0$) objective functions.

---

## 6. Conclusion
This paper presented a cooperative Multi-Agent Reinforcement Learning framework with dynamic vehicle routing for neighborhood-scale traffic signal management. By combining decentralized MAPPO policies with centralized value guidance and real-time path re-allocation, the system eliminates queue spillbacks and establishes coordinated green corridors across the network. Future work will investigate real-world deployment on open-source OpenStreetMap (OSM) metropolitan networks and edge-computing micro-controllers.

