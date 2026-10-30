"""Multi-Agent PPO (MAPPO) Training Pipeline for Neighborhood Traffic Control.

Executes:
1. Environment instantiation with MultiIntersectionEnv.
2. Dynamic vehicle routing synchronization.
3. Centralized Critic MAPPO trajectory collection.
4. Periodic checkpointing and loss logging.
"""

import os
import argparse
import numpy as np
import torch
import pandas as pd

from multi_env import MultiIntersectionEnv
from routing import DynamicRouter
from agents.mappo_agent import MAPPOTrafficController


def parse_args():
    parser = argparse.ArgumentParser(description="Train MAPPO on Neighborhood Traffic Grid.")
    parser.add_argument("--net-file", type=str, default="data/neighborhood.net.xml", help="SUMO network file")
    parser.add_argument("--route-file", type=str, default="data/neighborhood.rou.xml", help="SUMO routes file")
    parser.add_argument("--sumo-cfg", type=str, default="data/neighborhood.sumocfg", help="SUMO config file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=1800, help="Simulation steps per episode")
    parser.add_argument("--rollout-len", type=int, default=64, help="PPO trajectory rollout buffer length")
    parser.add_argument("--model-dir", type=str, default="models/mappo", help="Directory to save model weights")
    parser.add_argument("--results-dir", type=str, default="results/marl", help="Directory to save training logs")
    parser.add_argument("--alpha", type=float, default=0.5, help="Cooperative reward factor")
    parser.add_argument("--enable-routing", action="store_true", default=True, help="Enable dynamic vehicle rerouting")
    parser.add_argument("--gui", action="store_true", help="Launch SUMO GUI for visual inspection")
    return parser.parse_args()


def train():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 65)
    print(" [TRAINING] Starting Neighborhood MAPPO Training Pipeline")
    print(f" Target Episodes: {args.episodes} | Steps per Episode: {args.max_steps}")
    print(f" Dynamic Routing: {'Enabled' if args.enable_routing else 'Disabled'}")
    print(f" Reward Alpha: {args.alpha}")
    print("=" * 65)

    env = MultiIntersectionEnv(
        net_file=args.net_file,
        route_file=args.route_file,
        sumo_cfg_file=args.sumo_cfg,
        use_gui=args.gui,
        max_steps=args.max_steps,
        delta_time=5,
        alpha=args.alpha
    )

    router = DynamicRouter() if args.enable_routing else None

    # Retrieve agent specifications
    sample_obs, _ = env.reset()
    n_agents = len(sample_obs)
    sample_tid = list(sample_obs.keys())[0]
    obs_dim = sample_obs[sample_tid].shape[0]
    action_dim = env.num_actions

    controller = MAPPOTrafficController(
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device="cpu"
    )

    episode_rewards = []
    episode_queues = []

    for ep in range(args.episodes):
        obs_dict, info = env.reset()
        if router:
            router.reset()

        trajectory_buffer = []
        ep_reward = 0.0
        step_count = 0
        done = False

        print(f"\n--- Commencing Episode {ep + 1}/{args.episodes} ---")

        while not done:
            # 1. Action selection via decentralized actor & centralized critic
            actions, log_probs, val, gstate = controller.select_actions(obs_dict)

            # 2. Step environment
            next_obs_dict, rewards_dict, term_dict, trunc_dict, step_info = env.step(actions)
            step_count = step_info.get("step", 0)

            # 3. Dynamic vehicle rerouting
            if router:
                router.step_reroute(step=step_count)

            done = all(term_dict.values()) or all(trunc_dict.values())
            mean_step_reward = sum(rewards_dict.values()) / n_agents
            ep_reward += mean_step_reward

            # Record trajectory step
            trajectory_buffer.append({
                "obs": obs_dict,
                "actions": actions,
                "log_probs": log_probs,
                "rewards": rewards_dict,
                "global_state": gstate,
                "value": val,
                "done": done
            })

            obs_dict = next_obs_dict

            # 4. Perform PPO update when rollout buffer fills
            if len(trajectory_buffer) >= args.rollout_len or done:
                loss_dict = controller.update(trajectory_buffer)
                trajectory_buffer = []

        episode_rewards.append(ep_reward)
        last_q = step_info.get("total_network_queue", 0)
        episode_queues.append(last_q)

        print(f"[Ep {ep+1} Finished] Total Reward: {ep_reward:.2f} | Final Network Queue: {last_q}")

        # Checkpoint model every 2 episodes
        if (ep + 1) % 2 == 0:
            ckpt_path = os.path.join(args.model_dir, f"mappo_ep_{ep+1}.pt")
            controller.save(ckpt_path)
            print(f"[Checkpoint] Saved weights to {ckpt_path}")

    env.close()

    # Save final model
    final_path = os.path.join(args.model_dir, "mappo_traffic_final.pt")
    controller.save(final_path)
    print(f"\n[Success] Training complete! Saved final weights to {final_path}")

    # Save training trajectory log
    log_df = pd.DataFrame({
        "episode": list(range(1, args.episodes + 1)),
        "reward": episode_rewards,
        "queue": episode_queues
    })
    log_path = os.path.join(args.results_dir, "training_log.csv")
    log_df.to_csv(log_path, index=False)
    print(f"[Success] Training log saved to {log_path}")


if __name__ == "__main__":
    train()

