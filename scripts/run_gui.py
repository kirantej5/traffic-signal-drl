"""
Run MARL Traffic Simulation in SUMO-GUI Mode
=============================================
Launches the SUMO graphical interface so you can visually watch:
  - Vehicles moving through the 4x4 neighborhood grid
  - Traffic lights switching (Green/Yellow/Red) controlled by MAPPO agents
  - Color-coded vehicle speeds (red=stopped, green=fast)
  - Real-time simulation step counter

Usage:
    python scripts/run_gui.py                    # Trained MAPPO agents
    python scripts/run_gui.py --fixed-time       # Fixed-time baseline (compare!)
    python scripts/run_gui.py --delay 100        # Slower playback (ms per step)
    python scripts/run_gui.py --steps 3600       # Run for 1 hour of sim time

Controls in SUMO-GUI:
    [Space]        Play / Pause
    [+] / [-]      Speed up / Slow down
    [Scroll]       Zoom in / out
    [Ctrl+A]       Fit entire network into view
    [F]            Toggle vehicle labels
    [T]            Toggle junction names
"""

import os
import sys
import time
import argparse
import threading

# ── Path setup ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

SUMO_CANDIDATES = [
    r"C:\Program Files (x86)\SUMO",
    r"C:\Program Files\SUMO",
    os.environ.get("SUMO_HOME", ""),
]
for _candidate in SUMO_CANDIDATES:
    if _candidate and os.path.exists(_candidate):
        os.environ["SUMO_HOME"] = _candidate
        _tools = os.path.join(_candidate, "tools")
        if _tools not in sys.path:
            sys.path.insert(0, _tools)
        break

try:
    import traci
    import sumolib
except ImportError:
    print("[ERROR] TraCI not found. Install it: pip install traci sumolib")
    sys.exit(1)

import numpy as np

# ── Argument parsing ─────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SUMO-GUI visualization of MARL traffic control.")
    p.add_argument("--net-file",   default="data/neighborhood.net.xml", help="SUMO network file")
    p.add_argument("--route-file", default="data/neighborhood.rou.xml", help="SUMO routes file")
    p.add_argument("--model-path", default="models/mappo/mappo_traffic_final.pt",
                   help="Trained MAPPO model path")
    p.add_argument("--steps",   type=int, default=1800, help="Number of simulation steps")
    p.add_argument("--delay",   type=int, default=50,   help="GUI delay in ms per step (lower=faster)")
    p.add_argument("--fixed-time", action="store_true",
                   help="Use fixed-time control instead of MAPPO (for comparison)")
    p.add_argument("--phase-duration", type=int, default=30,
                   help="Fixed phase duration in seconds (only with --fixed-time)")
    p.add_argument("--gui-settings", default="configs/gui_settings.xml",
                   help="Path to SUMO-GUI view settings XML")
    return p.parse_args()


# ── Fixed-time controller (baseline) ────────────────────────
class FixedTimeController:
    """Cycles through phases every `phase_duration` seconds."""
    def __init__(self, tls_ids, phase_duration=30):
        self.tls_ids = tls_ids
        self.phase_duration = phase_duration
        self.timers = {tid: 0 for tid in tls_ids}

    def act(self, step):
        for tid in self.tls_ids:
            try:
                num_phases = len(traci.trafficlight.getAllProgramLogics(tid)[0].phases)
                current    = traci.trafficlight.getPhase(tid)
                elapsed    = traci.trafficlight.getNextSwitch(tid) - traci.simulation.getTime()
                # Let SUMO handle fixed-time naturally — just don't override
            except Exception:
                pass


# ── MAPPO controller (inference only) ───────────────────────
class MAPPOInferenceController:
    """Loads trained actor networks and runs inference to pick actions."""
    def __init__(self, tls_ids, model_path, obs_dim=38):
        self.tls_ids  = tls_ids
        self.obs_dim  = obs_dim
        self.actors   = {}
        self._load_model(model_path)

    def _load_model(self, model_path):
        try:
            import torch
            from agents.mappo_agent import MAPPOTrafficController
            self.controller = MAPPOTrafficController(
                agent_ids=self.tls_ids,
                obs_dim=self.obs_dim,
                global_state_dim=self.obs_dim * len(self.tls_ids),
                action_dim=2,
            )
            self.controller.load(model_path)
            self.use_model = True
            print(f"[OK] Loaded MAPPO model from {model_path}")
        except Exception as e:
            print(f"[WARN] Could not load model ({e}). Falling back to random policy.")
            self.use_model = False

    def _get_obs(self, tid, controlled_lanes):
        obs = np.zeros(38, dtype=np.float32)
        lanes = controlled_lanes.get(tid, [])[:8]
        for i, lane in enumerate(lanes):
            try:
                obs[i*3]   = min(traci.lane.getLastStepHaltingNumber(lane) / 20.0, 1.0)
                obs[i*3+1] = min(traci.lane.getLastStepVehicleNumber(lane) / 20.0, 1.0)
                obs[i*3+2] = min(traci.lane.getLastStepMeanSpeed(lane)     / 15.0, 1.0)
            except Exception:
                pass
        return obs

    def act(self, tls_ids, controlled_lanes, step, yellow_timers):
        obs_dict = {tid: self._get_obs(tid, controlled_lanes) for tid in tls_ids}
        import torch
        actions = {}
        for tid in tls_ids:
            if yellow_timers.get(tid, 0) > 0:
                yellow_timers[tid] -= 1
                actions[tid] = None  # in yellow phase, don't override
                continue
            if self.use_model:
                obs_t = torch.FloatTensor(obs_dict[tid]).unsqueeze(0)
                with torch.no_grad():
                    logits = self.controller.actors[tid].actor_net(obs_t)
                    action = torch.argmax(logits).item()
            else:
                action = int(step // 30) % 2
            actions[tid] = action
        return actions


# ── HUD overlay helper ───────────────────────────────────────
def print_hud(step, total_steps, tls_ids, mode, elapsed_real):
    """Prints a live terminal HUD alongside the GUI."""
    try:
        n_vehicles   = traci.vehicle.getIDCount()
        n_teleported = traci.simulation.getEndingTeleportNumber()
        sim_time     = traci.simulation.getTime()

        # Compute mean queue across all TLS
        total_q = 0
        for tid in tls_ids:
            try:
                lanes = traci.trafficlight.getControlledLanes(tid)
                total_q += sum(traci.lane.getLastStepHaltingNumber(l) for l in set(lanes))
            except Exception:
                pass
        mean_q = total_q / max(len(tls_ids), 1)

        bar_len  = 30
        progress = int(bar_len * step / total_steps)
        bar      = "[" + "#" * progress + "-" * (bar_len - progress) + "]"

        print(f"\r{bar} {step:>4}/{total_steps}  "
              f"Sim: {sim_time:>6.0f}s  "
              f"Mode: {mode:<10}  "
              f"Vehicles: {n_vehicles:>4}  "
              f"Mean Queue: {mean_q:>4.1f}  "
              f"Real: {elapsed_real:.0f}s",
              end="", flush=True)
    except Exception:
        pass


# ── Main GUI runner ──────────────────────────────────────────
def run_gui(args):
    net_file    = os.path.join(PROJECT_ROOT, args.net_file)
    route_file  = os.path.join(PROJECT_ROOT, args.route_file)
    gui_xml     = os.path.join(PROJECT_ROOT, args.gui_settings)
    model_path  = os.path.join(PROJECT_ROOT, args.model_path)

    # Validate files
    for label, fpath in [("net-file", net_file), ("route-file", route_file)]:
        if not os.path.exists(fpath):
            print(f"[ERROR] {label} not found: {fpath}")
            sys.exit(1)

    sumo_gui_bin = sumolib.checkBinary("sumo-gui")

    cmd = [
        sumo_gui_bin,
        "--net-file",      net_file,
        "--route-files",   route_file,
        "--step-length",   "1.0",
        "--no-step-log",   "true",
        "--no-warnings",   "true",
        "--collision.action", "none",
        "--delay",         str(args.delay),
        "--quit-on-end",   "true",
        "--start",                          # auto-start (don't wait for play button)
    ]
    if os.path.exists(gui_xml):
        cmd += ["--gui-settings-file", gui_xml]

    mode = "Fixed-Time" if args.fixed_time else "MAPPO (Trained)"
    print("=" * 65)
    print(f"  SUMO-GUI Visualization — {mode}")
    print(f"  Network : {os.path.basename(net_file)}")
    print(f"  Routes  : {os.path.basename(route_file)}")
    print(f"  Steps   : {args.steps}  |  GUI delay: {args.delay} ms/step")
    print("=" * 65)
    print("SUMO-GUI Controls:")
    print("  [Space]    Play / Pause        [+/-]   Speed up / Slow down")
    print("  [Ctrl+A]   Fit all in view     [F]     Toggle vehicle labels")
    print("  [T]        Junction names      [Scroll] Zoom")
    print("=" * 65)

    traci.start(cmd)

    # Discover TLS IDs
    tls_ids = sorted(traci.trafficlight.getIDList())
    print(f"[OK] Discovered {len(tls_ids)} traffic lights: {tls_ids[:4]} ...")

    # Build controlled lanes map
    controlled_lanes = {}
    for tid in tls_ids:
        controlled_lanes[tid] = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tid)))

    # Instantiate controller
    if args.fixed_time:
        controller = FixedTimeController(tls_ids, args.phase_duration)
    else:
        controller = MAPPOInferenceController(tls_ids, model_path)

    yellow_timers    = {tid: 0 for tid in tls_ids}
    YELLOW_DURATION  = 4
    GREEN_PHASES     = {}  # cache of green phase indices per TLS

    for tid in tls_ids:
        try:
            logic  = traci.trafficlight.getAllProgramLogics(tid)[0]
            greens = [i for i, ph in enumerate(logic.phases)
                      if 'G' in ph.state or 'g' in ph.state]
            GREEN_PHASES[tid] = greens if greens else [0]
        except Exception:
            GREEN_PHASES[tid] = [0]

    print("\nSimulation running — watch the SUMO-GUI window!")
    print("Press [Ctrl+C] here to stop early.\n")

    t_start = time.time()
    try:
        for step in range(args.steps):
            traci.simulationStep()

            if not args.fixed_time and isinstance(controller, MAPPOInferenceController):
                actions = controller.act(tls_ids, controlled_lanes, step, yellow_timers)
                for tid, action in actions.items():
                    if action is None:
                        continue
                    try:
                        greens    = GREEN_PHASES[tid]
                        target_ph = greens[action % len(greens)]
                        cur_ph    = traci.trafficlight.getPhase(tid)
                        if cur_ph != target_ph:
                            # Insert yellow transition
                            logic  = traci.trafficlight.getAllProgramLogics(tid)[0]
                            phases = logic.phases
                            # Find yellow phase after current green
                            yellow_ph = (cur_ph + 1) % len(phases)
                            traci.trafficlight.setPhase(tid, yellow_ph)
                            yellow_timers[tid] = YELLOW_DURATION
                    except Exception:
                        pass

            if step % 10 == 0:
                print_hud(step, args.steps, tls_ids, mode, time.time() - t_start)

    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
    except traci.exceptions.FatalTraCIError:
        print("\n[GUI window was closed]")
    finally:
        try:
            traci.close()
        except Exception:
            pass

    elapsed = time.time() - t_start
    print(f"\n\n{'='*65}")
    print(f"  Simulation complete!  Wall time: {elapsed:.1f}s")
    print(f"  Mode: {mode} | Steps simulated: {min(step+1, args.steps)}")
    print(f"{'='*65}")


if __name__ == "__main__":
    run_gui(parse_args())

