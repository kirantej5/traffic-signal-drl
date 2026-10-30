"""
Real-Time MARL Traffic Dashboard
=================================
Launches a live browser dashboard that updates every second while the simulation runs.

Shows:
  - Queue length per intersection (live bar chart)
  - Network-wide reward over time (line chart)
  - Speed heatmap across the 4x4 grid
  - Agent action table (which phase each TLS is in)
  - Summary stats: total vehicles, throughput, teleports

Usage:
    pip install flask                            # one-time install
    python scripts/dashboard.py                  # opens http://localhost:5000
    python scripts/dashboard.py --port 8080      # custom port
    python scripts/dashboard.py --fixed-time     # run baseline for comparison

Then open your browser to http://localhost:5000
"""

import os
import sys
import json
import time
import threading
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# ── SUMO path setup ──────────────────────────────────────────
SUMO_CANDIDATES = [
    r"C:\Program Files (x86)\SUMO",
    r"C:\Program Files\SUMO",
    os.environ.get("SUMO_HOME", ""),
]
for _c in SUMO_CANDIDATES:
    if _c and os.path.exists(_c):
        os.environ["SUMO_HOME"] = _c
        _t = os.path.join(_c, "tools")
        if _t not in sys.path:
            sys.path.insert(0, _t)
        break

try:
    import traci, sumolib
except ImportError:
    print("[ERROR] TraCI not found: pip install traci sumolib")
    sys.exit(1)

try:
    from flask import Flask, jsonify, render_template_string
except ImportError:
    print("[ERROR] Flask not found: pip install flask")
    sys.exit(1)

# ── Shared simulation state (thread-safe via lock) ───────────
_state_lock = threading.Lock()
_state = {
    "step":        0,
    "sim_time":    0.0,
    "mode":        "loading",
    "running":     True,
    "tls_ids":     [],
    "agents": {},           # {tid: {queue, speed, phase, reward}}
    "history": {
        "steps":   [],
        "rewards": [],
        "queues":  [],
        "speeds":  [],
        "throughput": [],
    },
    "totals": {
        "vehicles":   0,
        "teleports":  0,
        "throughput": 0,
    }
}

# ── HTML Dashboard template ──────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="1">   <!-- auto-refresh every 1s -->
<title>MARL Traffic Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root { --bg:#0f0f1a; --card:#1a1a2e; --accent:#3498db; --green:#1abc9c;
          --red:#e74c3c; --yellow:#f39c12; --text:#e0e0e0; --muted:#888; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:'Segoe UI',sans-serif;
         padding:16px; }
  h1 { text-align:center; color:var(--accent); margin-bottom:4px; font-size:1.5rem; }
  .subtitle { text-align:center; color:var(--muted); font-size:.85rem; margin-bottom:16px; }

  .stats-row { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:16px; }
  .stat-card { background:var(--card); border-radius:10px; padding:14px 20px;
               flex:1; min-width:140px; text-align:center; border:1px solid #2a2a4a; }
  .stat-card .val  { font-size:1.9rem; font-weight:bold; color:var(--accent); }
  .stat-card .lbl  { font-size:.75rem; color:var(--muted); margin-top:4px; }
  .stat-card.green .val { color:var(--green); }
  .stat-card.red .val   { color:var(--red); }
  .stat-card.yellow .val{ color:var(--yellow); }

  .charts-row { display:grid; grid-template-columns:1fr 1fr; gap:12px;
                margin-bottom:12px; }
  .chart-card { background:var(--card); border-radius:10px; padding:12px;
                border:1px solid #2a2a4a; }
  .chart-card h3 { font-size:.9rem; color:var(--muted); margin-bottom:8px; }

  .grid-section { background:var(--card); border-radius:10px; padding:14px;
                  margin-bottom:12px; border:1px solid #2a2a4a; }
  .grid-section h3 { font-size:.9rem; color:var(--muted); margin-bottom:10px; }
  .tls-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; }
  .tls-cell { border-radius:8px; padding:8px 6px; text-align:center;
              font-size:.75rem; transition:background .3s; }
  .tls-cell .name  { font-weight:bold; font-size:.85rem; }
  .tls-cell .queue { font-size:1.1rem; font-weight:bold; }
  .tls-cell .spd   { color:var(--muted); font-size:.7rem; }

  .table-section { background:var(--card); border-radius:10px; padding:14px;
                   border:1px solid #2a2a4a; overflow-x:auto; }
  .table-section h3 { font-size:.9rem; color:var(--muted); margin-bottom:8px; }
  table { width:100%; border-collapse:collapse; font-size:.8rem; }
  th { background:#16213e; color:var(--muted); padding:6px 10px; text-align:left; }
  td { padding:5px 10px; border-bottom:1px solid #2a2a4a; }
  tr:last-child td { border-bottom:none; }
  .phase-G { color:var(--green); font-weight:bold; }
  .phase-Y { color:var(--yellow); font-weight:bold; }
  .phase-R { color:var(--red); }

  .mode-badge { display:inline-block; padding:3px 12px; border-radius:20px;
                font-size:.75rem; font-weight:bold; margin-left:8px; }
  .mode-mappo  { background:#1a5276; color:var(--accent); }
  .mode-fixed  { background:#512e0e; color:var(--yellow); }
  .footer { text-align:center; color:var(--muted); font-size:.75rem; margin-top:12px; }
</style>
</head>
<body>
<h1>MARL Traffic Signal Control
  <span class="mode-badge {{ 'mode-mappo' if data.mode == 'MAPPO' else 'mode-fixed' }}">
    {{ data.mode }}
  </span>
</h1>
<div class="subtitle">
  Neighborhood 4x4 Grid &bull; {{ data.totals.vehicles }} vehicles active &bull;
  Simulation step {{ data.step }} / {{ data.max_steps }} &bull;
  Sim time {{ data.sim_time | round(0) | int }}s
</div>

<!-- ── Stats row ── -->
<div class="stats-row">
  <div class="stat-card green">
    <div class="val">{{ "%.2f"|format(data.mean_queue) }}</div>
    <div class="lbl">Mean Queue (veh)</div>
  </div>
  <div class="stat-card">
    <div class="val">{{ "%.1f"|format(data.mean_speed) }}</div>
    <div class="lbl">Mean Speed (m/s)</div>
  </div>
  <div class="stat-card yellow">
    <div class="val">{{ data.totals.throughput }}</div>
    <div class="lbl">Throughput (veh/h)</div>
  </div>
  <div class="stat-card red">
    <div class="val">{{ data.totals.teleports }}</div>
    <div class="lbl">Teleports</div>
  </div>
  <div class="stat-card">
    <div class="val">{{ data.totals.vehicles }}</div>
    <div class="lbl">Active Vehicles</div>
  </div>
  <div class="stat-card green">
    <div class="val">{{ "%.3f"|format(data.fairness) }}</div>
    <div class="lbl">Jain's Fairness</div>
  </div>
</div>

<!-- ── Charts ── -->
<div class="charts-row">
  <div class="chart-card">
    <h3>Queue Length Over Time (network mean)</h3>
    <canvas id="queueChart" height="120"></canvas>
  </div>
  <div class="chart-card">
    <h3>Network Mean Speed Over Time</h3>
    <canvas id="speedChart" height="120"></canvas>
  </div>
</div>

<!-- ── 4x4 Intersection Grid ── -->
<div class="grid-section">
  <h3>Live Intersection Status — Queue Length (veh) &bull; Speed (m/s)</h3>
  <div class="tls-grid">
    {% for tid, ag in data.agents.items() %}
    <div class="tls-cell" style="background:{{ ag.color }}">
      <div class="name">{{ tid }}</div>
      <div class="queue">{{ "%.1f"|format(ag.queue) }}</div>
      <div class="spd">{{ "%.1f"|format(ag.speed) }} m/s</div>
      <div style="font-size:.65rem;margin-top:2px">
        <span class="phase-{{ ag.phase_state }}">{{ ag.phase_label }}</span>
      </div>
    </div>
    {% endfor %}
  </div>
</div>

<!-- ── Agent table ── -->
<div class="table-section">
  <h3>Agent Details</h3>
  <table>
    <tr>
      <th>Intersection</th><th>Queue</th><th>Speed (m/s)</th>
      <th>Phase</th><th>Lanes</th><th>Halted</th>
    </tr>
    {% for tid, ag in data.agents.items() %}
    <tr>
      <td><b>{{ tid }}</b></td>
      <td>{{ "%.1f"|format(ag.queue) }}</td>
      <td>{{ "%.2f"|format(ag.speed) }}</td>
      <td><span class="phase-{{ ag.phase_state }}">{{ ag.phase_label }}</span></td>
      <td>{{ ag.n_lanes }}</td>
      <td>{{ ag.halted }}</td>
    </tr>
    {% endfor %}
  </table>
</div>

<div class="footer">
  Auto-refreshing every 1 second &bull; MARL Traffic Signal Control Research Project
</div>

<script>
const steps = {{ data.history.steps | tojson }};
const queues = {{ data.history.queues | tojson }};
const speeds = {{ data.history.speeds | tojson }};

const common = {
  borderWidth: 2,
  pointRadius: 0,
  tension: 0.3,
  fill: true,
};

new Chart(document.getElementById('queueChart'), {
  type: 'line',
  data: {
    labels: steps,
    datasets: [{
      ...common,
      label: 'Mean Queue',
      data: queues,
      borderColor: '#e74c3c',
      backgroundColor: 'rgba(231,76,60,0.15)',
    }]
  },
  options: {
    animation: false,
    scales: {
      x: { ticks: { color:'#888', maxTicksLimit:8 }, grid: { color:'#2a2a4a' } },
      y: { ticks: { color:'#888' }, grid: { color:'#2a2a4a' }, min: 0 }
    },
    plugins: { legend: { labels: { color:'#aaa' } } }
  }
});

new Chart(document.getElementById('speedChart'), {
  type: 'line',
  data: {
    labels: steps,
    datasets: [{
      ...common,
      label: 'Mean Speed (m/s)',
      data: speeds,
      borderColor: '#1abc9c',
      backgroundColor: 'rgba(26,188,156,0.15)',
    }]
  },
  options: {
    animation: false,
    scales: {
      x: { ticks: { color:'#888', maxTicksLimit:8 }, grid: { color:'#2a2a4a' } },
      y: { ticks: { color:'#888' }, grid: { color:'#2a2a4a' }, min: 0 }
    },
    plugins: { legend: { labels: { color:'#aaa' } } }
  }
});
</script>
</body>
</html>
"""

app = Flask(__name__)

def _queue_to_color(q):
    """Map queue length to a background color: low=green, high=red."""
    r = min(255, int(q * 35))
    g = max(40,  int(180 - q * 20))
    return f"rgba({r},{g},60,0.75)"

def _phase_label(phase_state_str):
    s = (phase_state_str or "").upper()
    if 'G' in s: return "GREEN", "G"
    if 'Y' in s: return "YELLOW", "Y"
    return "RED", "R"

@app.route("/")
def index():
    with _state_lock:
        data = dict(_state)
        data["mean_queue"] = float(np.mean([ag["queue"] for ag in data["agents"].values()])) if data["agents"] else 0.0
        data["mean_speed"] = float(np.mean([ag["speed"] for ag in data["agents"].values()])) if data["agents"] else 0.0
        # Jain's fairness on speeds
        speeds = [ag["speed"] for ag in data["agents"].values()]
        if speeds:
            s = np.array(speeds)
            data["fairness"] = float((s.sum()**2) / (len(s) * (s**2).sum() + 1e-9))
        else:
            data["fairness"] = 1.0
        data["max_steps"] = _max_steps
    return render_template_string(HTML, data=data)

@app.route("/api/state")
def api_state():
    with _state_lock:
        return jsonify(_state)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--net-file",   default="data/neighborhood.net.xml")
    p.add_argument("--route-file", default="data/neighborhood.rou.xml")
    p.add_argument("--model-path", default="models/mappo/mappo_traffic_final.pt")
    p.add_argument("--steps",  type=int, default=1800)
    p.add_argument("--port",   type=int, default=5000)
    p.add_argument("--fixed-time", action="store_true")
    p.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    return p.parse_args()

_max_steps = 1800

def simulation_thread(args):
    """Runs the SUMO simulation headlessly, updating _state for the dashboard."""
    global _max_steps
    _max_steps = args.steps

    net_file   = os.path.join(PROJECT_ROOT, args.net_file)
    route_file = os.path.join(PROJECT_ROOT, args.route_file)

    sumo_bin = sumolib.checkBinary("sumo")
    cmd = [
        sumo_bin,
        "--net-file",      net_file,
        "--route-files",   route_file,
        "--step-length",   "1.0",
        "--no-step-log",   "true",
        "--no-warnings",   "true",
        "--collision.action", "none",
    ]

    try:
        traci.start(cmd)
    except Exception as e:
        print(f"[ERROR] SUMO failed to start: {e}")
        with _state_lock:
            _state["mode"] = "ERROR"
            _state["running"] = False
        return

    tls_ids = sorted(traci.trafficlight.getIDList())
    controlled_lanes = {
        tid: list(dict.fromkeys(traci.trafficlight.getControlledLanes(tid)))
        for tid in tls_ids
    }

    with _state_lock:
        _state["tls_ids"] = tls_ids
        _state["mode"]    = "Fixed-Time" if args.fixed_time else "MAPPO"

    # Load model if MAPPO
    ctrl = None
    if not args.fixed_time:
        try:
            import torch
            from agents.mappo_agent import MAPPOTrafficController
            ctrl = MAPPOTrafficController(
                n_agents=len(tls_ids), obs_dim=38, action_dim=2,
            )
            ctrl.load(os.path.join(PROJECT_ROOT, args.model_path))
            print(f"[OK] MAPPO model loaded ({len(tls_ids)} agents)")
        except Exception as e:
            print(f"[WARN] Model load failed ({e}), using random policy")

    yellow_timers = {tid: 0 for tid in tls_ids}
    YELLOW_DUR    = 4
    arrived_total = 0
    t_start       = time.time()

    for step in range(args.steps):
        try:
            traci.simulationStep()
        except traci.exceptions.FatalTraCIError:
            break

        arrived_total += traci.simulation.getArrivedNumber()

        # ── Collect per-TLS metrics ──
        agents_snapshot = {}
        for tid in tls_ids:
            lanes = controlled_lanes.get(tid, [])
            halted = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes[:8])
            speeds = [traci.lane.getLastStepMeanSpeed(l) for l in lanes[:8] if lanes]
            queue  = halted / max(len(lanes), 1)
            speed  = float(np.mean(speeds)) if speeds else 0.0
            try:
                ph_idx = traci.trafficlight.getPhase(tid)
                logic  = traci.trafficlight.getAllProgramLogics(tid)[0]
                ph_str = logic.phases[ph_idx].state if ph_idx < len(logic.phases) else ""
            except Exception:
                ph_str = ""
            ph_label, ph_state = _phase_label(ph_str)
            agents_snapshot[tid] = {
                "queue":       round(queue, 2),
                "speed":       round(speed, 2),
                "phase_label": ph_label,
                "phase_state": ph_state,
                "n_lanes":     len(lanes),
                "halted":      halted,
                "color":       _queue_to_color(queue),
            }

        # ── Apply MAPPO actions ──
        if ctrl and not args.fixed_time:
            try:
                obs_dict = {
                    tid: np.array([
                        agents_snapshot[tid]["queue"],
                        agents_snapshot[tid]["speed"] / 15.0,
                    ] + [0.0]*36, dtype=np.float32)
                    for tid in tls_ids
                }
                model_actions, _, _, _ = ctrl.select_actions(obs_dict, deterministic=True)
                for tid in tls_ids:
                    if yellow_timers.get(tid, 0) > 0:
                        yellow_timers[tid] -= 1
                        continue
                    action = model_actions.get(tid, 0)
                    logic  = traci.trafficlight.getAllProgramLogics(tid)[0]
                    greens = [i for i,ph in enumerate(logic.phases) if 'G' in ph.state or 'g' in ph.state]
                    if greens:
                        target = greens[action % len(greens)]
                        if traci.trafficlight.getPhase(tid) != target:
                            traci.trafficlight.setPhase(tid, (target-1)%len(logic.phases))
                            yellow_timers[tid] = YELLOW_DUR
            except Exception:
                pass

        # ── Update shared state ──
        mean_q   = float(np.mean([a["queue"] for a in agents_snapshot.values()]))
        mean_spd = float(np.mean([a["speed"] for a in agents_snapshot.values()]))
        elapsed  = time.time() - t_start
        tp       = int(arrived_total / max(elapsed / 3600, 1e-6))  # veh/h

        with _state_lock:
            _state["step"]       = step
            _state["sim_time"]   = float(traci.simulation.getTime())
            _state["agents"]     = agents_snapshot
            _state["totals"]["vehicles"]  = traci.vehicle.getIDCount()
            _state["totals"]["teleports"] = traci.simulation.getEndingTeleportNumber()
            _state["totals"]["throughput"] = tp
            hist = _state["history"]
            if step % 5 == 0:         # update charts every 5 steps
                hist["steps"].append(step)
                hist["queues"].append(round(mean_q, 3))
                hist["speeds"].append(round(mean_spd, 3))
                # Keep last 200 points for chart readability
                for k in ["steps","queues","speeds"]:
                    if len(hist[k]) > 200:
                        hist[k] = hist[k][-200:]

    try:
        traci.close()
    except Exception:
        pass

    with _state_lock:
        _state["running"] = False
    print("\n[OK] Simulation complete.")


if __name__ == "__main__":
    args = parse_args()
    _max_steps = args.steps

    print("=" * 60)
    print("  MARL Traffic Real-Time Dashboard")
    print(f"  Mode  : {'Fixed-Time' if args.fixed_time else 'MAPPO (Trained)'}")
    print(f"  Steps : {args.steps}")
    print(f"  URL   : http://localhost:{args.port}")
    print("=" * 60)
    print("Starting simulation in background thread...")

    # Start SUMO simulation in a daemon thread
    sim_t = threading.Thread(target=simulation_thread, args=(args,), daemon=True)
    sim_t.start()

    # Auto-open browser after 2 seconds
    if not args.no_browser:
        def _open():
            time.sleep(2)
            import webbrowser
            webbrowser.open(f"http://localhost:{args.port}")
        threading.Thread(target=_open, daemon=True).start()

    # Start Flask (blocks until Ctrl+C)
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)

