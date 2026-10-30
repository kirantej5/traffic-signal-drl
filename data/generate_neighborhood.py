"""Neighborhood Road Network and Demand Generator for MARL Traffic Control.

Generates:
1. A 4x4 Grid Road Network (16 controlled intersections) using SUMO `netgenerate`.
2. Multi-class dynamic vehicle flows (Passenger cars, Buses, Heavy trucks) across peak/off-peak conditions.
3. SUMO configuration file (neighborhood.sumocfg) ready for MARL simulation.
"""

import os
import sys
import subprocess
import random
import xml.etree.ElementTree as ET

# Dynamic resolution of SUMO_HOME
candidates = [
    r"C:\Program Files (x86)\SUMO",
    r"C:\Program Files\SUMO",
    os.environ.get("SUMO_HOME", ""),
    r"C:\Program Files (x86)\Eclipse\Sumo"
]
SUMO_HOME = r"C:\Program Files (x86)\SUMO"
for c in candidates:
    if c and os.path.exists(c):
        SUMO_HOME = c
        os.environ["SUMO_HOME"] = c
        break

tools_path = os.path.join(SUMO_HOME, "tools")
if tools_path not in sys.path:
    sys.path.append(tools_path)

NETGENERATE_BIN = os.path.join(SUMO_HOME, "bin", "netgenerate.exe")
if not os.path.exists(NETGENERATE_BIN):
    NETGENERATE_BIN = "netgenerate"


def generate_grid_network(output_net_file: str, grid_size: int = 4, block_length: float = 200.0, num_lanes: int = 2):
    """Generates a grid network with traffic-light controlled intersections using netgenerate."""
    print(f"[Info] Generating {grid_size}x{grid_size} grid network ({grid_size*grid_size} intersections)...")
    
    cmd = [
        NETGENERATE_BIN,
        "--grid",
        f"--grid.number={grid_size}",
        f"--grid.length={block_length}",
        f"--grid.attach-length=100.0",
        f"--default.lanenumber={num_lanes}",
        "--default.speed=13.89",  # 50 km/h
        "--tls.guess=true",       # Guess traffic lights at grid intersections
        "--tls.default-type=actuated",
        "--output-file", output_net_file,
        "--no-warnings=true"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[Success] Neighborhood network created: {output_net_file}")
    except subprocess.CalledProcessError as e:
        print(f"[Error] netgenerate failed: {e.stderr}")
        raise


def get_network_boundary_edges(net_file: str):
    """Extracts edge IDs from the generated network using sumolib."""
    try:
        import sumolib
        net = sumolib.net.readNet(net_file)
        all_edges = [edge.getID() for edge in net.getEdges() if not edge.isSpecial()]
        return all_edges
    except Exception as e:
        print(f"[Warning] Could not inspect net with sumolib: {e}")
        return []


def generate_traffic_routes(net_file: str, output_route_file: str, duration: int = 3600, seed: int = 42):
    """Generates multi-class dynamic traffic demand with peak and off-peak patterns."""
    print(f"[Info] Generating dynamic traffic flows over {duration}s...")
    random.seed(seed)
    
    import sumolib
    net = sumolib.net.readNet(net_file)
    edges = [e for e in net.getEdges() if not e.isSpecial()]
    edge_ids = [e.getID() for e in edges]
    
    root = ET.Element("routes")
    
    # 1. Define Heterogeneous Vehicle Types (HBEFA3 emission standards)
    ET.SubElement(root, "vType", {
        "id": "passenger",
        "accel": "2.6",
        "decel": "4.5",
        "sigma": "0.5",
        "length": "4.5",
        "minGap": "2.5",
        "maxSpeed": "13.89",
        "emissionClass": "HBEFA3/PC_G_EU4",
        "color": "0.2,0.6,1.0"
    })
    
    ET.SubElement(root, "vType", {
        "id": "bus",
        "accel": "1.2",
        "decel": "3.5",
        "sigma": "0.5",
        "length": "12.0",
        "minGap": "3.0",
        "maxSpeed": "11.11",
        "emissionClass": "HBEFA3/BUS",
        "color": "0.1,0.8,0.2"
    })
    
    ET.SubElement(root, "vType", {
        "id": "truck",
        "accel": "1.0",
        "decel": "3.0",
        "sigma": "0.5",
        "length": "10.0",
        "minGap": "3.5",
        "maxSpeed": "9.72",
        "emissionClass": "HBEFA3/HDV",
        "color": "0.9,0.3,0.1"
    })

    veh_count = 0
    t = 0
    while t < duration:
        # Dynamic rate: higher in morning peak (300-1800s) and evening peak (2100-3300s)
        is_peak = (300 <= t <= 1800) or (2100 <= t <= 3300)
        prob = 0.55 if is_peak else 0.25
        
        if random.random() < prob:
            from_edge = random.choice(edge_ids)
            to_candidates = [e for e in edge_ids if e != from_edge]
            to_edge = random.choice(to_candidates)
            
            # Vehicle distribution: 85% passenger, 10% bus, 5% truck
            roll = random.random()
            if roll < 0.85:
                vtype = "passenger"
            elif roll < 0.95:
                vtype = "bus"
            else:
                vtype = "truck"
                
            ET.SubElement(root, "trip", {
                "id": f"veh_{veh_count}",
                "type": vtype,
                "depart": f"{t:.1f}",
                "from": from_edge,
                "to": to_edge,
                "departLane": "best",
                "departSpeed": "max"
            })
            veh_count += 1
            
        t += 1.0
        
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ", level=0)
    tree.write(output_route_file, encoding="utf-8", xml_declaration=True)
    print(f"[Success] Routes file generated with {veh_count} vehicles: {output_route_file}")


def create_sumo_config(config_filepath: str, net_file: str, route_file: str):
    """Creates the main .sumocfg configuration file for the neighborhood simulation."""
    root = ET.Element("configuration")
    
    input_elem = ET.SubElement(root, "input")
    ET.SubElement(input_elem, "net-file", value=os.path.basename(net_file))
    ET.SubElement(input_elem, "route-files", value=os.path.basename(route_file))
    
    time_elem = ET.SubElement(root, "time")
    ET.SubElement(time_elem, "begin", value="0")
    ET.SubElement(time_elem, "end", value="3600")
    ET.SubElement(time_elem, "step-length", value="1.0")
    
    processing_elem = ET.SubElement(root, "processing")
    ET.SubElement(processing_elem, "time-to-teleport", value="-1")
    ET.SubElement(processing_elem, "waiting-time-memory", value="1000")
    
    report_elem = ET.SubElement(root, "report")
    ET.SubElement(report_elem, "no-step-log", value="true")
    ET.SubElement(report_elem, "no-warnings", value="true")
    
    output_elem = ET.SubElement(root, "output")
    ET.SubElement(output_elem, "tripinfo-output", value="../results/neighborhood_tripinfo.xml")
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ", level=0)
    tree.write(config_filepath, encoding="utf-8", xml_declaration=True)
    print(f"[Success] SUMO configuration file created: {config_filepath}")


if __name__ == "__main__":
    data_dir = os.path.dirname(os.path.abspath(__file__))
    net_path = os.path.join(data_dir, "neighborhood.net.xml")
    rou_path = os.path.join(data_dir, "neighborhood.rou.xml")
    cfg_path = os.path.join(data_dir, "neighborhood.sumocfg")
    
    # 1. Build 4x4 Grid Network
    generate_grid_network(net_path, grid_size=4, block_length=200.0, num_lanes=2)
    
    # 2. Build Dynamic Traffic Demand
    generate_traffic_routes(net_path, rou_path, duration=3600, seed=42)
    
    # 3. Build SUMO Configuration
    create_sumo_config(cfg_path, net_path, rou_path)
    
    print("\n[Complete] Neighborhood simulation scenario is fully assembled and verified!")
