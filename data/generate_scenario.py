import os
import subprocess
import sys
import xml.etree.ElementTree as ET

def create_nodes_file(filepath):
    """Creates the nodes XML file for a standard 4-way intersection."""
    root = ET.Element("nodes")
    
    # Central intersection node (controlled by Traffic Light)
    ET.SubElement(root, "node", id="C", x="0.0", y="0.0", type="traffic_light")
    
    # Outer nodes
    ET.SubElement(root, "node", id="N", x="0.0", y="250.0", type="priority")
    ET.SubElement(root, "node", id="S", x="0.0", y="-250.0", type="priority")
    ET.SubElement(root, "node", id="E", x="250.0", y="0.0", type="priority")
    ET.SubElement(root, "node", id="W", x="-250.0", y="0.0", type="priority")
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ", level=0)
    tree.write(filepath, encoding="utf-8", xml_declaration=True)
    print(f"[Info] Created node definition file: {filepath}")

def create_edges_file(filepath):
    """Creates the edges XML file connecting outer nodes to the center."""
    root = ET.Element("edges")
    
    # Define speed limits (m/s) and lane counts
    speed_limit = "13.89"  # ~50 km/h
    num_lanes = "3"        # Left turn lane, straight, right/straight lane
    
    # Incoming edges
    ET.SubElement(root, "edge", id="N2C", **{"from": "N", "to": "C", "numLanes": num_lanes, "speed": speed_limit})
    ET.SubElement(root, "edge", id="S2C", **{"from": "S", "to": "C", "numLanes": num_lanes, "speed": speed_limit})
    ET.SubElement(root, "edge", id="E2C", **{"from": "E", "to": "C", "numLanes": num_lanes, "speed": speed_limit})
    ET.SubElement(root, "edge", id="W2C", **{"from": "W", "to": "C", "numLanes": num_lanes, "speed": speed_limit})
    
    # Outgoing edges
    ET.SubElement(root, "edge", id="C2N", **{"from": "C", "to": "N", "numLanes": num_lanes, "speed": speed_limit})
    ET.SubElement(root, "edge", id="C2S", **{"from": "C", "to": "S", "numLanes": num_lanes, "speed": speed_limit})
    ET.SubElement(root, "edge", id="C2E", **{"from": "C", "to": "E", "numLanes": num_lanes, "speed": speed_limit})
    ET.SubElement(root, "edge", id="C2W", **{"from": "C", "to": "W", "numLanes": num_lanes, "speed": speed_limit})
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ", level=0)
    tree.write(filepath, encoding="utf-8", xml_declaration=True)
    print(f"[Info] Created edge definition file: {filepath}")

def compile_network(nod_file, edg_file, net_file):
    """Compiles the network file using netconvert."""
    print("[Info] Compiling network using netconvert...")
    
    # Check if SUMO_HOME is set
    sumo_home = os.environ.get("SUMO_HOME")
    netconvert_bin = "netconvert"
    if sumo_home:
        netconvert_bin = os.path.join(sumo_home, "bin", "netconvert")
        
    cmd = [
        netconvert_bin,
        f"--node-files={nod_file}",
        f"--edge-files={edg_file}",
        f"--output-file={net_file}",
        "--no-turnarounds=true",              # Prevent vehicle U-turns at the intersection
        "--tls.default-type=static",          # Initialize with a standard static TLS program
        "--tls.guess=true"                    # Automatic traffic light generation for central node C
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        print(f"[Success] Network compiled successfully at: {net_file}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[Error] Failed to compile network using netconvert. Details: {e}", file=sys.stderr)
        if isinstance(e, subprocess.CalledProcessError):
            print(f"netconvert Output: {e.stdout}", file=sys.stderr)
            print(f"netconvert Error: {e.stderr}", file=sys.stderr)
        print("\n[Warning] Please verify that SUMO is installed and added to your system PATH or environment variable SUMO_HOME.", file=sys.stderr)
        return False

def generate_route_file(filepath, num_steps=3600, passenger_flow_rate=0.15, truck_flow_rate=0.03):
    """Generates realistic vehicle flows and routes for a 4-way intersection.
    
    passenger_flow_rate: probability of generating a passenger vehicle per second
    truck_flow_rate: probability of generating a truck vehicle per second
    """
    root = ET.Element("routes")
    
    # Vehicle types definitions
    ET.SubElement(root, "vType", id="passenger", accel="2.6", decel="4.5", sigma="0.5", length="5.0", minGap="2.5", maxSpeed="16.67", guiShape="passenger", color="0,1,0")
    ET.SubElement(root, "vType", id="truck", accel="1.2", decel="3.0", sigma="0.5", length="12.0", minGap="3.0", maxSpeed="11.11", guiShape="truck", color="1,0,0")
    
    # Routes definitions
    routes = {
        "N_to_S": "N2C C2S",
        "N_to_E": "N2C C2E",
        "N_to_W": "N2C C2W",
        
        "S_to_N": "S2C C2N",
        "S_to_E": "S2C C2E",
        "S_to_W": "S2C C2W",
        
        "E_to_W": "E2C C2W",
        "E_to_N": "E2C C2N",
        "E_to_S": "E2C C2S",
        
        "W_to_E": "W2C C2E",
        "W_to_N": "W2C C2N",
        "W_to_S": "W2C C2S",
    }
    
    for route_id, edges in routes.items():
        ET.SubElement(root, "route", id=route_id, edges=edges)
        
    # Generate random flows of vehicles dynamically over the steps
    import random
    random.seed(42) # For reproducible route file generation
    
    veh_counter = 0
    
    # Add comment in XML
    root.append(ET.Comment("Vehicle flow definitions"))
    
    for step in range(num_steps):
        # Sample for passenger cars
        for route_id in routes.keys():
            # Check generation probability for passenger cars
            if random.random() < passenger_flow_rate / len(routes):
                ET.SubElement(root, "vehicle", {
                    "id": f"veh_{veh_counter}",
                    "type": "passenger",
                    "route": route_id,
                    "depart": str(step),
                    "departLane": "best",
                    "departSpeed": "max"
                })
                veh_counter += 1
                
            # Check generation probability for trucks
            if random.random() < truck_flow_rate / len(routes):
                ET.SubElement(root, "vehicle", {
                    "id": f"veh_{veh_counter}",
                    "type": "truck",
                    "route": route_id,
                    "depart": str(step),
                    "departLane": "best",
                    "departSpeed": "max"
                })
                veh_counter += 1
                
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ", level=0)
    tree.write(filepath, encoding="utf-8", xml_declaration=True)
    print(f"[Info] Created routes file with {veh_counter} vehicles: {filepath}")

def create_sumo_config(filepath, net_file, rou_file):
    """Creates the main .sumocfg configuration file linking net and route files."""
    root = ET.Element("configuration", **{"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance", "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/sumoConfiguration.xsd"})
    
    input_node = ET.SubElement(root, "input")
    ET.SubElement(input_node, "net-file", value=os.path.basename(net_file))
    ET.SubElement(input_node, "route-files", value=os.path.basename(rou_file))
    
    time_node = ET.SubElement(root, "time")
    ET.SubElement(time_node, "begin", value="0")
    ET.SubElement(time_node, "end", value="3600")
    
    report_node = ET.SubElement(root, "report")
    ET.SubElement(report_node, "verbose", value="true")
    ET.SubElement(report_node, "no-step-log", value="true")
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ", level=0)
    tree.write(filepath, encoding="utf-8", xml_declaration=True)
    print(f"[Info] Created SUMO configuration file: {filepath}")

if __name__ == "__main__":
    # Base directory set to where this file lives
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    nod_file = os.path.join(base_dir, "intersection.nod.xml")
    edg_file = os.path.join(base_dir, "intersection.edg.xml")
    net_file = os.path.join(base_dir, "intersection.net.xml")
    rou_file = os.path.join(base_dir, "intersection.rou.xml")
    cfg_file = os.path.join(base_dir, "intersection.sumocfg")
    
    # 1. Create nodes and edges
    create_nodes_file(nod_file)
    create_edges_file(edg_file)
    
    # 2. Compile network
    compiled = compile_network(nod_file, edg_file, net_file)
    
    # 3. Create route file with realistic flows
    generate_route_file(rou_file, num_steps=3600, passenger_flow_rate=0.20, truck_flow_rate=0.04)
    
    # 4. Create SUMO config
    create_sumo_config(cfg_file, net_file, rou_file)
    
    print("\n[Complete] Scenario generation finished.")
