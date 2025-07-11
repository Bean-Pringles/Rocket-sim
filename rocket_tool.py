"""
Rocket Operations Suite - Advanced TI-84 Style Terminal App
All files (logs, configs, reports) are saved in the 'Rocket Simulation' folder.
Features:
- Mission planning with motor presets and JSON config load/save
- Realistic drag and apogee simulation physics
- Advanced flight event scripting with conditions
- Flight log analysis with matplotlib graphs
- Telemetry replay and drift estimation
- Generates text reports and checklists

Run with Python 3. Requires: pandas, matplotlib
"""

import os
import json
import csv
import time
import math
import random
from datetime import datetime
from collections import deque

import pandas as pd
import matplotlib.pyplot as plt

# --- Set directories inside "Rocket Simulation" folder ---
OUTPUT_DIR = "Rocket Simulation"
CONFIG_DIR = os.path.join(OUTPUT_DIR, "configs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# === Motor presets: name -> dict with thrust curve and params ===
MOTOR_PRESETS = {
    "Estes D12": {
        "total_impulse": 12.0,  # Newton-seconds
        "burn_time": 1.6,       # seconds
        "thrust_curve": [(0, 12), (1.6, 0)],
        "avg_thrust": 7.5,
        "mass": 0.024  # kg
    },
    "Estes E9": {
        "total_impulse": 9.0,
        "burn_time": 1.2,
        "thrust_curve": [(0, 9), (1.2, 0)],
        "avg_thrust": 7.5,
        "mass": 0.021
    },
    "Custom": {}
}

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def pause():
    input("\nPress ENTER to return to the main menu...")
    clear()

# --- Utility: interpolate thrust from thrust curve ---
def interp_thrust(thrust_curve, t):
    if not thrust_curve or len(thrust_curve) < 2:
        return 0
    for i in range(len(thrust_curve)-1):
        t0, thrust0 = thrust_curve[i]
        t1, thrust1 = thrust_curve[i+1]
        if t0 <= t <= t1:
            return thrust0 + (thrust1 - thrust0) * (t - t0)/(t1 - t0)
    if t > thrust_curve[-1][0]:
        return 0
    return thrust_curve[0][1]

# --- Physics constants ---
GRAVITY = 9.81  # m/s^2
AIR_DENSITY = 1.225  # kg/m^3 at sea level

# === MissionConfig class for JSON save/load ===
class MissionConfig:
    def __init__(self):
        self.name = "NewMission"
        self.rocket_mass = 0.5  # kg
        self.rocket_diameter = 0.05  # meters
        self.drag_coefficient = 0.75
        self.motor = "Custom"
        self.motor_params = MOTOR_PRESETS.get("Custom", {})
        self.payload_mass = 0.0
        self.events = []  # List of event dicts
    
    def load(self, filename):
        path = os.path.join(CONFIG_DIR, filename)
        if not os.path.isfile(path):
            print(f"No config file found at {path}")
            return False
        with open(path, 'r') as f:
            data = json.load(f)
            self.__dict__.update(data)
        return True
    
    def save(self, filename):
        path = os.path.join(CONFIG_DIR, filename)
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
        print(f"Config saved to {path}")

# === Advanced event scripting ===
def check_conditions(conditions, state):
    if not conditions:
        return True
    for cond, val in conditions.items():
        if cond == "altitude_gt" and state.get('altitude',0) <= val:
            return False
        if cond == "altitude_lt" and state.get('altitude',0) >= val:
            return False
        if cond == "time_gt" and state.get('time',0) <= val:
            return False
        if cond == "time_lt" and state.get('time',0) >= val:
            return False
    return True

def run_events(events, state):
    triggered = []
    for ev in events:
        if not ev.get("_triggered", False) and state["time"] >= ev["time"]:
            if check_conditions(ev.get("condition", {}), state):
                triggered.append(ev)
                ev["_triggered"] = True
    return triggered

# === Simulation with drag and event scripting ===
def simulate_flight(config):
    print(f"\nSimulating flight for mission '{config.name}'...")
    dt = 0.05
    t = 0.0
    velocity = 0.0
    altitude = 0.0
    mass = config.rocket_mass + config.motor_params.get("mass",0) + config.payload_mass
    radius = config.rocket_diameter / 2
    area = math.pi * radius**2
    drag_coeff = config.drag_coefficient
    thrust_curve = config.motor_params.get("thrust_curve", [])
    
    data = []
    events = config.events.copy()
    
    while altitude >= 0:
        thrust = interp_thrust(thrust_curve, t)
        drag = 0.5 * AIR_DENSITY * velocity**2 * drag_coeff * area * (1 if velocity > 0 else -1)
        accel = (thrust - drag - mass * GRAVITY) / mass
        velocity += accel * dt
        altitude += velocity * dt
        if altitude < 0: altitude = 0
        t += dt
        
        state = {"time": t, "altitude": altitude, "velocity": velocity, "acceleration": accel}
        triggered_events = run_events(events, state)
        for ev in triggered_events:
            print(f"Event triggered at {t:.2f}s: {ev['type']}")
        
        data.append({"time": t, "altitude": altitude, "velocity": velocity, "acceleration": accel})
        
        if t > 300:  # Safety cutoff 5 minutes max
            print("Simulation timeout reached (5 minutes).")
            break
    
    return data

# === Plotting ===
def plot_flight(data):
    times = [d["time"] for d in data]
    altitudes = [d["altitude"] for d in data]
    velocities = [d["velocity"] for d in data]
    accels = [d["acceleration"] for d in data]
    
    plt.figure(figsize=(10,6))
    plt.subplot(311)
    plt.plot(times, altitudes, label="Altitude (m)")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(312)
    plt.plot(times, velocities, label="Velocity (m/s)", color="orange")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(313)
    plt.plot(times, accels, label="Acceleration (m/s²)", color="green")
    plt.grid(True)
    plt.legend()
    
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

# === Flight log analysis ===
def analyze_flight_log():
    print("\n-- ANALYZE FLIGHT LOG --")
    path = input("CSV log file: ").strip()
    if not os.path.isfile(path):
        print("File not found.")
        pause()
        return
    df = pd.read_csv(path)
    max_alt = df['altitude'].max()
    max_vel = df['velocity'].max() if 'velocity' in df else float('nan')
    min_alt = df['altitude'].min()
    flight_time = df['time'].max()
    avg_accel = df['acceleration'].mean() if 'acceleration' in df else float('nan')
    
    print(f"\nMax Altitude: {max_alt:.2f} m")
    print(f"Max Velocity: {max_vel:.2f} m/s")
    print(f"Flight Time: {flight_time:.2f} s")
    print(f"Average Acceleration: {avg_accel:.2f} m/s²")
    
    plot_choice = input("Plot flight data? (y/n): ").strip().lower()
    if plot_choice == 'y':
        plot_flight(df.to_dict('records'))
    pause()

# === Mission planner ===
def plan_mission():
    clear()
    print("\n-- PLAN A MISSION --")
    name = input("Mission name: ").strip()
    mc = MissionConfig()
    mc.name = name
    
    use_preset = input("Use motor preset? (y/n): ").strip().lower()
    if use_preset == 'y':
        print("Available motors:")
        for i, motor_name in enumerate(MOTOR_PRESETS.keys(), 1):
            print(f"{i}. {motor_name}")
        choice = input("> ").strip()
        try:
            selected = list(MOTOR_PRESETS.keys())[int(choice)-1]
            mc.motor = selected
            mc.motor_params = MOTOR_PRESETS[selected]
            print(f"Selected motor: {selected}")
        except:
            print("Invalid choice, using Custom.")
            mc.motor = "Custom"
            mc.motor_params = MOTOR_PRESETS["Custom"]
    else:
        # Manual motor params input
        mc.motor = "Custom"
        ti = float(input("Enter total impulse (N·s): "))
        bt = float(input("Enter burn time (s): "))
        avg_t = float(input("Enter average thrust (N): "))
        mc.motor_params = {
            "total_impulse": ti,
            "burn_time": bt,
            "avg_thrust": avg_t,
            "thrust_curve": [(0, avg_t), (bt, 0)],
            "mass": float(input("Enter motor mass (kg): "))
        }
    
    mc.rocket_mass = float(input("Rocket dry mass (kg): "))
    mc.rocket_diameter = float(input("Rocket diameter (m): "))
    mc.drag_coefficient = float(input("Drag coefficient (typical 0.4-0.8): "))
    mc.payload_mass = float(input("Payload mass (kg): "))
    
    # Event scripting
    print("\nDefine flight events (e.g. deploy parachute)")
    mc.events = []
    while True:
        etime = input("Event time (s, blank to end): ").strip()
        if etime == '':
            break
        etype = input("Event type (e.g., deploy_parachute): ").strip()
        cond_raw = input("Condition (json dict or blank): ").strip()
        cond = {}
        if cond_raw:
            try:
                cond = json.loads(cond_raw)
            except Exception as e:
                print(f"Invalid JSON: {e}")
                cond = {}
        mc.events.append({"time": float(etime), "type": etype, "condition": cond})
    
    # Save config
    fname = name.replace(" ", "_") + ".json"
    mc.save(fname)
    
    # Run simulation
    print("\nRunning flight simulation...")
    sim_data = simulate_flight(mc)
    print(f"Simulation complete, {len(sim_data)} data points.")
    
    # Save simulation to CSV
    csv_fname = os.path.join(OUTPUT_DIR, f"{name}_sim.csv")
    keys = sim_data[0].keys()
    with open(csv_fname, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(sim_data)
    print(f"Simulation data saved to {csv_fname}")
    
    # Plot result
    plot_flight(sim_data)
    pause()

# === Generate pre-flight checklist ===
def generate_checklist():
    clear()
    print("\n-- GENERATE CHECKLIST --")
    mission = input("Mission name: ").strip()
    checklist = [
        "Safety Switch ON",
        "Igniter Connected",
        "Recovery System Armed",
        "Telemetry Link Active",
        "GPS Locked",
        "Launch Pad Clear",
        "Countdown Initiated"
    ]
    filename = os.path.join(OUTPUT_DIR, f"{mission}_checklist.txt")
    with open(filename, 'w') as f:
        f.write(f"Pre-Flight Checklist: {mission}\n\n")
        for i, item in enumerate(checklist, 1):
            f.write(f"[ ] {item}\n")
    print(f"Checklist saved to {filename}")
    pause()

# === Create mission script file ===
def create_mission_script():
    clear()
    print("\n-- CREATE MISSION SCRIPT --")
    name = input("Script name: ").strip()
    events = []
    while True:
        timecode = input("Event time (s, blank to end): ").strip()
        if timecode == '':
            break
        action = input("Action: ").strip()
        events.append({"time": float(timecode), "type": action, "condition": {}})
    filename = os.path.join(OUTPUT_DIR, f"{name}_script.json")
    with open(filename, 'w') as f:
        json.dump(events, f, indent=2)
    print(f"Script saved to {filename}")
    pause()

# === Compare multiple flight logs ===
def compare_flights():
    clear()
    print("\n-- COMPARE FLIGHTS --")
    files = input("Enter CSV files separated by commas: ").strip().split(',')
    results = []
    for file in files:
        file = file.strip()
        if not os.path.isfile(file):
            print(f"File not found: {file}")
            continue
        try:
            df = pd.read_csv(file)
            max_alt = df['altitude'].max() if 'altitude' in df else float('nan')
            max_vel = df['velocity'].max() if 'velocity' in df else float('nan')
            flight_time = df['time'].max() if 'time' in df else float('nan')
            results.append((file, max_alt, max_vel, flight_time))
        except Exception as e:
            print(f"Error reading {file}: {e}")
    print("\nFlight Comparison Results:")
    print(f"{'File':30} {'Max Alt (m)':>12} {'Max Vel (m/s)':>14} {'Flight Time (s)':>15}")
    for res in results:
        print(f"{os.path.basename(res[0]):30} {res[1]:12.2f} {res[2]:14.2f} {res[3]:15.2f}")
    pause()

# === Replay telemetry log with controls ===
def replay_telemetry():
    clear()
    print("\n-- REPLAY TELEMETRY LOG --")
    path = input("CSV log file: ").strip()
    if not os.path.isfile(path):
        print("File not found.")
        pause()
        return
    df = pd.read_csv(path)
    print("Starting replay... Press Ctrl+C to stop.")
    try:
        for i, row in df.iterrows():
            print(f"T+{row['time']:.2f}s | Altitude: {row['altitude']:.2f} m | Velocity: {row.get('velocity', float('nan')):.2f} m/s")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nReplay interrupted by user.")
    pause()

# === Estimate recovery drift ===
def estimate_drift():
    clear()
    print("\n-- ESTIMATE RECOVERY DRIFT --")
    descent_rate = float(input("Descent rate (m/s): "))
    time_to_ground = float(input("Time to ground (s): "))
    wind_speed = float(input("Wind speed (m/s): "))
    drift = wind_speed * time_to_ground
    print(f"\nEstimated drift distance: {drift:.2f} meters")
    pause()

# === Generate mission report with basic stats ===
def generate_mission_report():
    clear()
    print("\n-- GENERATE MISSION REPORT --")
    name = input("Mission name: ").strip()
    notes = input("Notes or summary: ").strip()
    # Try to attach flight data if available
    sim_csv = os.path.join(OUTPUT_DIR, f"{name}_sim.csv")
    if os.path.isfile(sim_csv):
        df = pd.read_csv(sim_csv)
        max_alt = df['altitude'].max()
        max_vel = df['velocity'].max() if 'velocity' in df else float('nan')
        flight_time = df['time'].max()
    else:
        max_alt = max_vel = flight_time = float('nan')
    filename = os.path.join(OUTPUT_DIR, f"{name}__report.txt")
    with open(filename, 'w') as f:
        f.write(f"Mission Report: {name}\n")
        f.write(f"Date: {datetime.now()}\n\n")
        f.write(f"Max Altitude: {max_alt:.2f} m\n")
        f.write(f"Max Velocity: {max_vel:.2f} m/s\n")
        f.write(f"Flight Time: {flight_time:.2f} s\n\n")
        f.write("Notes:\n")
        f.write(notes + "\n")
    print(f"Report saved to {filename}")
    pause()

# === Main menu loop ===
def main_menu():
    clear()
    while True:
        print("=== ROCKET OPS MAIN MENU ===")
        print("1. Plan a Mission")
        print("2. Generate Pre-Flight Checklist")
        print("3. Simulate Flight Profile (from config)")
        print("4. Create Mission Script")
        print("5. Analyze a Flight Log")
        print("6. Compare Multiple Flights")
        print("7. Replay Telemetry Log")
        print("8. Estimate Recovery Drift")
        print("9. Generate Mission Report")
        print("0. Exit")
        choice = input("> ").strip()
        clear()
        if choice == '1':
            plan_mission()
        elif choice == '2':
            generate_checklist()
        elif choice == '3':
            configs = [f for f in os.listdir(CONFIG_DIR) if f.endswith('.json')]
            if not configs:
                print("No mission configs found. Please plan a mission first.")
                pause()
                continue
            print("Available mission configs:")
            for i, cfg in enumerate(configs, 1):
                print(f"{i}. {cfg}")
            sel = input("Select config to simulate: ").strip()
            try:
                idx = int(sel) - 1
                mc = MissionConfig()
                if mc.load(configs[idx]):
                    sim_data = simulate_flight(mc)
                    csv_fname = os.path.join(OUTPUT_DIR, f"{mc.name}_sim.csv")
                    keys = sim_data[0].keys()
                    with open(csv_fname, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=keys)
                        writer.writeheader()
                        writer.writerows(sim_data)
                    print(f"Simulation complete and saved to {csv_fname}")
                    plot_flight(sim_data)
                else:
                    print("Failed to load config.")
            except Exception:
                print("Invalid selection.")
            pause()
        elif choice == '4':
            create_mission_script()
        elif choice == '5':
            analyze_flight_log()
        elif choice == '6':
            compare_flights()
        elif choice == '7':
            replay_telemetry()
        elif choice == '8':
            estimate_drift()
        elif choice == '9':
            generate_mission_report()
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice.")
            pause()

if __name__ == "__main__":
    clear()
    main_menu()

