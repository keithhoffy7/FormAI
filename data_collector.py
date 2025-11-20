#!/usr/bin/env python3
"""
Real-time data collection script for gym lift form analysis.
Collects phone and watch accelerometer/gyroscope data and saves to CSV.

Usage:
    python data_collector.py
    
Then point your Sensor Logger app to: http://YOUR_IP:8001/data
"""

from flask import Flask, request
import json
import csv
import os
from datetime import datetime
from collections import deque
import math

# ========== CONFIGURATION ==========
# Directory where CSV files will be saved
OUTPUT_DIR = "data"  # Change this to your desired directory

# Server configuration
PORT = 8000  # Different port from main app (8000)
HOST = "0.0.0.0"

# Optional: Filter for specific watch device ID
WATCH_DEVICE_ID = os.environ.get("WATCH_DEVICE_ID")

# Units configuration
ACCEL_UNITS = (os.environ.get("ACCEL_UNITS") or "m_s2").lower()
GYRO_UNITS = (os.environ.get("GYRO_UNITS") or "rad_s").lower()
# ===================================

app = Flask(__name__)

# Data storage for current recording session
recording_data = {
    "phone_accel": {"time": [], "x": [], "y": [], "z": []},
    "phone_gyro": {"time": [], "x": [], "y": [], "z": []},
    "watch_accel": {"time": [], "x": [], "y": [], "z": []},
    "watch_gyro": {"time": [], "x": [], "y": [], "z": []}
}

is_recording = False
recording_start_time = None
current_filename = None

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _to_datetime(value):
    """Convert various timestamp magnitudes (s, ms, us, ns) to datetime."""
    if value is None:
        return None
    try:
        v = float(value)
        if v > 1e17:  # nanoseconds
            seconds = v / 1e9
        elif v > 1e14:  # microseconds
            seconds = v / 1e6
        elif v > 1e11:  # milliseconds
            seconds = v / 1e3
        else:  # seconds
            seconds = v
        return datetime.fromtimestamp(seconds)
    except Exception:
        return None


def _to_seconds(value):
    """Convert datetime or timestamp to seconds since epoch."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    try:
        v = float(value)
        if v > 1e17:  # nanoseconds
            return v / 1e9
        elif v > 1e14:  # microseconds
            return v / 1e6
        elif v > 1e11:  # milliseconds
            return v / 1e3
        else:  # seconds
            return v
    except Exception:
        return None


def _extract_xyz(values):
    """Return (x, y, z) from dict with keys or from list/tuple."""
    if values is None:
        return None
    if isinstance(values, dict):
        return values.get("x"), values.get("y"), values.get("z")
    if isinstance(values, (list, tuple)) and len(values) >= 3:
        return values[0], values[1], values[2]
    return None


def _extract_wrist_motion_accel(values):
    """Extract accelerometer data from Apple Watch wrist motion values."""
    if values is None or not isinstance(values, dict):
        return None
    return (
        values.get("accelerationX"),
        values.get("accelerationY"),
        values.get("accelerationZ")
    )


def _extract_wrist_motion_gyro(values):
    """Extract gyroscope data from Apple Watch wrist motion values."""
    if values is None or not isinstance(values, dict):
        return None
    return (
        values.get("rotationRateX"),
        values.get("rotationRateY"),
        values.get("rotationRateZ")
    )


def _convert_accel(v):
    """Convert accelerometer value to m/s²."""
    if v is None:
        return None
    if ACCEL_UNITS in ("g", "gee", "grav"):
        return float(v) * 9.80665
    return float(v)


def _convert_gyro(v):
    """Convert gyroscope value to rad/s."""
    if v is None:
        return None
    if GYRO_UNITS in ("deg_s", "deg/s", "degrees_per_second", "degrees_s"):
        return float(v) * (math.pi / 180.0)
    return float(v)


def start_recording():
    """Start a new recording session."""
    global is_recording, recording_start_time, recording_data, current_filename
    
    is_recording = True
    recording_start_time = datetime.now()
    recording_data = {
        "phone_accel": {"time": [], "x": [], "y": [], "z": []},
        "phone_gyro": {"time": [], "x": [], "y": [], "z": []},
        "watch_accel": {"time": [], "x": [], "y": [], "z": []},
        "watch_gyro": {"time": [], "x": [], "y": [], "z": []}
    }
    
    timestamp = recording_start_time.strftime("%Y%m%d_%H%M%S_%f")
    current_filename = f"sample_{timestamp}.csv"
    
    print(f"\n🔴 Recording started: {current_filename}")
    print("   Collecting data... (Press Ctrl+C to stop and save)")


def save_current_lift(lift_type=None, form_label=None, keep_recording=True):
    """Save current lift data to CSV, optionally clearing buffer and continuing recording."""
    global is_recording, current_filename, recording_data, recording_start_time
    
    if not is_recording:
        print("❌ No active recording to save")
        return False
    
    if not recording_data["phone_accel"]["time"] and not recording_data["watch_accel"]["time"]:
        print("❌ No data collected")
        return False
    
    # Determine save location
    if lift_type and form_label:
        save_dir = os.path.join(OUTPUT_DIR, lift_type, form_label)
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"sample_{timestamp}.csv"
        filepath = os.path.join(save_dir, filename)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"sample_{timestamp}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Make a copy of current data for saving
    data_to_save = {
        "phone_accel": {
            "time": recording_data["phone_accel"]["time"].copy(),
            "x": recording_data["phone_accel"]["x"].copy(),
            "y": recording_data["phone_accel"]["y"].copy(),
            "z": recording_data["phone_accel"]["z"].copy()
        },
        "phone_gyro": {
            "time": recording_data["phone_gyro"]["time"].copy(),
            "x": recording_data["phone_gyro"]["x"].copy(),
            "y": recording_data["phone_gyro"]["y"].copy(),
            "z": recording_data["phone_gyro"]["z"].copy()
        },
        "watch_accel": {
            "time": recording_data["watch_accel"]["time"].copy(),
            "x": recording_data["watch_accel"]["x"].copy(),
            "y": recording_data["watch_accel"]["y"].copy(),
            "z": recording_data["watch_accel"]["z"].copy()
        },
        "watch_gyro": {
            "time": recording_data["watch_gyro"]["time"].copy(),
            "x": recording_data["watch_gyro"]["x"].copy(),
            "y": recording_data["watch_gyro"]["y"].copy(),
            "z": recording_data["watch_gyro"]["z"].copy()
        }
    }
    
    # Combine and align data by timestamp
    success = save_combined_csv(filepath, data_to_save)
    
    if success:
        print(f"✅ Sample saved: {filepath}")
        print(f"   Phone accel: {len(data_to_save['phone_accel']['time'])} points")
        print(f"   Phone gyro:  {len(data_to_save['phone_gyro']['time'])} points")
        print(f"   Watch accel: {len(data_to_save['watch_accel']['time'])} points")
        print(f"   Watch gyro:  {len(data_to_save['watch_gyro']['time'])} points")
        
        # Clear buffer if continuing to record
        if keep_recording:
            recording_data = {
                "phone_accel": {"time": [], "x": [], "y": [], "z": []},
                "phone_gyro": {"time": [], "x": [], "y": [], "z": []},
                "watch_accel": {"time": [], "x": [], "y": [], "z": []},
                "watch_gyro": {"time": [], "x": [], "y": [], "z": []}
            }
            recording_start_time = datetime.now()
            print("   🔴 Recording continues... Ready for next lift")
    else:
        print(f"❌ Error saving file")
    
    return success


def stop_and_save_recording(lift_type=None, form_label=None):
    """Stop recording and save data to CSV."""
    global is_recording
    
    if not is_recording:
        print("❌ No active recording to save")
        return False
    
    if not lift_type or not form_label:
        print("❌ Both lift_type and form_label required to save")
        is_recording = False
        return False
    
    success = save_current_lift(lift_type, form_label, keep_recording=False)
    is_recording = False
    return success


def save_combined_csv(filepath, data):
    """Save combined phone and watch data to CSV, aligned by timestamp.
    Only saves data from the overlapping time period where both devices have data.
    """
    try:
        # Find the overlapping time period where both phone and watch have data
        phone_times = set()
        watch_times = set()
        
        # Collect phone timestamps (from any phone sensor)
        for sensor_type in ["phone_accel", "phone_gyro"]:
            phone_times.update(data[sensor_type]["time"])
        
        # Collect watch timestamps (from any watch sensor)
        for sensor_type in ["watch_accel", "watch_gyro"]:
            watch_times.update(data[sensor_type]["time"])
        
        if not phone_times or not watch_times:
            print("⚠️  Warning: Missing data from one device. Phone: {}, Watch: {}".format(
                len(phone_times), len(watch_times)))
            # Still save what we have, but warn the user
            all_times = phone_times.union(watch_times)
        else:
            # Find overlapping time period
            phone_min = min(_to_seconds(ts) for ts in phone_times)
            phone_max = max(_to_seconds(ts) for ts in phone_times)
            watch_min = min(_to_seconds(ts) for ts in watch_times)
            watch_max = max(_to_seconds(ts) for ts in watch_times)
            
            # Overlapping period
            overlap_start = max(phone_min, watch_min)
            overlap_end = min(phone_max, watch_max)
            
            if overlap_end <= overlap_start:
                print("⚠️  Warning: No overlapping time period between phone and watch data")
                all_times = phone_times.union(watch_times)
            else:
                # Only use timestamps within the overlapping period
                all_times = set()
                for ts in phone_times.union(watch_times):
                    ts_sec = _to_seconds(ts)
                    if overlap_start <= ts_sec <= overlap_end:
                        all_times.add(ts)
                
                print(f"📊 Overlapping period: {overlap_end - overlap_start:.2f} seconds")
                print(f"   Phone range: {phone_max - phone_min:.2f}s, Watch range: {watch_max - watch_min:.2f}s")
        
        if not all_times:
            return False
        
        # Convert to seconds for easier alignment
        time_to_seconds = {}
        for ts in all_times:
            time_to_seconds[ts] = _to_seconds(ts)
        
        # Sort by time
        sorted_times = sorted(all_times, key=lambda t: time_to_seconds[t])
        
        # Create lookup dictionaries for each sensor
        phone_accel_lookup = {ts: (x, y, z) for ts, x, y, z in 
                             zip(data["phone_accel"]["time"], 
                                 data["phone_accel"]["x"],
                                 data["phone_accel"]["y"],
                                 data["phone_accel"]["z"])}
        phone_gyro_lookup = {ts: (x, y, z) for ts, x, y, z in 
                            zip(data["phone_gyro"]["time"],
                                data["phone_gyro"]["x"],
                                data["phone_gyro"]["y"],
                                data["phone_gyro"]["z"])}
        watch_accel_lookup = {ts: (x, y, z) for ts, x, y, z in 
                             zip(data["watch_accel"]["time"],
                                 data["watch_accel"]["x"],
                                 data["watch_accel"]["y"],
                                 data["watch_accel"]["z"])}
        watch_gyro_lookup = {ts: (x, y, z) for ts, x, y, z in 
                            zip(data["watch_gyro"]["time"],
                                data["watch_gyro"]["x"],
                                data["watch_gyro"]["y"],
                                data["watch_gyro"]["z"])}
        
        # Write CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "timestamp",
                "phone_accel_x", "phone_accel_y", "phone_accel_z",
                "phone_gyro_x", "phone_gyro_y", "phone_gyro_z",
                "watch_accel_x", "watch_accel_y", "watch_accel_z",
                "watch_gyro_x", "watch_gyro_y", "watch_gyro_z"
            ])
            
            # Data rows
            for ts in sorted_times:
                ts_seconds = time_to_seconds[ts]
                
                # Get phone data
                phone_accel = phone_accel_lookup.get(ts, (None, None, None))
                phone_gyro = phone_gyro_lookup.get(ts, (None, None, None))
                
                # Get watch data
                watch_accel = watch_accel_lookup.get(ts, (None, None, None))
                watch_gyro = watch_gyro_lookup.get(ts, (None, None, None))
                
                writer.writerow([
                    ts_seconds,
                    phone_accel[0], phone_accel[1], phone_accel[2],
                    phone_gyro[0], phone_gyro[1], phone_gyro[2],
                    watch_accel[0], watch_accel[1], watch_accel[2],
                    watch_gyro[0], watch_gyro[1], watch_gyro[2]
                ])
        
        return True
        
    except Exception as e:
        print(f"Error saving CSV: {e}")
        return False


@app.route("/data", methods=["POST"])
def receive_data():
    """Receive sensor data from phone/watch."""
    global is_recording, recording_data
    
    if not is_recording:
        return "not recording", 200
    
    # Parse JSON
    payload_obj = request.get_json(silent=True)
    if payload_obj is None:
        try:
            payload_obj = json.loads(request.data)
        except Exception:
            return "bad request", 400
    
    # Optional device filter
    if WATCH_DEVICE_ID:
        device_id = payload_obj.get("deviceId") or payload_obj.get("device_id")
        if device_id is not None and str(device_id) != str(WATCH_DEVICE_ID):
            return "success", 200
    
    # Process payload
    for d in payload_obj.get('payload', []):
        name_raw = d.get("name") or d.get("sensor") or ""
        name = str(name_raw).lower()
        ts_raw = d.get("time") or d.get("timestamp") or d.get("ts")
        ts = _to_datetime(ts_raw)
        values_obj = d.get("values") or d.get("value") or d.get("data")
        
        if ts is None:
            continue
        
        # Handle Apple Watch wrist motion data
        if name in ("wrist motion", "wristmotion"):
            accel_xyz = _extract_wrist_motion_accel(values_obj)
            gyro_xyz = _extract_wrist_motion_gyro(values_obj)
            
            if accel_xyz is not None:
                ax, ay, az = accel_xyz
                if ax is not None and ay is not None and az is not None:
                    recording_data["watch_accel"]["time"].append(ts)
                    recording_data["watch_accel"]["x"].append(_convert_accel(ax))
                    recording_data["watch_accel"]["y"].append(_convert_accel(ay))
                    recording_data["watch_accel"]["z"].append(_convert_accel(az))
            
            if gyro_xyz is not None:
                gx, gy, gz = gyro_xyz
                if gx is not None and gy is not None and gz is not None:
                    recording_data["watch_gyro"]["time"].append(ts)
                    recording_data["watch_gyro"]["x"].append(_convert_gyro(gx))
                    recording_data["watch_gyro"]["y"].append(_convert_gyro(gy))
                    recording_data["watch_gyro"]["z"].append(_convert_gyro(gz))
        
        # Handle iPhone accelerometer and gyroscope data
        else:
            xyz = _extract_xyz(values_obj)
            if xyz is None:
                continue
            
            x, y, z = xyz
            
            if name in ("accelerometer", "accel"):
                recording_data["phone_accel"]["time"].append(ts)
                recording_data["phone_accel"]["x"].append(_convert_accel(x))
                recording_data["phone_accel"]["y"].append(_convert_accel(y))
                recording_data["phone_accel"]["z"].append(_convert_accel(z))
            
            elif name in ("gyroscope", "gyro"):
                recording_data["phone_gyro"]["time"].append(ts)
                recording_data["phone_gyro"]["x"].append(_convert_gyro(x))
                recording_data["phone_gyro"]["y"].append(_convert_gyro(y))
                recording_data["phone_gyro"]["z"].append(_convert_gyro(z))
    
    return "success", 200


@app.route("/start", methods=["POST"])
def start_endpoint():
    """API endpoint to start recording."""
    start_recording()
    return {"status": "recording", "filename": current_filename}, 200


@app.route("/stop", methods=["POST"])
def stop_endpoint():
    """API endpoint to stop recording and save."""
    lift_type = request.args.get("lift_type")
    form_label = request.args.get("form_label")
    success = stop_and_save_recording(lift_type, form_label)
    return {"status": "stopped", "saved": success}, 200


def get_local_ip():
    """Get the local IP address of this machine."""
    import socket
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            # Fallback: get hostname
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            return ip
        except Exception:
            return "localhost"


def main():
    """Main function with interactive CLI."""
    local_ip = get_local_ip()
    print("=" * 60)
    print("Gym Lift Data Collector")
    print("=" * 60)
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"\n📡 Server running on:")
    print(f"   http://{local_ip}:{PORT}/data")
    print(f"   http://localhost:{PORT}/data")
    print("\nCommands:")
    print("  'start' - Start recording session")
    print("  'save [lift_type] [form_label]' - Save current lift and continue (e.g., 'save curl good')")
    print("  'stop [lift_type] [form_label]' - Stop recording and save final lift")
    print("  'status' - Show current recording status")
    print("  'quit' - Exit")
    print("\n" + "=" * 60)
    
    import threading
    import logging
    
    # Suppress Flask's default output
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    # Start Flask server in background thread
    def run_server():
        app.run(port=PORT, host=HOST, debug=False, use_reloader=False)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Give server a moment to start
    import time
    time.sleep(0.5)
    
    # Interactive CLI
    try:
        while True:
            cmd = input("\n> ").strip().lower()
            
            if cmd == "start":
                start_recording()
            
            elif cmd.startswith("save"):
                parts = cmd.split()
                lift_type = parts[1] if len(parts) > 1 else None
                form_label = parts[2] if len(parts) > 2 else None
                if not is_recording:
                    print("❌ Not recording. Type 'start' first.")
                elif not lift_type or not form_label:
                    print("❌ Please specify both lift type and form label.")
                    print("   Usage: save <lift_type> <form_label>")
                    print("   Example: save curl good")
                else:
                    save_current_lift(lift_type, form_label, keep_recording=True)
            
            elif cmd.startswith("stop"):
                parts = cmd.split()
                lift_type = parts[1] if len(parts) > 1 else None
                form_label = parts[2] if len(parts) > 2 else None
                # Require both arguments to save
                if not lift_type or not form_label:
                    print("❌ Please specify both lift type and form label.")
                    print("   Usage: stop <lift_type> <form_label>")
                    print("   Example: stop curl good")
                    if is_recording:
                        has_data = (recording_data["phone_accel"]["time"] or 
                                   recording_data["watch_accel"]["time"])
                        if has_data:
                            print("   ⚠️  You have data collected. Use 'stop <lift_type> <form_label>' to save it.")
                else:
                    stop_and_save_recording(lift_type, form_label)
            
            elif cmd == "status":
                if is_recording:
                    count = len(recording_data["phone_accel"]["time"]) + len(recording_data["watch_accel"]["time"])
                    print(f"🔴 Recording: {count} data points collected")
                else:
                    print("⏸️  Not recording")
            
            elif cmd in ("quit", "exit", "q"):
                if is_recording:
                    response = input("Recording in progress. Stop and save? (y/n): ")
                    if response.lower() == 'y':
                        stop_and_save_recording()
                print("Goodbye!")
                break
            
            else:
                print("Unknown command. Try 'start', 'save', 'stop', 'status', or 'quit'")
    
    except KeyboardInterrupt:
        if is_recording:
            print("\n\nRecording interrupted. Saving...")
            stop_and_save_recording()
        print("\nGoodbye!")


if __name__ == "__main__":
    main()

