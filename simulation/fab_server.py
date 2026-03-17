"""
simulation/fab_server.py
─────────────────────────
FabGuardian – Simulation Backend Server

Flask REST API that:
  1. Streams live sensor readings from the simulator every 3 seconds
  2. Detects anomalies and automatically calls the watsonx Orchestrate agent
  3. Returns agent responses to the dashboard UI via Server-Sent Events (SSE)

Usage:
    cd FabGuardian
    python simulation/fab_server.py

Then open:  http://localhost:5001
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, send_from_directory
from flask_cors import CORS

# Add project root to path so we can import the simulator
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from simulation.sensor_simulator import get_all_readings, reset_wear, MACHINES
from tools.failure_predictor import failure_predictor
from tools.work_order_manager import work_order_manager

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

POLL_INTERVAL_SECONDS = 3       # How often to read sensors
AGENT_COOLDOWN_SECONDS = 15     # Min seconds between agent calls per machine

# Watsonx Orchestrate REST endpoint
ORCHESTRATE_URL = os.getenv(
    "WATSONX_URL",
    "https://api.us-east-1.watson-orchestrate.cloud.ibm.com"
).rstrip("/instances/") if "/instances/" in os.getenv("WATSONX_URL", "") \
    else os.getenv("WATSONX_URL", "https://api.us-east-1.watson-orchestrate.cloud.ibm.com")

INSTANCE_ID  = os.getenv("WATSONX_URL", "").split("/instances/")[-1] \
               if "/instances/" in os.getenv("WATSONX_URL", "") else ""
API_KEY      = os.getenv("WATSONX_APIKEY", "")
AGENT_NAME   = "FabGuardian_Orchestrator"

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder="static")
CORS(app)

_event_queue: queue.Queue = queue.Queue(maxsize=200)
_sensor_history: dict[str, list] = {mid: [] for mid in MACHINES}
_agent_last_called: dict[str, float] = {mid: 0.0 for mid in MACHINES}
_agent_responses: list[dict] = []
_work_orders: list[dict] = []


# ---------------------------------------------------------------------------
# Watsonx Orchestrate agent caller
# ---------------------------------------------------------------------------

def _unwrap(result) -> str:
    """Extract plain string from a ToolResponse or plain str."""
    if isinstance(result, str):
        return result
    # ToolResponse object — try common attribute names
    for attr in ("content", "text", "output", "result", "value"):
        val = getattr(result, attr, None)
        if val is not None:
            return str(val)
    # Last resort — convert to string
    return str(result)


def _call_agent(machine_id: str, reading: dict, anomaly_type: str) -> str:
    """
    Invoke the FabGuardian tools directly in Python.
    This is the most reliable approach — identical logic to the deployed agent,
    no auth or API dependency, instant response.
    """
    try:
        # Step 1: Run failure predictor
        fp_result = json.loads(_unwrap(failure_predictor(
            machine_id            = machine_id,
            air_temperature_k     = reading["air_temperature_k"],
            process_temperature_k = reading["process_temperature_k"],
            rotational_speed_rpm  = reading["rotational_speed_rpm"],
            torque_nm             = reading["torque_nm"],
            tool_wear_min         = reading["tool_wear_min"],
            product_type          = reading["product_type"],
        )))

        risk_level   = fp_result["risk_level"]
        fail_prob    = fp_result["failure_probability"]
        triggered    = fp_result.get("triggered_features", [])
        rec          = fp_result.get("recommendation", "")

        # Step 2: Create work order if MEDIUM or HIGH risk
        wo_id = None
        wo_msg = ""
        if risk_level in ("HIGH", "MEDIUM"):
            priority = "P1" if risk_level == "HIGH" else "P2"
            wo_result = json.loads(_unwrap(work_order_manager(
                action      = "create",
                priority    = priority,
                wo_type     = "MAINTENANCE",
                machine_id  = machine_id,
                title       = f"{priority} – {anomaly_type.replace('_',' ').title()} on {machine_id}",
                description = (
                    f"Anomaly type: {anomaly_type}. "
                    f"Failure probability: {fail_prob*100:.1f}%. "
                    f"Triggered features: {triggered}. "
                    f"Sensor snapshot: rpm={reading['rotational_speed_rpm']:.0f}, "
                    f"torque={reading['torque_nm']:.1f}Nm, "
                    f"wear={reading['tool_wear_min']:.0f}min."
                ),
                tags        = [anomaly_type, risk_level.lower(), machine_id],
            )))
            wo_id  = wo_result["work_order"]["work_order_id"]
            wo_msg = f"\n🔧 Work Order Created: {wo_id} [{priority}]"

        # Format the agent-style response
        icon = "🔴" if risk_level == "HIGH" else "🟡" if risk_level == "MEDIUM" else "🟢"
        response = (
            f"{icon} Risk Level: {risk_level}\n"
            f"📊 Failure Probability: {fail_prob*100:.1f}%\n"
            f"⚠️  Anomaly: {anomaly_type.replace('_',' ').upper()}\n"
            f"🔬 Machine: {machine_id}\n"
            f"\n{rec}"
            f"{wo_msg}"
        )
        return response

    except Exception as e:
        import traceback
        print(f"[Agent] Direct tool error: {traceback.format_exc()}")
        return f"[Tool execution error: {e}]"


# ---------------------------------------------------------------------------
# Background polling loop
# ---------------------------------------------------------------------------

def _polling_loop() -> None:
    """Runs in a background thread — reads sensors and triggers agent on anomaly."""
    while True:
        try:
            readings = get_all_readings()
            now = time.time()

            for r in readings:
                mid = r["machine_id"]

                # Keep rolling history (last 40 readings per machine)
                _sensor_history[mid].append(r)
                if len(_sensor_history[mid]) > 40:
                    _sensor_history[mid].pop(0)

                # Push sensor update event to SSE queue
                _event_queue.put(json.dumps({
                    "type":    "sensor",
                    "payload": r,
                }))

                # Trigger agent if anomaly detected and cooldown has passed
                if (r["anomaly_type"]
                        and r["status"] in ("WARNING", "CRITICAL")
                        and (now - _agent_last_called[mid]) > AGENT_COOLDOWN_SECONDS):

                    _agent_last_called[mid] = now

                    # Call agent in a separate thread so it doesn't block the poll loop
                    threading.Thread(
                        target=_handle_agent_call,
                        args=(mid, r, r["anomaly_type"]),
                        daemon=True,
                    ).start()

        except Exception as e:
            print(f"[Polling error] {e}")

        time.sleep(POLL_INTERVAL_SECONDS)


def _handle_agent_call(machine_id: str, reading: dict, anomaly_type: str) -> None:
    """Call the agent and push the response to the SSE queue."""

    # Notify dashboard that agent is thinking
    _event_queue.put(json.dumps({
        "type": "agent_thinking",
        "payload": {
            "machine_id":   machine_id,
            "anomaly_type": anomaly_type,
            "status":       reading["status"],
            "timestamp":    reading["timestamp"],
        },
    }))

    response_text = _call_agent(machine_id, reading, anomaly_type)

    entry = {
        "machine_id":    machine_id,
        "anomaly_type":  anomaly_type,
        "status":        reading["status"],
        "sensor_snapshot": {
            "air_temperature_k":     reading["air_temperature_k"],
            "process_temperature_k": reading["process_temperature_k"],
            "rotational_speed_rpm":  reading["rotational_speed_rpm"],
            "torque_nm":             reading["torque_nm"],
            "tool_wear_min":         reading["tool_wear_min"],
        },
        "agent_response": response_text,
        "timestamp":      reading["timestamp"],
    }
    _agent_responses.append(entry)

    # Push agent response event to SSE queue
    _event_queue.put(json.dumps({
        "type":    "agent_response",
        "payload": entry,
    }))


# ---------------------------------------------------------------------------
# REST API routes
# ---------------------------------------------------------------------------

@app.route("/api/sensors")
def api_sensors():
    """Return latest reading for each machine."""
    latest = {
        mid: hist[-1] if hist else {}
        for mid, hist in _sensor_history.items()
    }
    return jsonify(latest)


@app.route("/api/history/<machine_id>")
def api_history(machine_id: str):
    """Return sensor history for a specific machine."""
    return jsonify(_sensor_history.get(machine_id, []))


@app.route("/api/agent-responses")
def api_agent_responses():
    """Return all agent responses so far."""
    return jsonify(_agent_responses)


@app.route("/api/reset-wear/<machine_id>", methods=["POST"])
def api_reset_wear(machine_id: str):
    """Simulate a maintenance action — resets tool wear."""
    reset_wear(machine_id)
    return jsonify({"status": "ok", "machine_id": machine_id, "message": "Tool wear reset."})


@app.route("/stream")
def stream():
    """Server-Sent Events endpoint — dashboard subscribes here for live updates."""
    def generate():
        yield "data: {\"type\": \"connected\"}\n\n"
        while True:
            try:
                event = _event_queue.get(timeout=20)
                yield f"data: {event}\n\n"
            except queue.Empty:
                yield "data: {\"type\": \"heartbeat\"}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/")
def index():
    return send_from_directory(
        str(Path(__file__).parent / "static"), "index.html"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Start background sensor polling thread
    t = threading.Thread(target=_polling_loop, daemon=True)
    t.start()

    print("=" * 60)
    print("  FabGuardian Simulation Server")
    print("=" * 60)
    print(f"  Dashboard:    http://localhost:5001")
    print(f"  Sensor API:   http://localhost:5001/api/sensors")
    print(f"  SSE Stream:   http://localhost:5001/stream")
    print(f"  Agent:        {AGENT_NAME}")
    print(f"  Machines:     {', '.join(MACHINES.keys())}")
    print("=" * 60)

    if not API_KEY:
        print("\n  ⚠️  WATSONX_APIKEY not set — agent calls will fail.")
        print("     Run: export WATSONX_APIKEY='your-key-here'")
    if not INSTANCE_ID:
        print("\n  ⚠️  WATSONX_URL not set or missing instance ID.")
        print("     Run: export WATSONX_URL='https://api.us-east-1.watson-orchestrate.cloud.ibm.com/instances/YOUR-ID'")

    print("\n  Sensor polling every 3 seconds. Open the dashboard to begin.\n")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
