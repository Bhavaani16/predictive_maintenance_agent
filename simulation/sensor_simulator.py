"""
simulation/sensor_simulator.py
────────────────────────────────
FabGuardian – Fab Sensor Simulator

Generates realistic sensor readings for 4 machines with normal drift,
and periodically injects anomalies to trigger the AI agent.

Run standalone:  python simulation/sensor_simulator.py
Used by:         simulation/fab_server.py  (REST API backend)
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, asdict
from typing import Literal

# ---------------------------------------------------------------------------
# Machine profiles  — healthy operating ranges
# ---------------------------------------------------------------------------

MACHINES = {
    "CMP-01": {
        "type": "L",
        "air_temp_base":   299.0,
        "proc_temp_base":  309.0,
        "rpm_base":        1540,
        "torque_base":     41.5,
        "wear_rate":       0.08,   # min of wear added per tick
    },
    "LITHO-01": {
        "type": "M",
        "air_temp_base":   298.5,
        "proc_temp_base":  308.5,
        "rpm_base":        1510,
        "torque_base":     43.0,
        "wear_rate":       0.06,
    },
    "ETCH-02": {
        "type": "H",
        "air_temp_base":   300.0,
        "proc_temp_base":  310.0,
        "rpm_base":        1560,
        "torque_base":     40.0,
        "wear_rate":       0.10,
    },
    "CVD-03": {
        "type": "M",
        "air_temp_base":   299.5,
        "proc_temp_base":  309.5,
        "rpm_base":        1525,
        "torque_base":     42.0,
        "wear_rate":       0.07,
    },
}

# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------

_state: dict[str, dict] = {
    mid: {
        "tool_wear":        random.uniform(10, 60),
        "anomaly_active":   False,
        "anomaly_ticks":    0,
        "anomaly_type":     None,
        "tick":             0,
    }
    for mid in MACHINES
}


@dataclass
class SensorReading:
    machine_id: str
    product_type: Literal["L", "M", "H"]
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float
    status: Literal["NORMAL", "WARNING", "CRITICAL"]
    anomaly_type: str | None
    timestamp: float


# ---------------------------------------------------------------------------
# Anomaly injection profiles
# ---------------------------------------------------------------------------

ANOMALY_PROFILES = {
    "bearing_wear": {
        "rpm_delta":    -320,
        "torque_delta": +30,
        "temp_delta":   +5.5,
        "duration":     6,
    },
    "thermal_runaway": {
        "rpm_delta":    -80,
        "torque_delta": +12,
        "temp_delta":   +9.0,
        "duration":     5,
    },
    "tool_overload": {
        "rpm_delta":    -150,
        "torque_delta": +32,
        "temp_delta":   +3.5,
        "duration":     7,
    },
    "spindle_fault": {
        "rpm_delta":    -420,
        "torque_delta": +25,
        "temp_delta":   +6.0,
        "duration":     5,
    },
}


def _noise(scale: float = 1.0) -> float:
    return random.gauss(0, scale)


def get_reading(machine_id: str) -> SensorReading:
    """Generate one sensor reading for a machine, with possible anomaly injection."""
    profile = MACHINES[machine_id]
    state   = _state[machine_id]

    state["tick"] += 1
    state["tool_wear"] += profile["wear_rate"] + random.uniform(0, 0.05)
    state["tool_wear"]  = min(state["tool_wear"], 250.0)

    # Randomly inject anomaly (5% chance per tick, only if not already active)
    if not state["anomaly_active"] and random.random() < 0.05:
        state["anomaly_active"] = True
        state["anomaly_type"]   = random.choice(list(ANOMALY_PROFILES.keys()))
        state["anomaly_ticks"]  = 0

    # Build base readings with natural sinusoidal drift
    t = state["tick"]
    air_temp  = profile["air_temp_base"]  + 0.4 * math.sin(t * 0.1) + _noise(0.15)
    proc_temp = profile["proc_temp_base"] + 0.5 * math.sin(t * 0.1) + _noise(0.15)
    rpm       = profile["rpm_base"]       + 12  * math.sin(t * 0.07) + _noise(8)
    torque    = profile["torque_base"]    + 0.8 * math.sin(t * 0.13) + _noise(0.4)
    wear      = state["tool_wear"]

    anomaly_label = None
    status        = "NORMAL"

    # Apply anomaly deltas if active
    if state["anomaly_active"]:
        anom   = ANOMALY_PROFILES[state["anomaly_type"]]
        ramp   = min(1.0, state["anomaly_ticks"] / 2.0)   # ramp up over 2 ticks

        air_temp  += anom["temp_delta"]  * ramp * 0.4
        proc_temp += anom["temp_delta"]  * ramp
        rpm       += anom["rpm_delta"]   * ramp
        torque    += anom["torque_delta"] * ramp

        anomaly_label = state["anomaly_type"]
        state["anomaly_ticks"] += 1

        # Severity-based status
        if ramp >= 0.8:
            status = "CRITICAL"
        else:
            status = "WARNING"

        # Clear anomaly after duration
        if state["anomaly_ticks"] >= anom["duration"]:
            state["anomaly_active"] = False
            state["anomaly_type"]   = None
            state["anomaly_ticks"]  = 0

    return SensorReading(
        machine_id            = machine_id,
        product_type          = profile["type"],
        air_temperature_k     = round(air_temp,  2),
        process_temperature_k = round(proc_temp, 2),
        rotational_speed_rpm  = round(max(rpm, 800), 1),
        torque_nm             = round(max(torque, 5.0), 2),
        tool_wear_min         = round(wear, 1),
        status                = status,
        anomaly_type          = anomaly_label,
        timestamp             = time.time(),
    )


def get_all_readings() -> list[dict]:
    """Return one reading per machine as a list of dicts."""
    return [asdict(get_reading(mid)) for mid in MACHINES]


def reset_wear(machine_id: str) -> None:
    """Reset tool wear after a maintenance work order."""
    if machine_id in _state:
        _state[machine_id]["tool_wear"]      = random.uniform(5, 15)
        _state[machine_id]["anomaly_active"] = False


# ---------------------------------------------------------------------------
# CLI standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    print("Streaming sensor data (Ctrl+C to stop)...\n")
    while True:
        readings = get_all_readings()
        for r in readings:
            flag = "🔴 ANOMALY" if r["anomaly_type"] else "🟢"
            print(f"{flag}  {r['machine_id']}  |  "
                  f"rpm={r['rotational_speed_rpm']:.0f}  "
                  f"torque={r['torque_nm']:.1f}Nm  "
                  f"wear={r['tool_wear_min']:.0f}min  "
                  f"status={r['status']}")
        print()
        time.sleep(3)
