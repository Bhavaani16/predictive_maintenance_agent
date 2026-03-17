"""
tools/failure_predictor.py
──────────────────────────
FabGuardian – Predictive Maintenance Tool

Zero external ML dependencies — uses only numpy (always available).

Anomaly detection approach:
  - Mahalanobis distance from healthy baseline mean/covariance
  - Rule-based risk thresholds derived from UCI Predictive Maintenance data
  - Failure probability estimated via sigmoid of normalised distance
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np
from ibm_watsonx_orchestrate.agent_builder.tools import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Healthy baseline statistics (derived from UCI dataset, healthy samples only)
# Features: [air_temp_K, proc_temp_K, rpm, torque_Nm, tool_wear_min]
# ---------------------------------------------------------------------------

_HEALTHY_MEAN = np.array([299.4, 309.6, 1537.0, 41.8, 107.0])
_HEALTHY_STD  = np.array([  0.9,   0.9,   55.0,   2.5,  58.0])

# Operational hard limits  (lo, hi)
_LIMITS = {
    "air_temperature_K":     (295.0, 304.0),
    "process_temperature_K": (305.0, 313.0),
    "rotational_speed_rpm":  (1000.0, 2500.0),
    "torque_Nm":             (  3.0,  75.0),
    "tool_wear_min":         (  0.0, 250.0),
}

# Risk thresholds on normalised Mahalanobis distance
_HIGH_DIST   = 4.5
_MEDIUM_DIST = 2.5

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    machine_id: str
    mahalanobis_distance: float
    failure_probability: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    triggered_features: list[str]
    recommendation: str


# ---------------------------------------------------------------------------
# Core math (pure numpy — no sklearn)
# ---------------------------------------------------------------------------

def _mahalanobis(x: np.ndarray) -> float:
    """Simplified Mahalanobis using diagonal covariance (std per feature)."""
    z = (x - _HEALTHY_MEAN) / (_HEALTHY_STD + 1e-9)
    return float(np.sqrt(np.dot(z, z)))


def _failure_prob(dist: float) -> float:
    """Sigmoid mapping of Mahalanobis distance to [0, 1] failure probability."""
    return round(1.0 / (1.0 + math.exp(-0.8 * (dist - 3.5))), 4)


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

@tool
def failure_predictor(
    machine_id: str,
    air_temperature_k: float,
    process_temperature_k: float,
    rotational_speed_rpm: float,
    torque_nm: float,
    tool_wear_min: float,
    product_type: Literal["L", "M", "H"] = "M",
) -> str:
    """
    Analyse sensor readings for a piece of fab equipment and return a risk assessment.

    Args:
        machine_id:               Equipment identifier, e.g. "CMP-01" or "LITHO-01".
        air_temperature_k:        Ambient temperature in Kelvin (e.g. 298.1).
        process_temperature_k:    Process temperature in Kelvin (e.g. 308.6).
        rotational_speed_rpm:     Spindle speed in RPM (e.g. 1500).
        torque_nm:                Torque in Newton-metres (e.g. 42.8).
        tool_wear_min:            Cumulative tool usage in minutes (e.g. 120).
        product_type:             Quality tier – "L", "M", or "H".

    Returns:
        JSON string with mahalanobis_distance, failure_probability, risk_level,
        triggered_features, and recommendation.
    """
    x = np.array([
        air_temperature_k,
        process_temperature_k,
        rotational_speed_rpm,
        torque_nm,
        tool_wear_min,
    ])

    dist      = _mahalanobis(x)
    fail_prob = _failure_prob(dist)

    # Feature-level limit checking
    values = {
        "air_temperature_K":     air_temperature_k,
        "process_temperature_K": process_temperature_k,
        "rotational_speed_rpm":  rotational_speed_rpm,
        "torque_Nm":             torque_nm,
        "tool_wear_min":         tool_wear_min,
    }
    triggered = [
        f for f, (lo, hi) in _LIMITS.items()
        if values[f] < lo or values[f] > hi
    ]

    # Risk classification
    if dist >= _HIGH_DIST or len(triggered) >= 2:
        risk_level = "HIGH"
    elif dist >= _MEDIUM_DIST or len(triggered) == 1:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Override: extremely high torque or tool wear always HIGH
    if torque_nm > 70.0 or tool_wear_min > 220.0:
        risk_level = "HIGH"

    if risk_level == "HIGH":
        rec = (
            f"IMMEDIATE ACTION for {machine_id}. "
            f"Failure probability: {fail_prob*100:.1f}% | Distance from healthy baseline: {dist:.2f}. "
            f"Out-of-bounds features: {triggered}. "
            "Halt production run, dispatch field engineer, create P1 work order."
        )
    elif risk_level == "MEDIUM":
        rec = (
            f"Schedule inspection for {machine_id} within 24 h. "
            f"Failure probability: {fail_prob*100:.1f}% | Distance from healthy baseline: {dist:.2f}. "
            f"Elevated features: {triggered}. "
            "Create P2 work order and increase monitoring frequency."
        )
    else:
        rec = (
            f"{machine_id} operating within normal parameters. "
            f"Failure probability: {fail_prob*100:.1f}% | Distance from healthy baseline: {dist:.2f}. "
            "No action required."
        )

    result = PredictionResult(
        machine_id=machine_id,
        mahalanobis_distance=round(dist, 4),
        failure_probability=fail_prob,
        risk_level=risk_level,
        triggered_features=triggered,
        recommendation=rec,
    )
    return json.dumps(asdict(result), indent=2)


if __name__ == "__main__":
    print("=== Healthy (CMP-01) ===")
    print(failure_predictor("CMP-01", 298.1, 308.7, 1551, 42.8, 0, "M"))
    print("\n=== Degraded (LITHO-01) ===")
    print(failure_predictor("LITHO-01", 301.0, 309.0, 1200, 72.0, 230, "L"))