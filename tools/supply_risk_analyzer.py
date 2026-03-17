"""
tools/supply_risk_analyzer.py
──────────────────────────────
FabGuardian – Supply Chain Risk Analyzer

Calculates the effective inventory buffer for critical fab consumables and
raises alerts when stock coverage falls below safe thresholds.

Buffer formula:
    effective_lead_time = lead_time_days × safety_factor
    buffer_days         = stock_days − effective_lead_time

Risk classification:
    buffer_days  < 0  → CRITICAL  (stock-out before next replenishment)
    0 ≤ buffer   < 7  → WARNING   (less than one working-week of buffer)
    buffer       ≥ 7  → HEALTHY
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from typing import Literal

from ibm_watsonx_orchestrate.agent_builder.tools import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SupplyInput:
    part_name: str
    stock_days: float       # Days of current on-hand inventory
    lead_time_days: float   # Nominal supplier lead time in days
    safety_factor: float    # Multiplier to pad lead time (e.g. 1.2 for +20 %)


@dataclass
class SupplyRiskResult:
    part_name: str
    stock_days: float
    lead_time_days: float
    safety_factor: float
    effective_lead_time: float
    buffer_days: float
    risk_level: Literal["HEALTHY", "WARNING", "CRITICAL"]
    alert_message: str
    recommended_action: str


# ---------------------------------------------------------------------------
# Risk thresholds (configurable)
# ---------------------------------------------------------------------------

WARNING_THRESHOLD_DAYS = 7    # Less than this → WARNING
CRITICAL_THRESHOLD_DAYS = 0   # Less than this → CRITICAL


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

@tool
def supply_risk_analyzer(
    part_name: str,
    stock_days: float,
    lead_time_days: float,
    safety_factor: float = 1.2,
) -> str:
    """
    Evaluate the supply chain risk for a fab consumable or spare part.

    Args:
        part_name:       Human-readable name, e.g. "CMP Slurry – ILD Grade".
        stock_days:      Current on-hand stock expressed in days of consumption.
        lead_time_days:  Supplier lead time in calendar days.
        safety_factor:   Lead-time safety multiplier (default 1.2 = +20 %).

    Returns:
        JSON string with buffer_days, risk_level, alert_message, and
        recommended_action.
    """
    supply = SupplyInput(
        part_name=part_name,
        stock_days=stock_days,
        lead_time_days=lead_time_days,
        safety_factor=safety_factor,
    )

    effective_lead_time = round(supply.lead_time_days * supply.safety_factor, 2)
    buffer_days = round(supply.stock_days - effective_lead_time, 2)

    risk_level, alert_message, recommended_action = _classify_risk(
        part_name, buffer_days, stock_days, effective_lead_time
    )

    result = SupplyRiskResult(
        part_name=supply.part_name,
        stock_days=supply.stock_days,
        lead_time_days=supply.lead_time_days,
        safety_factor=supply.safety_factor,
        effective_lead_time=effective_lead_time,
        buffer_days=buffer_days,
        risk_level=risk_level,
        alert_message=alert_message,
        recommended_action=recommended_action,
    )

    logger.info("supply_risk_analyzer result: %s", result)
    return json.dumps(asdict(result), indent=2)


def _classify_risk(
    part_name: str,
    buffer_days: float,
    stock_days: float,
    effective_lead_time: float,
) -> tuple[str, str, str]:
    """Return (risk_level, alert_message, recommended_action)."""

    if buffer_days < CRITICAL_THRESHOLD_DAYS:
        risk_level = "CRITICAL"
        alert_message = (
            f"CRITICAL: {part_name} will stock-out {abs(buffer_days):.1f} days "
            f"BEFORE the next replenishment arrives. "
            f"Stock covers only {stock_days:.1f} days; effective lead time is "
            f"{effective_lead_time:.1f} days."
        )
        recommended_action = (
            "Immediately place an emergency purchase order. "
            "Contact alternate qualified suppliers. "
            "Create a P1 work order for procurement team. "
            "Evaluate whether affected tools must be idled to conserve stock."
        )

    elif buffer_days < WARNING_THRESHOLD_DAYS:
        risk_level = "WARNING"
        alert_message = (
            f"WARNING: {part_name} buffer is critically low at {buffer_days:.1f} days "
            f"(threshold: {WARNING_THRESHOLD_DAYS} days). "
            f"Stock covers {stock_days:.1f} days; effective lead time is "
            f"{effective_lead_time:.1f} days."
        )
        recommended_action = (
            "Expedite standard purchase order immediately. "
            "Confirm delivery commitment with primary supplier. "
            "Create a P2 work order to track procurement status. "
            "Increase monitoring frequency to daily."
        )

    else:
        risk_level = "HEALTHY"
        alert_message = (
            f"HEALTHY: {part_name} has a comfortable buffer of {buffer_days:.1f} days. "
            f"Stock covers {stock_days:.1f} days; effective lead time is "
            f"{effective_lead_time:.1f} days."
        )
        recommended_action = (
            "No immediate action required. "
            "Review at next scheduled inventory cycle."
        )

    return risk_level, alert_message, recommended_action


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Healthy buffer ===")
    print(supply_risk_analyzer("CMP Slurry – ILD Grade", stock_days=45.0, lead_time_days=21.0))

    print("\n=== Warning buffer ===")
    print(supply_risk_analyzer("Photoresist AR-EXT-248", stock_days=10.0, lead_time_days=8.0))

    print("\n=== Critical stock-out ===")
    print(supply_risk_analyzer("Etch Gas – HBr Cylinders", stock_days=5.0, lead_time_days=6.0, safety_factor=1.3))
