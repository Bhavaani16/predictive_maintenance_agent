"""
tools/work_order_manager.py
────────────────────────────
FabGuardian – Work Order Manager

Creates, retrieves, and closes structured maintenance or procurement work
orders.  Work orders are persisted in-memory during a session; in production
this module would integrate with a CMMS (e.g. IBM Maximo) or an ERP system
via its REST API.

Work Order ID format:  WO-XXXXXXXX  (WO- prefix + 8 hex digits)
Priority levels:        P1 (critical / safety) | P2 (elevated / planned)
Status values:          OPEN | IN_PROGRESS | CLOSED
"""

from __future__ import annotations

import json
import logging
import secrets
import datetime
from dataclasses import dataclass, asdict, field
from typing import Literal

from ibm_watsonx_orchestrate.agent_builder.tools import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory store  (replace with DB / CMMS integration in production)
# ---------------------------------------------------------------------------

_WORK_ORDER_STORE: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

Priority = Literal["P1", "P2"]
Status   = Literal["OPEN", "IN_PROGRESS", "CLOSED"]
WOType   = Literal["MAINTENANCE", "PROCUREMENT", "INSPECTION"]


@dataclass
class WorkOrder:
    work_order_id: str
    wo_type: WOType
    priority: Priority
    status: Status
    machine_id: str | None
    part_name: str | None
    title: str
    description: str
    created_by: str
    created_at: str
    updated_at: str
    resolution_notes: str = ""
    tags: list[str] = field(default_factory=list)


def _generate_wo_id() -> str:
    return f"WO-{secrets.token_hex(4).upper()}"


def _now_iso() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

@tool
def work_order_manager(
    action: Literal["create", "get", "update_status", "list"],
    priority: Priority | None = None,
    wo_type: WOType | None = None,
    machine_id: str | None = None,
    part_name: str | None = None,
    title: str | None = None,
    description: str | None = None,
    created_by: str = "FabGuardian-Agent",
    work_order_id: str | None = None,
    new_status: Status | None = None,
    resolution_notes: str = "",
    tags: list[str] | None = None,
) -> str:
    """
    Create, retrieve, update, or list FabGuardian work orders.

    Args:
        action:           One of "create", "get", "update_status", "list".
        priority:         "P1" (critical) or "P2" (planned).  Required for create.
        wo_type:          "MAINTENANCE", "PROCUREMENT", or "INSPECTION".
        machine_id:       Equipment ID (for maintenance/inspection orders).
        part_name:        Consumable name (for procurement orders).
        title:            Short summary of the work required.
        description:      Detailed description of the issue and required actions.
        created_by:       Originator identifier (defaults to FabGuardian-Agent).
        work_order_id:    Existing WO ID (required for get / update_status).
        new_status:       Target status for update_status action.
        resolution_notes: Closure notes when setting status to CLOSED.
        tags:             Optional list of classification tags.

    Returns:
        JSON string with the work order record or a list of records.
    """
    if action == "create":
        return _create_work_order(
            priority=priority or "P2",
            wo_type=wo_type or "MAINTENANCE",
            machine_id=machine_id,
            part_name=part_name,
            title=title or "Untitled Work Order",
            description=description or "",
            created_by=created_by,
            tags=tags or [],
        )
    elif action == "get":
        return _get_work_order(work_order_id)
    elif action == "update_status":
        return _update_status(work_order_id, new_status, resolution_notes)
    elif action == "list":
        return _list_work_orders()
    else:
        return json.dumps({"error": f"Unknown action '{action}'. Use create/get/update_status/list."})


def _create_work_order(
    priority: Priority,
    wo_type: WOType,
    machine_id: str | None,
    part_name: str | None,
    title: str,
    description: str,
    created_by: str,
    tags: list[str],
) -> str:
    wo_id = _generate_wo_id()
    now   = _now_iso()

    wo = WorkOrder(
        work_order_id=wo_id,
        wo_type=wo_type,
        priority=priority,
        status="OPEN",
        machine_id=machine_id,
        part_name=part_name,
        title=title,
        description=description,
        created_by=created_by,
        created_at=now,
        updated_at=now,
        tags=tags,
    )

    _WORK_ORDER_STORE[wo_id] = asdict(wo)
    logger.info("Created work order %s [%s / %s]", wo_id, priority, wo_type)

    response = {
        "status": "created",
        "work_order": asdict(wo),
        "message": (
            f"Work order {wo_id} created successfully with priority {priority}. "
            f"{'Dispatch field engineer immediately.' if priority == 'P1' else 'Schedule within next maintenance window.'}"
        ),
    }
    return json.dumps(response, indent=2)


def _get_work_order(work_order_id: str | None) -> str:
    if not work_order_id:
        return json.dumps({"error": "work_order_id is required for get action."})

    wo = _WORK_ORDER_STORE.get(work_order_id)
    if not wo:
        return json.dumps({"error": f"Work order '{work_order_id}' not found."})

    return json.dumps({"status": "found", "work_order": wo}, indent=2)


def _update_status(
    work_order_id: str | None,
    new_status: Status | None,
    resolution_notes: str,
) -> str:
    if not work_order_id:
        return json.dumps({"error": "work_order_id is required for update_status action."})
    if not new_status:
        return json.dumps({"error": "new_status is required for update_status action."})

    wo = _WORK_ORDER_STORE.get(work_order_id)
    if not wo:
        return json.dumps({"error": f"Work order '{work_order_id}' not found."})

    old_status = wo["status"]
    wo["status"] = new_status
    wo["updated_at"] = _now_iso()

    if new_status == "CLOSED":
        wo["resolution_notes"] = resolution_notes or "Resolved by FabGuardian-Agent."

    _WORK_ORDER_STORE[work_order_id] = wo
    logger.info("Work order %s transitioned %s → %s", work_order_id, old_status, new_status)

    return json.dumps(
        {
            "status": "updated",
            "work_order_id": work_order_id,
            "previous_status": old_status,
            "new_status": new_status,
            "work_order": wo,
        },
        indent=2,
    )


def _list_work_orders() -> str:
    orders = list(_WORK_ORDER_STORE.values())
    summary = [
        {
            "work_order_id": o["work_order_id"],
            "title":         o["title"],
            "priority":      o["priority"],
            "status":        o["status"],
            "wo_type":       o["wo_type"],
            "created_at":    o["created_at"],
        }
        for o in orders
    ]
    return json.dumps({"total": len(summary), "work_orders": summary}, indent=2)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Create P1 Maintenance Order ===")
    r1 = work_order_manager(
        action="create",
        priority="P1",
        wo_type="MAINTENANCE",
        machine_id="LITHO-01",
        title="Critical vibration anomaly – immediate inspection",
        description=(
            "IsolationForest anomaly score -0.42 detected on LITHO-01. "
            "Vibration RMS 6.8 mm/s (limit 4.5), temperature 92 °C (limit 80). "
            "Halt production run and dispatch field engineer."
        ),
        tags=["vibration", "thermal", "P1-escalation"],
    )
    print(r1)

    print("\n=== Create P2 Procurement Order ===")
    r2 = work_order_manager(
        action="create",
        priority="P2",
        wo_type="PROCUREMENT",
        part_name="Photoresist AR-EXT-248",
        title="Low inventory buffer – expedite PO",
        description="Buffer days = 3.4, below 7-day threshold. Expedite purchase order.",
        tags=["supply-chain", "photoresist"],
    )
    print(r2)

    print("\n=== List all work orders ===")
    print(work_order_manager(action="list"))
