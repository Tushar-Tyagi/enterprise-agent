"""
End-to-End Tests for Scenario B: Quality Hold & Lot Reallocation in Free-Form Agent.
Verifies the complete lifecycle:
- Unprompted detection of lot on quality hold impacting Production Order 4820.
- Autonomous investigation of candidate lots (rejecting hold trap L-2095).
- Gating interception of reallocate_lot_for_order under Casey Chen (u-201).
- Execution of reallocation and notification to supervisor Sam Taylor (u-301).
- Saga compensation unwinding for lot reallocations.
- Fallback branch: Shortage alert email to Purchasing Manager when no good lot covers it.
"""

import sqlite3
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from environment.db import create_company_database
from environment.api import SQLiteCompanyAPI
from detector import Detector
from agent_freeform import create_freeform_agent_graph


@pytest.fixture
def api():
    conn = create_company_database(":memory:", seed=True)
    return SQLiteCompanyAPI(conn=conn, current_user_id="u-201")


def test_scenario_b_detection_and_full_lifecycle(api, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    # 1. Detection
    detector = Detector(api)
    trigger = detector.scan_for_quality_hold_shortages(horizon_days=5)
    assert trigger is not None
    assert trigger["type"] == "quality_hold_impacting_production"
    assert trigger["production_order_id"] == "4820"
    assert trigger["lot_id"] == "L-2093"

    # 2. Graph Initialization
    cp_conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(cp_conn)
    checkpointer.setup()

    graph = create_freeform_agent_graph(api=api, checkpointer=checkpointer)

    initial_state = {
        "messages": [],
        "user_id": "u-201",
        "attention_item": trigger,
        "pending_tool_calls": [],
        "pending_action": None,
        "approval_status": "none",
        "approver_id": "u-201",
        "primary_approver_id": "u-201",
        "escalated_to_backup": False,
        "idempotency_records": {},
        "compensation_stack": [],
        "audit_trail": [],
        "llm_telemetry": {},
    }

    config = {"configurable": {"thread_id": "thread-scen-b-001"}}

    # Turn 1: Inspects lots and pauses at Gating Checkpoint for reallocate_lot_for_order
    state_after_gate_1 = graph.invoke(initial_state, config)
    assert state_after_gate_1["approval_status"] == "pending"
    pending_1 = state_after_gate_1["pending_action"]
    assert pending_1["name"] == "reallocate_lot_for_order"
    assert pending_1["args"]["from_lot_id"] == "L-2093"
    assert pending_1["args"]["to_lot_id"] == "L-2094"
    assert state_after_gate_1["approver_id"] == "u-201"

    # Turn 2: Quality Manager Casey Chen approves reallocation -> Pauses at Gate for notify_production
    approved_state_1 = dict(state_after_gate_1)
    approved_state_1["approval_status"] = "approved"

    state_after_gate_2 = graph.invoke(approved_state_1, config)
    assert state_after_gate_2["approval_status"] == "pending"
    pending_2 = state_after_gate_2["pending_action"]
    assert pending_2["name"] == "notify_production"
    assert pending_2["args"]["supervisor_id"] == "u-301"
    assert pending_2["args"]["order_id"] == "4820"

    # Turn 3: Casey Chen approves notification -> Completes workflow
    approved_state_2 = dict(state_after_gate_2)
    approved_state_2["approval_status"] = "approved"

    final_state = graph.invoke(approved_state_2, config)
    assert final_state["approval_status"] == "none"

    # Verify ERP DB State
    lot_2094 = api.get_quality_lot("L-2094")
    assert lot_2094["allocated_order_id"] == "4820"

    lot_2093 = api.get_quality_lot("L-2093")
    assert lot_2093["allocated_order_id"] is None

    notifications = api.get_production_notifications("4820")
    assert len(notifications) == 1
    assert notifications[0]["supervisor_id"] == "u-301"


def test_scenario_b_shortage_fallback_when_no_covering_lot(api, monkeypatch):
    """
    When candidate lot L-2094 is also placed on hold, no lot covers Order 4820.
    The agent must autonomously propose flagging a shortage via email to Purchasing Manager u-101.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    # Place L-2094 on hold as well
    api.update_quality_lot("L-2094", status="hold", hold_reason="Calibration drift")

    detector = Detector(api)
    trigger = detector.scan_for_quality_hold_shortages(horizon_days=5)
    assert trigger is not None
    assert trigger["covered_by_alternate"] is False

    cp_conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(cp_conn)
    checkpointer.setup()

    graph = create_freeform_agent_graph(api=api, checkpointer=checkpointer)

    initial_state = {
        "messages": [],
        "user_id": "u-201",
        "attention_item": trigger,
        "pending_tool_calls": [],
        "pending_action": None,
        "approval_status": "none",
        "approver_id": "u-201",
        "primary_approver_id": "u-201",
        "escalated_to_backup": False,
        "idempotency_records": {},
        "compensation_stack": [],
        "audit_trail": [],
        "llm_telemetry": {},
    }

    config = {"configurable": {"thread_id": "thread-scen-b-shortage-001"}}

    # Turn 1: Agent inspects lots, finds NO available covering lot, gates send_email to u-101
    state_after_gate = graph.invoke(initial_state, config)
    assert state_after_gate["approval_status"] == "pending"
    pending = state_after_gate["pending_action"]
    assert pending["name"] == "send_email"
    assert pending["args"]["recipient_id"] == "u-101"
    assert "Shortage" in pending["args"]["subject"]

    # Turn 2: Approve shortage alert email
    approved_state = dict(state_after_gate)
    approved_state["approval_status"] = "approved"

    final_state = graph.invoke(approved_state, config)
    assert final_state["approval_status"] == "none"

    # Verify email was received by Dana Whitfield (u-101)
    emails = api.get_emails(recipient_id="u-101")
    shortage_mail = next(e for e in emails if "Shortage" in e["subject"])
    assert shortage_mail is not None
    assert "4820" in shortage_mail["body"]


def test_scenario_b_compensation_rollback(api, monkeypatch):
    """
    Verifies Saga compensation unwinding for lot reallocations:
    if an error occurs after lot reallocation, the original hold lot is restored back to the order.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    trigger = {
        "type": "quality_hold_shortage_risk",
        "order_id": "4820",
        "part_id": "P-1180",
        "allocated_lot": "L-2093",
        "hold_reason": "Surface finish",
    }

    cp_conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(cp_conn)
    checkpointer.setup()

    graph = create_freeform_agent_graph(api=api, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "thread-scen-b-comp-001"}}

    # 1. Gate 1: Reallocate lot
    s1 = graph.invoke({
        "messages": [],
        "user_id": "u-201",
        "attention_item": trigger,
        "pending_tool_calls": [],
        "pending_action": None,
        "approval_status": "none",
        "approver_id": "u-201",
        "primary_approver_id": "u-201",
        "escalated_to_backup": False,
        "idempotency_records": {},
        "compensation_stack": [],
        "audit_trail": [],
        "llm_telemetry": {},
    }, config)

    # Approve reallocation
    s1_approved = dict(s1)
    s1_approved["approval_status"] = "approved"

    # 2. Gate 2: Pauses at notify_production
    s2 = graph.invoke(s1_approved, config)
    assert s2["pending_action"]["name"] == "notify_production"

    # Confirm L-2094 is currently allocated before failure
    assert api.get_quality_lot("L-2094")["allocated_order_id"] == "4820"

    # Inject an intentional failure during notify_production execution by tampering with order_id
    tampered_state = dict(s2)
    tampered_state["approval_status"] = "approved"
    tampered_state["pending_action"]["args"]["order_id"] = "NONEXISTENT_ORDER_9999"

    # Resume graph -> Execution fails, unwinding compensation stack
    res = graph.invoke(tampered_state, config)

    # Assert compensation restored L-2093 back to order 4820 and released L-2094
    lot_2093 = api.get_quality_lot("L-2093")
    assert lot_2093["allocated_order_id"] == "4820"

    lot_2094 = api.get_quality_lot("L-2094")
    assert lot_2094["allocated_order_id"] is None
