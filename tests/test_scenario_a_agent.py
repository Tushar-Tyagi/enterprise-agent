import sqlite3
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from environment import SQLiteCompanyAPI, create_company_database
from detector import Detector
from agent import create_scenario_a_graph


def test_scenario_a_agent_full_lifecycle():
    # 1. Initialize DB and API as Dana Whitfield (u-101)
    conn = create_company_database(":memory:", seed=True)
    api = SQLiteCompanyAPI(conn=conn, current_user_id="u-101")

    # 2. Run detector
    detector = Detector(api)
    attention = detector.scan_for_attention_items("u-101")
    assert attention is not None
    assert attention["po_id"] == "PO-77812"
    assert attention["production_order_id"] == "4812"
    assert attention["severity"] == "CRITICAL"

    # 3. Create checkpointer and graph
    cp_conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(cp_conn)
    checkpointer.setup()

    app = create_scenario_a_graph(api=api, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "test-scenario-a"}}

    initial_state = {
        "user_id": "u-101",
        "attention_item": attention,
        "context": {},
        "proposed_plan": {},
        "approval_status": "pending",
        "approver_id": "",
        "audit_trail": [],
    }

    # 4. Invoke graph up to the gate
    state_at_gate = app.invoke(initial_state, config)
    assert state_at_gate["approval_status"] == "pending"
    assert state_at_gate["approver_id"] == "u-102"  # Routed to Alex Morgan due to Dana's OOO
    assert state_at_gate["proposed_plan"]["alternate_supplier_id"] == "S-Z"

    # Verify audit trail contains context, plan, and gate check
    audit_text = " ".join(state_at_gate["audit_trail"])
    assert "Context Gathered" in audit_text
    assert "Planner Output" in audit_text
    assert "Gate Check" in audit_text
    assert "Out of Office tomorrow" in audit_text

    # 5. Human approval and resumption
    app.update_state(config, {"approval_status": "approved"}, as_node="gate")
    final_state = app.invoke(None, config)
    assert final_state["approval_status"] == "executed"

    # 6. Verify database side effects
    new_po = api.get_purchase_order("PO-77815")
    assert new_po["status"] == "OPEN"
    assert new_po["supplier_id"] == "S-Z"
    assert new_po["part_id"] == "P-4471"

    old_po = api.get_purchase_order("PO-77812")
    assert old_po["status"] == "CANCELLED"

    # 7. Advance clock to next Tuesday (2026-09-08)
    api.advance_clock(6)
    assert api.get_clock() == "2026-09-08"

    # Verify Tuesday check event fired
    dana_emails = api.get_emails(recipient_id="u-101")
    check_email = next((e for e in dana_emails if "M-CHECK-PO-77815" in e["mail_id"]), None)
    assert check_email is not None
    assert "Verify Arrival" in check_email["subject"]


def test_compensation_rollback_restores_original_po(monkeypatch):
    """
    Verify that if a downstream action (e.g. notify or schedule) fails:
    1. Replacement PO-77815 is voided (CANCELLED).
    2. Original PO-77812 is restored back to OPEN.
    """
    conn = create_company_database(":memory:", seed=True)
    api = SQLiteCompanyAPI(conn=conn, current_user_id="u-101")
    detector = Detector(api)
    attention = detector.scan_for_attention_items("u-101")

    cp_conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(cp_conn)
    checkpointer.setup()

    app = create_scenario_a_graph(api=api, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "test-compensation-failure"}}

    initial_state = {
        "user_id": "u-101",
        "attention_item": attention,
        "context": {},
        "proposed_plan": {},
        "approval_status": "pending",
        "approver_id": "",
        "audit_trail": [],
    }

    # Reach the gate and approve
    app.invoke(initial_state, config)
    app.update_state(config, {"approval_status": "approved"}, as_node="gate")

    # Inject failure into schedule_event (simulating downstream failure after PO cancellation)
    def failing_schedule_event(*args, **kwargs):
        raise RuntimeError("Downstream queuing failure during arrival check staging")

    monkeypatch.setattr(api, "schedule_event", failing_schedule_event)

    # Resume graph execution and verify it raises the error
    with pytest.raises(RuntimeError) as exc_info:
        app.invoke(None, config)
    assert "Downstream queuing failure" in str(exc_info.value)

    # Verify Compensation Rollback:
    # 1. Original PO-77812 must be restored to OPEN!
    original_po = api.get_purchase_order("PO-77812")
    assert original_po["status"] == "OPEN"

    # 2. Replacement PO-77815 must be voided (CANCELLED)!
    replacement_po = api.get_purchase_order("PO-77815")
    assert replacement_po["status"] == "CANCELLED"

