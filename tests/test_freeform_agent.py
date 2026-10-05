import sqlite3
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from environment.db import create_company_database
from environment.api import SQLiteCompanyAPI
from agent_freeform import create_freeform_agent_graph


def test_freeform_graph_compilation_and_gating(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    conn = create_company_database(":memory:", seed=True)
    api = SQLiteCompanyAPI(conn=conn, current_user_id="u-101")
    cp_conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(cp_conn)
    checkpointer.setup()

    graph = create_freeform_agent_graph(api=api, checkpointer=checkpointer)
    assert graph is not None

    attention_item = {
        "type": "delayed_shipment_impacting_production",
        "po_id": "PO-77812",
        "part_id": "P-4471",
        "production_order_id": "4812",
        "supplier_id": "S-Y",
        "quantity": 100,
        "delayed_promised_date": "2026-09-08",
        "production_scheduled_start": "2026-09-07",
    }

    initial_state = {
        "messages": [],
        "user_id": "u-101",
        "attention_item": attention_item,
        "pending_action": None,
        "approval_status": "none",
        "approver_id": "u-101",
        "primary_approver_id": "u-101",
        "escalated_to_backup": False,
        "idempotency_records": {},
        "compensation_stack": [],
        "audit_trail": [],
        "llm_telemetry": {},
    }

    config = {"configurable": {"thread_id": "test-ff-001"}}
    res = graph.invoke(initial_state, config)
    assert "messages" in res
    assert len(res["audit_trail"]) > 0


def test_freeform_mutating_interception_and_resume(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    conn = create_company_database(":memory:", seed=True)
    api = SQLiteCompanyAPI(conn=conn, current_user_id="u-101")
    cp_conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(cp_conn)
    checkpointer.setup()

    graph = create_freeform_agent_graph(api=api, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "test-ff-002"}}

    attention_item = {
        "type": "delayed_shipment_impacting_production",
        "po_id": "PO-77812",
        "part_id": "P-4471",
        "production_order_id": "4812",
        "supplier_id": "S-Y",
        "quantity": 100,
        "delayed_promised_date": "2026-09-08",
        "production_scheduled_start": "2026-09-07",
    }

    initial_state = {
        "messages": [],
        "user_id": "u-101",
        "attention_item": attention_item,
        "pending_action": None,
        "approval_status": "none",
        "approver_id": "u-101",
        "primary_approver_id": "u-101",
        "escalated_to_backup": False,
        "idempotency_records": {},
        "compensation_stack": [],
        "audit_trail": [],
        "llm_telemetry": {},
    }

    # Step 1: Runs and pauses at mutating gate for create_purchase_order
    state_after_gate = graph.invoke(initial_state, config)
    assert state_after_gate["approval_status"] == "pending"
    assert state_after_gate["escalated_to_backup"] is True
    assert state_after_gate["approver_id"] == "u-102"  # Alex Morgan
    assert state_after_gate["pending_action"] is not None
    assert state_after_gate["pending_action"]["name"] == "create_purchase_order"
    assert "reason" in state_after_gate["pending_action"]["args"]

    # Step 2: Human approves create_purchase_order and resumes
    resume_state = dict(state_after_gate)
    resume_state["approval_status"] = "approved"
    state_after_create = graph.invoke(resume_state, config)

    # Verify replacement PO created and now paused at cancel_purchase_order gate
    replacement_po = api.get_purchase_order("PO-77815")
    assert replacement_po["status"] == "OPEN"
    assert state_after_create["approval_status"] == "pending"
    assert state_after_create["pending_action"]["name"] == "cancel_purchase_order"

    # Step 3: Human approves cancel_purchase_order and resumes
    resume_state_2 = dict(state_after_create)
    resume_state_2["approval_status"] = "approved"
    state_after_cancel = graph.invoke(resume_state_2, config)

    # Verify original PO cancelled
    original_po = api.get_purchase_order("PO-77812")
    assert original_po["status"] == "CANCELLED"


def test_freeform_idempotency_protection(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    conn = create_company_database(":memory:", seed=True)
    api = SQLiteCompanyAPI(conn=conn, current_user_id="u-101")
    cp_conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(cp_conn)
    checkpointer.setup()

    graph = create_freeform_agent_graph(api=api, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "test-ff-idemp"}}

    attention_item = {
        "type": "delayed_shipment_impacting_production",
        "po_id": "PO-77812",
        "part_id": "P-4471",
        "production_order_id": "4812",
        "supplier_id": "S-Y",
        "quantity": 100,
        "delayed_promised_date": "2026-09-08",
        "production_scheduled_start": "2026-09-07",
    }

    initial_state = {
        "messages": [],
        "user_id": "u-101",
        "attention_item": attention_item,
        "pending_action": None,
        "approval_status": "none",
        "approver_id": "u-101",
        "primary_approver_id": "u-101",
        "escalated_to_backup": False,
        "idempotency_records": {},
        "compensation_stack": [],
        "audit_trail": [],
        "llm_telemetry": {},
    }

    # First execution to create PO
    s1 = graph.invoke(initial_state, config)
    s1["approval_status"] = "approved"
    s2 = graph.invoke(s1, config)

    # Records should contain the idempotency key for PO creation
    assert "idemp-po-77815" in s2["idempotency_records"]
    assert s2["idempotency_records"]["idemp-po-77815"]["status"] == "SUCCESS"


def test_freeform_compensation_unwinding_on_error():
    conn = create_company_database(":memory:", seed=True)
    api = SQLiteCompanyAPI(conn=conn, current_user_id="u-102")
    from transaction import CompensationStack, IdempotencyRegistry

    idemp = IdempotencyRegistry()
    stack = CompensationStack(api=api, idempotency_registry=idemp)

    # 1. Simulate replacement PO creation
    api.create_po("PO-77815", "P-4471", "S-Z", 100, 210.0, "2026-09-04", user_id="u-102")
    idemp.record_success("idemp-po-77815", {"po_id": "PO-77815"})
    stack.push(
        action_name="cancel_po",
        compensating_callable=lambda: api.cancel_po("PO-77815", "Compensation rollback", user_id="u-102"),
        description="Void created PO-77815",
        idempotency_key="idemp-po-77815",
    )

    # 2. Simulate original PO cancellation
    api.cancel_po("PO-77812", reason="Replacing with S-Z", user_id="u-102")
    idemp.record_success("idemp-cancel-77812", {"po_id": "PO-77812"})
    stack.push(
        action_name="reopen_po",
        compensating_callable=lambda: api.reopen_po("PO-77812", "Compensation rollback", user_id="u-102"),
        description="Reopen cancelled PO-77812",
        idempotency_key="idemp-cancel-77812",
    )

    # 3. Simulate failure in downstream step 3 (e.g. notify_production fails)
    log = stack.unwind()
    assert len(log) == 2
    assert all(item["status"] == "COMPENSATED" for item in log)

    # Check both database states rolled back
    assert api.get_purchase_order("PO-77812")["status"] == "OPEN"
    assert api.get_purchase_order("PO-77815")["status"] == "CANCELLED"
    # Check idempotency records marked compensated
    assert idemp._records["idemp-po-77815"]["status"] == "COMPENSATED"
    assert idemp._records["idemp-cancel-77812"]["status"] == "COMPENSATED"

