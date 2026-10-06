import sqlite3
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from environment import SQLiteCompanyAPI, create_company_database
from detector import Detector
from agent import (
    create_scenario_a_graph,
    WORKFLOW_NAME,
    WORKFLOW_VERSION,
    WORKFLOW_STEPS,
)


def test_workflow_step_order_and_version():
    """
    Verify Part 2 requirement:
    - Definitions carry an explicit version ("1.0.0").
    - Step order is fixed by graph topology, not model improvisation:
      confirm_alternate_supplier -> confirm_lead_time -> create_new_po ->
      cancel_old_po -> notify_production -> schedule_arrival_check
    """
    assert WORKFLOW_VERSION == "1.0.0"
    assert WORKFLOW_NAME == "po_reroute_workflow"
    assert WORKFLOW_STEPS == [
        "confirm_alternate_supplier",
        "confirm_lead_time",
        "create_new_po",
        "cancel_old_po",
        "notify_production",
        "schedule_arrival_check",
    ]

    conn = create_company_database(":memory:", seed=True)
    api = SQLiteCompanyAPI(conn=conn, current_user_id="u-101")
    detector = Detector(api)
    attention = detector.scan_for_attention_items("u-101", dedupe=False)

    cp_conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(cp_conn)
    checkpointer.setup()

    app = create_scenario_a_graph(api=api, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "test-fixed-step-order"}}

    initial_state = {
        "workflow_name": WORKFLOW_NAME,
        "workflow_version": WORKFLOW_VERSION,
        "user_id": "u-101",
        "attention_item": attention,
        "context": {},
        "proposed_plan": {},
        "approval_status": "pending",
        "approver_id": "",
        "completed_steps": [],
        "compensation_stack": [],
        "step_data": {},
        "audit_trail": [],
    }

    # Reach gate and approve
    state_at_gate = app.invoke(initial_state, config)
    assert state_at_gate["approval_status"] == "pending"

    app.update_state(config, {"approval_status": "approved"}, as_node="gate")
    final_state = app.invoke(None, config)

    assert final_state["workflow_name"] == WORKFLOW_NAME
    assert final_state["workflow_version"] == WORKFLOW_VERSION
    assert final_state["completed_steps"] == WORKFLOW_STEPS
    assert final_state["approval_status"] == "executed"


def test_workflow_resumption_from_killed_step():
    """
    Verify Part 2 requirement:
    State is persisted after each step; a killed process can resume where it left off
    without duplicating previously executed mutations.
    """
    conn = create_company_database(":memory:", seed=True)
    api = SQLiteCompanyAPI(conn=conn, current_user_id="u-101")
    detector = Detector(api)
    attention = detector.scan_for_attention_items("u-101", dedupe=False)

    cp_conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(cp_conn)
    checkpointer.setup()

    app = create_scenario_a_graph(api=api, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "test-resumption-thread"}}

    initial_state = {
        "workflow_name": WORKFLOW_NAME,
        "workflow_version": WORKFLOW_VERSION,
        "user_id": "u-101",
        "attention_item": attention,
        "context": {},
        "proposed_plan": {},
        "approval_status": "pending",
        "approver_id": "",
        "completed_steps": [],
        "compensation_stack": [],
        "step_data": {},
        "audit_trail": [],
    }

    # 1. Run graph up to gate and approve
    app.invoke(initial_state, config)
    app.update_state(config, {"approval_status": "approved"}, as_node="gate")

    # 2. Execute step-by-step stream until create_new_po finishes, then simulate process kill
    executed_nodes = []
    for event in app.stream(None, config):
        for node_name in event.keys():
            executed_nodes.append(node_name)
        if "create_new_po" in executed_nodes:
            # Simulate unexpected process termination immediately after create_new_po
            break

    # Verify that at the kill point:
    # - create_new_po has executed and new PO is in DB
    new_po = api.get_purchase_order("PO-77815")
    assert new_po["status"] == "OPEN"
    # - cancel_old_po has NOT yet executed: original PO-77812 is still OPEN
    old_po = api.get_purchase_order("PO-77812")
    assert old_po["status"] == "OPEN"

    # 3. Simulate process restart: reload state from checkpointer on same thread
    # The new process resumes by invoking app.invoke(None, config)
    resumed_final_state = app.invoke(None, config)

    # 4. Verify successful completion of remaining steps without re-creating PO-77815
    assert resumed_final_state["approval_status"] == "executed"
    assert "cancel_old_po" in resumed_final_state["completed_steps"]
    assert "notify_production" in resumed_final_state["completed_steps"]
    assert "schedule_arrival_check" in resumed_final_state["completed_steps"]

    # Verify original PO is now cancelled
    assert api.get_purchase_order("PO-77812")["status"] == "CANCELLED"
    # Verify replacement PO is still OPEN and valid (not duplicated)
    assert api.get_purchase_order("PO-77815")["status"] == "OPEN"


def test_trigger_deduplication():
    """
    Verify Part 2 requirement & Section 9:
    Detector must deduplicate attention triggers across scans and persist
    deduplication state.
    """
    conn = create_company_database(":memory:", seed=True)
    api = SQLiteCompanyAPI(conn=conn, current_user_id="u-101")
    detector = Detector(api)

    # First scan should detect the critical delayed shipment
    item1 = detector.scan_for_attention_items("u-101", dedupe=True)
    assert item1 is not None
    assert item1["po_id"] == "PO-77812"
    assert "trigger_id" in item1
    trigger_id = item1["trigger_id"]

    # Verify trigger is durably marked in SQLite ProcessedTriggers table
    assert api.is_trigger_processed(trigger_id) is True

    # Second scan with dedupe=True must return None (deduplicated)
    item2 = detector.scan_for_attention_items("u-101", dedupe=True)
    assert item2 is None

    # Scan with dedupe=False should bypass deduplication
    item_bypassed = detector.scan_for_attention_items("u-101", dedupe=False)
    assert item_bypassed is not None
    assert item_bypassed["po_id"] == "PO-77812"


def test_bounded_llm_supplier_validation():
    """
    Verify bounded LLM constraint:
    If planner attempts to pick an unapproved or invalid supplier (e.g. S-X or S-BAD),
    the workflow strictly bounds the choice to approved suppliers for the part.
    """
    conn = create_company_database(":memory:", seed=True)
    api = SQLiteCompanyAPI(conn=conn, current_user_id="u-101")
    detector = Detector(api)
    attention = detector.scan_for_attention_items("u-101", dedupe=False)

    cp_conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(cp_conn)
    checkpointer.setup()

    app = create_scenario_a_graph(api=api, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "test-bounded-supplier"}}

    # Craft initial state where planner proposes an unapproved supplier 'S-X' (Apex from seed traps)
    initial_state = {
        "user_id": "u-101",
        "attention_item": attention,
        "context": {},
        "proposed_plan": {
            "alternate_supplier_id": "S-X",  # Trap unapproved supplier!
            "new_qty": 50,
            "actions": ["create_po", "cancel_po"],
        },
        "approval_status": "pending",
        "approver_id": "u-101",
        "completed_steps": [],
        "compensation_stack": [],
        "step_data": {},
        "audit_trail": [],
    }

    app.invoke(initial_state, config)
    app.update_state(config, {"approval_status": "approved"}, as_node="gate")
    final_state = app.invoke(None, config)

    # Workflow must have bounded supplier to approved S-Z, ignoring unapproved S-X
    assert final_state["step_data"]["supplier_id"] == "S-Z"
    new_po = api.get_purchase_order("PO-77815")
    assert new_po["supplier_id"] == "S-Z"
    assert new_po["status"] == "OPEN"
