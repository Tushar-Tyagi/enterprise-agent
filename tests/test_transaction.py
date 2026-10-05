import pytest
from environment.db import create_company_database
from environment.api import SQLiteCompanyAPI
from transaction import IdempotencyRegistry, CompensationStack


def test_idempotency_caching():
    reg = IdempotencyRegistry()
    assert not reg.has_executed("key-1")
    reg.record_success("key-1", {"status": "created", "po_id": "PO-100"})
    assert reg.has_executed("key-1")
    assert reg.get_result("key-1") == {"status": "created", "po_id": "PO-100"}


def test_compensation_stack_lifo_unwind():
    conn = create_company_database(":memory:", seed=True)
    api = SQLiteCompanyAPI(conn=conn, current_user_id="u-102")
    stack = CompensationStack(api=api)

    # Action 1: Create replacement PO
    api.create_po("PO-77815", "P-4471", "S-Z", 100, 210.0, "2026-09-04", user_id="u-102")
    stack.push(
        action_name="create_po",
        compensating_callable=lambda: api.cancel_po("PO-77815", "Compensating rollback", user_id="u-102"),
        description="Void PO-77815",
    )

    # Action 2: Cancel original PO
    api.cancel_po("PO-77812", "Replacing with S-Z", user_id="u-102")
    stack.push(
        action_name="cancel_po",
        compensating_callable=lambda: api.reopen_po("PO-77812", "Compensating reopen", user_id="u-102"),
        description="Reopen PO-77812",
    )

    assert api.get_purchase_order("PO-77815")["status"] == "OPEN"
    assert api.get_purchase_order("PO-77812")["status"] == "CANCELLED"

    # Trigger rollback
    unwind_log = stack.unwind()
    assert len(unwind_log) == 2
    assert api.get_purchase_order("PO-77812")["status"] == "OPEN"
    assert api.get_purchase_order("PO-77815")["status"] == "CANCELLED"
