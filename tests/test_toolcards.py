import pytest
from pydantic import ValidationError
from environment.db import create_company_database
from environment.api import SQLiteCompanyAPI
from environment.exceptions import UnauthorizedError
from toolcards import (
    GetPurchaseOrderInput,
    CreatePurchaseOrderInput,
    get_all_tools,
    build_toolcards_prompt,
)


def test_tool_requires_reason():
    with pytest.raises(ValidationError):
        GetPurchaseOrderInput(po_id="PO-77812")


def test_tool_succeeds_with_reason():
    inp = GetPurchaseOrderInput(po_id="PO-77812", reason="Verifying promised delivery date")
    assert inp.po_id == "PO-77812"
    assert inp.reason == "Verifying promised delivery date"


def test_tool_enforces_rbac_scope():
    conn = create_company_database(":memory:", seed=True)
    # Sam Taylor (u-301) is Production Supervisor, does not have po:create scope
    api = SQLiteCompanyAPI(conn=conn, current_user_id="u-301")
    tools = {t.name: t for t in get_all_tools(api=api, user_id="u-301")}
    create_tool = tools["create_purchase_order"]

    with pytest.raises(UnauthorizedError):
        create_tool.invoke({
            "po_id": "PO-99999",
            "part_id": "P-4471",
            "supplier_id": "S-Z",
            "quantity": 100,
            "unit_price": 210.0,
            "promised_date": "2026-09-04",
            "idempotency_key": "idemp-001",
            "reason": "Test order without permissions",
        })


def test_toolcards_prompt_contains_all_tools():
    prompt = build_toolcards_prompt()
    assert "create_purchase_order" in prompt
    assert "get_purchase_order" in prompt
    assert "reason" in prompt


def test_create_purchase_order_auto_resolves_id_collision():
    conn = create_company_database(":memory:", seed=True)
    api = SQLiteCompanyAPI(conn=conn, current_user_id="u-101")
    tools = {t.name: t for t in get_all_tools(api=api, user_id="u-101")}
    create_tool = tools["create_purchase_order"]

    # PO-77813 is an existing seeded PO in the database
    existing_po = api.get_purchase_order("PO-77813")
    assert existing_po is not None

    # Invoking create_purchase_order with colliding PO-77813 auto-resolves to PO-77815
    result = create_tool.invoke({
        "po_id": "PO-77813",
        "part_id": "P-4471",
        "supplier_id": "S-Z",
        "quantity": 50,
        "unit_price": 210.0,
        "promised_date": "2026-09-04",
        "idempotency_key": "idemp-safe-001",
        "reason": "Replacement order resolving collision",
    })

    assert result["po_id"] == "PO-77815"
    assert result["status"] == "OPEN"
    # Ensure original PO-77813 was not overwritten
    old_po = api.get_purchase_order("PO-77813")
    assert old_po["part_id"] == "P-9999"
    assert old_po["status"] == "CLOSED"

