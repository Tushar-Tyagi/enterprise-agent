"""
Tests for Scenario B Context Gathering and Persona Scoping.
Verifies that Casey Chen (u-201) receives scoped context, quality lots (including trap lots),
and strictly filtered authorized tools (excluding PO creation/cancellation).
"""

import pytest
from environment.db import create_company_database
from environment.api import SQLiteCompanyAPI
from context_quality import gather_quality_context
from toolcards import get_tools_for_user, build_toolcards_prompt


@pytest.fixture
def api():
    conn = create_company_database(":memory:", seed=True)
    return SQLiteCompanyAPI(conn=conn, current_user_id="u-201")


def test_gather_quality_context_scoping_and_inventory(api):
    attention = {
        "type": "quality_hold_shortage_risk",
        "severity": "CRITICAL",
        "order_id": "4820",
        "part_id": "P-1180",
        "allocated_lot": "L-2093",
    }

    ctx = gather_quality_context(api, attention, user_id="u-201")

    # Persona & Clock verification
    assert ctx["user_id"] == "u-201"
    assert ctx["user_name"] == "Casey Chen"
    assert ctx["user_title"] == "Quality Manager"
    assert ctx["clock_date"] == "2026-09-02"

    # Production Order constraints
    assert ctx["order_id"] == "4820"
    assert ctx["part_id"] == "P-1180"
    assert ctx["scheduled_start"] == "2026-09-05"
    assert ctx["supervisor_id"] == "u-301"
    assert ctx["supervisor_name"] == "Sam Taylor"

    # Lot inventory & Trap detection
    assert ctx["current_lot"]["lot_id"] == "L-2093"
    assert ctx["current_lot"]["status"] == "hold"

    # Exactly 1 valid candidate lot (L-2094)
    available_lot_ids = [l["lot_id"] for l in ctx["available_lots"]]
    assert "L-2094" in available_lot_ids

    # Trap lot (L-2095) is identified as on hold
    hold_lot_ids = [l["lot_id"] for l in ctx["hold_lots"]]
    assert "L-2095" in hold_lot_ids

    assert ctx["has_covering_lot"] is True
    assert ctx["recommended_lot"] == "L-2094"


def test_quality_manager_tool_authorization_filtering(api):
    # Retrieve Casey Chen's authorized tools
    tools = get_tools_for_user(api, "u-201")
    tool_names = {t.name for t in tools}

    # Authorized tools must be present
    assert "reallocate_lot_for_order" in tool_names
    assert "get_quality_lots" in tool_names
    assert "get_quality_lot" in tool_names
    assert "notify_production" in tool_names
    assert "send_email" in tool_names
    assert "read_emails" in tool_names

    # Unauthorized tools MUST NOT be in Casey's toolset
    assert "create_purchase_order" not in tool_names
    assert "cancel_purchase_order" not in tool_names

    # Filtered toolcards prompt does not expose PO creation/cancellation
    prompt = build_toolcards_prompt(allowed_tools=list(tool_names))
    assert "reallocate_lot_for_order" in prompt
    assert "create_purchase_order" not in prompt
    assert "cancel_purchase_order" not in prompt
