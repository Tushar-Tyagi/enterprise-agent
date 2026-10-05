import pytest
import sqlite3
from environment import (
    SQLiteCompanyAPI,
    create_company_database,
    UnauthorizedError,
    ApprovalLimitExceededError,
    EntityNotFoundError,
    ValidationError,
)


@pytest.fixture
def api():
    conn = create_company_database(":memory:", seed=True)
    return SQLiteCompanyAPI(conn=conn, current_user_id="u-101")


def test_system_clock(api):
    assert api.get_clock() == "2026-09-02"
    api.advance_clock(3)
    assert api.get_clock() == "2026-09-05"
    api.set_clock("2026-09-02")
    assert api.get_clock() == "2026-09-02"


def test_user_directory_and_permissions(api):
    user = api.get_user("u-101")
    assert user["name"] == "Dana Whitfield"
    assert user["po_create_max_value"] == 25000.0
    assert user["backup_approver_id"] == "u-102"
    assert "erp:po:create" in user["scopes"]

    # Casey Chen has quality scopes, but not po:create or po:cancel
    casey = api.get_user("u-201")
    assert "erp:quality:write" in casey["scopes"]
    assert "erp:po:create" not in casey["scopes"]
    assert "erp:po:cancel" not in casey["scopes"]


def test_permission_enforcement(api):
    # u-301 (Sam Taylor, supervisor) lacks erp:po:create
    with api.user_context("u-301"):
        with pytest.raises(UnauthorizedError) as exc_info:
            api.create_po(
                po_id="PO-TEST",
                part_id="P-4471",
                supplier_id="S-Z",
                quantity=10,
                unit_price=200.0,
                promised_date="2026-09-05",
            )
        assert "erp:po:create" in str(exc_info.value)

    # u-201 (Casey Chen, Quality) lacks erp:po:cancel
    with api.user_context("u-201"):
        with pytest.raises(UnauthorizedError) as exc_info:
            api.cancel_po("PO-77812")
        assert "erp:po:cancel" in str(exc_info.value)


def test_out_of_office_and_backup_approver(api):
    # On 2026-09-03, Dana (u-101) is OOO
    assert api.is_user_out_of_office("u-101", check_date="2026-09-03") is True
    assert api.is_user_out_of_office("u-101", check_date="2026-09-04") is True

    # Negative trap: E-003 on 2026-09-05 is Supplier Site Visit (out_of_office = false)
    assert api.is_user_out_of_office("u-101", check_date="2026-09-05") is False

    # Effective approver routing
    effective_approver_ooo = api.get_effective_approver("u-101", check_date="2026-09-03")
    assert effective_approver_ooo["user_id"] == "u-102"
    assert effective_approver_ooo["delegated_for"] == "u-101"

    # When not OOO
    effective_approver_normal = api.get_effective_approver("u-101", check_date="2026-09-02")
    assert effective_approver_normal["user_id"] == "u-101"


def test_approval_limit_enforcement(api):
    # Dana (u-101) limit is $25,000
    # $20,000 order succeeds
    po_ok = api.create_po(
        po_id="PO-OK",
        part_id="P-4471",
        supplier_id="S-Z",
        quantity=100,
        unit_price=200.0,
        promised_date="2026-09-05",
    )
    assert po_ok["po_id"] == "PO-OK"
    assert po_ok["total_amount"] == 20000.0

    # $26,000 order fails for Dana
    with pytest.raises(ApprovalLimitExceededError) as exc_info:
        api.create_po(
            po_id="PO-EXCEED",
            part_id="P-4471",
            supplier_id="S-Z",
            quantity=130,
            unit_price=200.0,
            promised_date="2026-09-05",
        )
    assert exc_info.value.backup_approver_id == "u-102"
    assert exc_info.value.amount == 26000.0

    # Alex (u-102) has limit $100,000, so $26,000 succeeds
    with api.user_context("u-102"):
        po_alex = api.create_po(
            po_id="PO-ALEX",
            part_id="P-4471",
            supplier_id="S-Z",
            quantity=130,
            unit_price=200.0,
            promised_date="2026-09-05",
        )
        assert po_alex["total_amount"] == 26000.0


def test_scenario_a_expedited_po_replacement(api):
    """
    Scenario A:
    - PO-77812 delayed to 2026-09-08 via email M-001.
    - Production order 4812 needs P-4471 by 2026-09-07.
    - Supplier traps:
      * S-X is unapproved (trap).
      * S-W has 14-day lead time (trap).
      * S-Z is approved and has 2-day lead time.
    """
    # Verify email trigger
    emails = api.get_emails()
    trigger_email = next(e for e in emails if e["mail_id"] == "M-001")
    assert "PO-77812" in trigger_email["subject"]
    assert "delayed until Tuesday 9/8" in trigger_email["body"]

    # Verify production order conflict
    prod_order = api.get_production_order("4812")
    assert prod_order["scheduled_start"] == "2026-09-07"
    assert prod_order["part_id"] == "P-4471"

    # Query suppliers for P-4471
    all_suppliers = api.query_suppliers(part_id="P-4471")
    assert len(all_suppliers) == 4

    approved_suppliers = api.query_suppliers(part_id="P-4471", approved_only=True)
    supplier_ids = [s["supplier_id"] for s in approved_suppliers]
    assert "S-X" not in supplier_ids  # S-X filtered out because approved=0
    assert "S-Z" in supplier_ids
    assert "S-Y" in supplier_ids
    assert "S-W" in supplier_ids

    # Find the only approved supplier that can deliver before 2026-09-07 (lead time <= 4 days)
    # Today is 2026-09-02; 2-day lead time delivers on 2026-09-04
    valid_alternates = [s for s in approved_suppliers if s["lead_time_days"] <= 4]
    assert len(valid_alternates) == 1
    selected_supplier = valid_alternates[0]
    assert selected_supplier["supplier_id"] == "S-Z"
    assert selected_supplier["unit_price"] == 210.0
    assert selected_supplier["part_pricing"]["P-4471"] == 210.0

    # Cancel old delayed PO
    cancelled = api.cancel_po("PO-77812")
    assert cancelled["status"] == "CANCELLED"

    # Create new PO with S-Z
    new_po = api.create_po(
        po_id="PO-77815",
        part_id="P-4471",
        supplier_id="S-Z",
        quantity=50,
        unit_price=210.0,
        promised_date="2026-09-04",
    )
    assert new_po["status"] == "OPEN"
    assert new_po["total_amount"] == 10500.0


def test_scenario_b_quality_lot_reallocation(api):
    """
    Scenario B:
    - Order 4820 consumes P-1180 on 2026-09-05.
    - Lot L-2093 is currently allocated but on hold.
    - Lot L-2095 is on hold (trap).
    - Lot L-2094 is available (good alternate).
    """
    with api.user_context("u-201"):
        # Check current lot allocated to 4820
        lots = api.get_quality_lots(part_id="P-1180")
        allocated_lot = next(l for l in lots if l["allocated_order_id"] == "4820")
        assert allocated_lot["lot_id"] == "L-2093"
        assert allocated_lot["status"] == "hold"

        # Search for available lots
        available_lots = api.get_quality_lots(part_id="P-1180", status="available")
        assert len(available_lots) == 1
        alternate_lot = available_lots[0]
        assert alternate_lot["lot_id"] == "L-2094"

        # Reallocate L-2094 to 4820
        res = api.reallocate_lot_for_order(
            order_id="4820",
            from_lot_id="L-2093",
            to_lot_id="L-2094",
        )
        assert res["status"] == "success"

        # Verify new allocation
        lot_2094 = api.get_quality_lot("L-2094")
        assert lot_2094["allocated_order_id"] == "4820"

        lot_2093 = api.get_quality_lot("L-2093")
        assert lot_2093["allocated_order_id"] is None

        # Notify production supervisor (Sam Taylor u-301)
        notif = api.notify_production(
            supervisor_id="u-301",
            order_id="4820",
            message="Reallocated inspected lot L-2094 to Order 4820. L-2093 placed on hold.",
        )
        assert notif["supervisor_id"] == "u-301"


def test_scenario_traps(api):
    """Verify that negative traps behave properly according to enterprise rules."""
    # Trap M-002 (Irrelevant delay)
    emails = api.get_emails()
    email_m002 = next(e for e in emails if e["mail_id"] == "M-002")
    assert "PO-77813" in email_m002["subject"]
    po_77813 = api.get_purchase_order("PO-77813")
    assert po_77813["status"] == "CLOSED"
    assert po_77813["part_id"] == "P-9999"

    # Verify no open production orders consume P-9999
    orders_p9999 = api.get_production_orders(part_id="P-9999")
    assert len(orders_p9999) == 0

    # Trap PO-77814 ($30,000 exceeds Dana's limit of $25,000)
    po_77814 = api.get_purchase_order("PO-77814")
    assert po_77814["total_amount"] == 30000.0
    assert po_77814["created_by"] == "u-102"  # Created by Director Alex, not Dana

    # Trap Lot L-2095 (On hold, should fail validation if attempting to reallocate)
    with api.user_context("u-201"):
        with pytest.raises(ValidationError) as exc_info:
            api.reallocate_lot_for_order(
                order_id="4820",
                from_lot_id="L-2093",
                to_lot_id="L-2095",
            )
        assert "not available" in str(exc_info.value)


def test_email_privacy_enforcement(api):
    # u-301 cannot read u-101's email directly
    with api.user_context("u-301"):
        with pytest.raises(UnauthorizedError):
            api.get_email_by_id("M-001")


def test_transaction_rollback_on_invalid_po(api):
    # Try creating PO with invalid supplier
    with pytest.raises(EntityNotFoundError):
        api.create_po(
            po_id="PO-FAIL",
            part_id="P-4471",
            supplier_id="NON_EXISTENT",
            quantity=10,
            unit_price=10.0,
            promised_date="2026-09-05",
        )

    # Verify nothing was inserted
    with pytest.raises(EntityNotFoundError):
        api.get_purchase_order("PO-FAIL")


def test_clock_driven_scheduled_events(api):
    """Verify Pattern 2: Future events in ScheduledEvents materialize only when clock advances."""
    # On 2026-09-02, M-004 (due 2026-09-03) and M-005 (due 2026-09-04) must not exist
    with pytest.raises(EntityNotFoundError):
        api.get_email_by_id("M-004")

    with api.user_context("u-201"):
        with pytest.raises(EntityNotFoundError):
            api.get_email_by_id("M-005")

    # Advance clock by 1 day to 2026-09-03
    api.advance_clock(1)
    assert api.get_clock() == "2026-09-03"

    # Now M-004 is materialized and accessible
    m004 = api.get_email_by_id("M-004")
    assert m004["mail_id"] == "M-004"
    assert m004["sender"] == "Supplier Z"
    assert "Expedited Delivery Confirmation" in m004["subject"]

    # M-005 is still not materialized
    with api.user_context("u-201"):
        with pytest.raises(EntityNotFoundError):
            api.get_email_by_id("M-005")

    # Advance clock by 1 more day to 2026-09-04
    api.advance_clock(1)
    assert api.get_clock() == "2026-09-04"

    # Now M-005 is materialized for Casey Chen (u-201)
    with api.user_context("u-201"):
        m005 = api.get_email_by_id("M-005")
        assert m005["mail_id"] == "M-005"
        assert m005["sender"] == "Quality Lab"

    # Test dynamic scheduling for a future production run
    api.schedule_event(
        trigger_date="2026-09-10",
        target_table="ProductionOrders",
        payload={
            "order_id": "5000",
            "part_id": "P-4471",
            "quantity": 80,
            "scheduled_start": "2026-09-15",
            "supervisor_id": "u-301",
            "status": "SCHEDULED",
        },
    )

    # Not visible on 2026-09-04
    with pytest.raises(EntityNotFoundError):
        api.get_production_order("5000")

    # Set clock to 2026-09-10
    api.set_clock("2026-09-10")
    order = api.get_production_order("5000")
    assert order["order_id"] == "5000"
    assert order["quantity"] == 80

