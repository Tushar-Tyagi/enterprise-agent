import pytest
from main import build_parser


def test_cli_mode_arguments():
    parser = build_parser()
    args_default = parser.parse_args([])
    assert args_default.mode == "deterministic"

    args_freeform = parser.parse_args(["--mode", "freeform"])
    assert args_freeform.mode == "freeform"


def test_run_freeform_mode_interactive_wait_and_accept(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from environment import SQLiteCompanyAPI, create_company_database
    from main import run_freeform_mode, build_parser

    conn = create_company_database(":memory:", seed=True)
    api = SQLiteCompanyAPI(conn=conn, current_user_id="u-101")
    parser = build_parser()
    audit_file = str(tmp_path / "test_audit_1.jsonl")
    args = parser.parse_args(["--mode", "freeform", "--audit-file", audit_file])

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

    # First prompt: "w" (wait and advance clock), remaining prompts: "a" (accept)
    inputs = iter(["w", "a", "a", "a", "a"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    # Should run, advance clock to 2026-09-03, escalate to u-102, and execute
    run_freeform_mode(args, api, attention_item)
    assert api.get_clock() == "2026-09-03"
    po = api.get_purchase_order("PO-77815")
    assert po["status"] == "OPEN"
    assert po["created_by"] == "u-102"


def test_run_freeform_mode_interactive_wait_on_second_action(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from environment import SQLiteCompanyAPI, create_company_database
    from main import run_freeform_mode, build_parser

    conn = create_company_database(":memory:", seed=True)
    api = SQLiteCompanyAPI(conn=conn, current_user_id="u-101")
    parser = build_parser()
    audit_file = str(tmp_path / "test_audit_2.jsonl")
    args = parser.parse_args(["--mode", "freeform", "--audit-file", audit_file])

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

    # Action 1: "a" (accept under u-101)
    # Action 2: "w" (wait, advances clock to 2026-09-03 and escalates to u-102)
    # Action 2 retry: "a" (accept under u-102)
    # Remaining: "a"
    inputs = iter(["a", "w", "a", "a", "a"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    run_freeform_mode(args, api, attention_item)
    assert api.get_clock() == "2026-09-03"
    # Action 1 was executed under u-101
    po_77815 = api.get_purchase_order("PO-77815")
    assert po_77815["status"] == "OPEN"
    assert po_77815["created_by"] == "u-101"
    # Cancel was executed under u-102
    assert api.get_purchase_order("PO-77812")["status"] == "CANCELLED"


def test_run_deterministic_mode_interactive_wait_and_accept(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from environment import SQLiteCompanyAPI, create_company_database
    from main import run_deterministic_mode, build_parser

    conn = create_company_database(":memory:", seed=True)
    api = SQLiteCompanyAPI(conn=conn, current_user_id="u-101")
    parser = build_parser()
    audit_file = str(tmp_path / "test_audit_det.jsonl")
    args = parser.parse_args(["--mode", "deterministic", "--audit-file", audit_file])

    attention_item = {
        "type": "delayed_shipment_impacting_production",
        "po_id": "PO-77812",
        "part_id": "P-4471",
        "production_order_id": "4812",
        "supplier_id": "S-Y",
        "quantity": 50,
        "delayed_promised_date": "2026-09-08",
        "production_scheduled_start": "2026-09-07",
    }

    # First prompt: "w" (wait, advances clock to 2026-09-03, triggers OOO backup escalation)
    # Second prompt: "a" (accept proposal from backup approver u-102)
    inputs = iter(["w", "a"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    run_deterministic_mode(args, api, attention_item)
    assert api.get_clock() == "2026-09-03"

    po_77815 = api.get_purchase_order("PO-77815")
    assert po_77815["status"] == "OPEN"
    assert po_77815["created_by"] == "u-102"
    assert api.get_purchase_order("PO-77812")["status"] == "CANCELLED"


def test_run_failure_cases_execution(tmp_path):
    from main import run_failure_cases
    import json

    audit_file = str(tmp_path / "test_failure_audit.jsonl")
    run_failure_cases(audit_file=audit_file)

    with open(audit_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["mode"] == "system_failure_invariants"
    cases = {entry["case"]: entry["status"] for entry in record["results"]}
    assert cases["RBAC_PO_WRITE"] == "PASS"
    assert cases["RBAC_LOT_WRITE"] == "PASS"
    assert cases["SUPPLIER_TRAP"] == "PASS"
    assert cases["LOT_TRAP"] == "PASS"
    assert cases["APPROVAL_LIMIT_TRAP"] == "PASS"
    assert cases["TRIGGER_DEDUPE"] == "PASS"
    assert cases["SAGA_COMPENSATION"] == "PASS"
    assert cases["IDEMPOTENCY_PROTECTION"] == "PASS"


def test_run_freeform_mode_interactive_decline_triggers_rollback_and_warning(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from environment import SQLiteCompanyAPI, create_company_database
    from main import run_freeform_mode, build_parser
    import json

    conn = create_company_database(":memory:", seed=True)
    api = SQLiteCompanyAPI(conn=conn, current_user_id="u-101")
    parser = build_parser()
    audit_file = str(tmp_path / "test_audit_decline.jsonl")
    args = parser.parse_args(["--mode", "freeform", "--audit-file", audit_file])

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

    # Action 1: "a" (accept create PO-77815)
    # Action 2: "d" (decline cancel PO-77812)
    inputs = iter(["a", "d"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    run_freeform_mode(args, api, attention_item)

    # Verify audit file was written with final_status = "DECLINED_WITH_WARNING"
    with open(audit_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["final_status"] == "DECLINED_WITH_WARNING"

    # Verify audit trail contains Gate Check [Declined] and Agent Warning
    audit_text = " ".join(record["audit_trail"])
    assert "Gate Check [Declined]" in audit_text
    assert "Agent Deliberation & Warning" in audit_text or "Agent Warning" in audit_text

    # Verify rollback: Action 1 (PO-77815) was rolled back / voided due to decline of Action 2
    po_77815 = api.get_purchase_order("PO-77815")
    assert po_77815["status"] == "CANCELLED"  # Voided by compensation

