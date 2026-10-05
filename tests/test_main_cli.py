import pytest
from main import build_parser


def test_cli_mode_arguments():
    parser = build_parser()
    args_default = parser.parse_args([])
    assert args_default.mode == "deterministic"

    args_freeform = parser.parse_args(["--mode", "freeform"])
    assert args_freeform.mode == "freeform"


def test_run_freeform_mode_interactive_wait_and_accept(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from environment import SQLiteCompanyAPI, create_company_database
    from main import run_freeform_mode, build_parser

    conn = create_company_database(":memory:", seed=True)
    api = SQLiteCompanyAPI(conn=conn, current_user_id="u-101")
    parser = build_parser()
    args = parser.parse_args(["--mode", "freeform"])

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
    assert api.get_purchase_order("PO-77815")["status"] == "OPEN"

