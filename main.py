import argparse
import os
import sqlite3
import sys

from langgraph.checkpoint.sqlite import SqliteSaver

from agent import create_scenario_a_graph
from agent_freeform import create_freeform_agent_graph
from detector import Detector
from environment import SQLiteCompanyAPI, create_company_database


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scenario A AI Agent Runner & CLI Harness")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["deterministic", "freeform"],
        default="deterministic",
        help="Agent mode: 'deterministic' (fixed LangGraph pipeline) or 'freeform' (autonomous tool-calling loop)",
    )
    parser.add_argument(
        "--auto-approve",
        "-y",
        action="store_true",
        help="Automatically approve gating requests without prompting",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default="scenario-a-001",
        help="LangGraph checkpoint thread identifier",
    )
    parser.add_argument(
        "--allow-mock-planner",
        action="store_true",
        help="Allow deterministic fallback planner if OPENROUTER_API_KEY is not set",
    )
    return parser


def run_deterministic_mode(args, api: SQLiteCompanyAPI, attention_item: dict):
    # 4. Instantiate LangGraph App with SqliteSaver Checkpointer
    checkpoint_conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(checkpoint_conn)
    checkpointer.setup()

    app = create_scenario_a_graph(api=api, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": args.thread_id}}

    initial_state = {
        "user_id": "u-101",
        "attention_item": attention_item,
        "context": {},
        "proposed_plan": {},
        "approval_status": "pending",
        "approver_id": "",
        "audit_trail": [],
    }

    # 5. Run graph until it pauses at the gate
    print("\n--- [Step 2: Planning & Gating] Executing Graph until Gate ---")
    paused_state = app.invoke(initial_state, config)

    # Display the specific Alert to the Manager
    alert_msg = paused_state.get("alert_message") or (
        f"Part {attention_item['part_id']} will likely cause production order {attention_item['production_order_id']} "
        f"to miss its scheduled start. Supplier Y said the shipment is delayed until Tuesday. "
        f"I can move the PO to Supplier Z and notify production. Want me to proceed?"
    )
    print("\n-----------------------------------------------------------------")
    print(" AGENT ALERT TO PURCHASING MANAGER (Dana Whitfield, u-101):")
    print(f' "{alert_msg}"')
    print("-----------------------------------------------------------------")

    # 6. Print Audit Trail to console
    print("\n--- [Audit Trail at Gating Checkpoint] ---")
    for log_entry in paused_state["audit_trail"]:
        print(f" > {log_entry}")

    approver_id = paused_state.get("approver_id", "u-102")
    approval_status = paused_state.get("approval_status")

    if approval_status != "pending":
        print(f"\nGraph did not pause at gate; current status: {approval_status}")
        return

    if paused_state.get("escalated_to_backup"):
        print("\n[BUSINESS RULE TRIGGERED]")
        print("  - Approval request was unanswered by Dana Whitfield at end of day (17:00).")
        print("  - Dana's calendar shows Out of Office tomorrow (2026-09-03).")
        print("  - Action: Routed authorization to designated backup approver Alex Morgan (u-102).")

    # 7. Prompt user for approval
    prompt_text = f"\nApproval required from [{approver_id}] (Backup routed due to OOO). Proceed? (y/n): "
    if args.auto_approve:
        print(f"{prompt_text}y [Auto-Approved via flag]")
        user_choice = "y"
    else:
        user_choice = input(prompt_text).strip().lower()

    if user_choice != "y":
        print("\n[!] Execution rejected by user. Aborting workflow.")
        app.update_state(config, {"approval_status": "rejected"}, as_node="gate")
        return

    # 8. Update graph state to approved and resume execution
    print(f"\n--- [Step 3: Execution] Updating graph state to 'approved' and resuming ---")
    app.update_state(config, {"approval_status": "approved"}, as_node="gate")
    final_state = app.invoke(None, config)

    print("\n--- [Audit Trail after Execution] ---")
    start_idx = len(paused_state["audit_trail"])
    for log_entry in final_state["audit_trail"][start_idx:]:
        print(f" > {log_entry}")

    print("\nExecution Status:", final_state.get("approval_status"))


def run_freeform_mode(args, api: SQLiteCompanyAPI, attention_item: dict):
    checkpoint_conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(checkpoint_conn)
    checkpointer.setup()

    graph = create_freeform_agent_graph(api=api, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": args.thread_id}}

    current_state = {
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

    print("\n--- [Step 2: Free-Form Tool Exploration & Human Gating Loop] ---")

    while True:
        current_state = graph.invoke(current_state, config)

        # Print latest audit trail items
        if current_state.get("pending_action"):
            pending = current_state["pending_action"]
            approver_id = current_state.get("approver_id", "u-101")

            print("\n-----------------------------------------------------------------")
            print(f" [MUTATING ACTION INTERCEPTED]: `{pending['name']}`")
            print(f"  * Model Reason: \"{pending.get('reason')}\"")
            print(f"  * Arguments: {pending.get('args')}")
            print(f"  * Routing Info: {pending.get('routing_reason')}")
            print(f"  * Authorized Approver: [{approver_id}]")
            print("-----------------------------------------------------------------")

            if args.auto_approve:
                print(f"\nAuto-approving action '{pending['name']}' from [{approver_id}] via flag.")
                user_choice = "a"
            else:
                if not is_escalated:
                    prompt_text = f"\nAction requires approval from [{approver_id}]. Choose [a]ccept, [d]ecline, [w]ait: "
                else:
                    prompt_text = f"\nAction requires approval from backup approver [{approver_id}]. Choose [a]ccept, [d]ecline: "
                user_choice = input(prompt_text).strip().lower()

            if user_choice in ("w", "wait") and not is_escalated:
                print(f"\n[WAIT SELECTED]: Request unanswered by [{approver_id}] at end of day (17:00).")
                print(f"Advancing clock from {api.get_clock()} by 1 day...")
                new_clock = api.advance_clock(1)
                print(f"System Clock is now: {new_clock}")

                from langchain_core.messages import SystemMessage
                current_state["messages"].append(
                    SystemMessage(
                        content=(
                            f"[System Update: System date has advanced to {new_clock}. "
                            f"The approval request to Dana Whitfield ({approver_id}) was left unanswered at end of day yesterday. "
                            f"Dana Whitfield is now Out of Office (OOO) on PTO today. "
                            f"Route approval authority to designated backup approver Alex Morgan (u-102).]"
                        )
                    )
                )
                current_state["unanswered_at_eod"] = True
                current_state["pending_action"] = None
                current_state["approval_status"] = "none"
                continue

            elif user_choice in ("a", "accept", "y", "yes"):
                current_state["approval_status"] = "approved"
            else:
                print("\n[!] Execution rejected by user. Aborting free-form workflow.")
                break
        else:
            # Reached natural termination
            break

    print("\n--- [Audit Trail after Free-Form Execution] ---")
    for log_entry in current_state.get("audit_trail", []):
        print(f" > {log_entry}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 1. Verify OPENROUTER_API_KEY
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not args.allow_mock_planner:
        print("\n" + "=" * 65)
        print(" [CONFIGURATION ERROR] OPENROUTER_API_KEY is not set.")
        print("=" * 65)
        print(" The agent planner requires OpenRouter to evaluate context and")
        print(" generate the action plan.")
        print("\n To resolve, export your API key in your shell:")
        print("     export OPENROUTER_API_KEY='sk-or-v1-...'")
        print("\n Or run with --allow-mock-planner to run the pipeline using fallback logic:")
        print("     python3 main.py --allow-mock-planner")
        print("=" * 65 + "\n")
        sys.exit(1)

    print("=================================================================")
    print(f" Enterprise Agent Harness: Scenario A [Mode: {args.mode.upper()}]")
    print("=================================================================")

    # 2. Initialize Database & API as u-101 (Dana Whitfield)
    conn = create_company_database(":memory:", seed=True)
    api = SQLiteCompanyAPI(conn=conn, current_user_id="u-101")

    print(f"System Clock Initialized: {api.get_clock()}")
    user = api.get_user("u-101")
    print(f"Active User Context: {user['name']} ({user['title']}, ID: {user['user_id']})")
    print(f"Spending Approval Threshold: ${user['po_create_max_value']:,.2f}")
    print(f"Designated Backup Approver: {user['backup_approver_id']}")

    # 3. Run Detector to scan operational channels
    print("\n--- [Step 1: Detection] Scanning operational channels for attention items ---")
    detector = Detector(api)
    attention_item = detector.scan_for_attention_items("u-101")

    if not attention_item:
        print("[!] No attention items detected. Operational baseline stable.")
        return

    print(f"Attention Item Detected: {attention_item['type']} (Severity: {attention_item['severity']})")
    print(f"  * Mail ID: {attention_item['mail_id']}")
    print(f"  * Description: {attention_item['description']}")
    print(f"  * Impact: Delayed PO {attention_item['po_id']} breaches Production Order {attention_item['production_order_id']} start date ({attention_item['production_scheduled_start']})")

    if args.mode == "freeform":
        run_freeform_mode(args, api, attention_item)
    else:
        run_deterministic_mode(args, api, attention_item)

    # Advance Clock to simulate time passing until next Tuesday (2026-09-08)
    print("\n--- [Time Simulation] Advancing clock to next Tuesday (2026-09-08) ---")
    current_date = api.get_clock()
    print(f"Clock before advancement: {current_date}")
    from datetime import datetime
    curr_dt = datetime.strptime(current_date, "%Y-%m-%d").date()
    target_dt = datetime.strptime("2026-09-08", "%Y-%m-%d").date()
    days_to_advance = max(0, (target_dt - curr_dt).days)
    if days_to_advance > 0:
        new_date = api.advance_clock(days_to_advance)
        print(f"Clock advanced to: {new_date}")
    else:
        print(f"Clock already at: {current_date}")

    print("\nChecking Dana Whitfield's inbox on 2026-09-08 for scheduled follow-up task:")
    dana_emails = api.get_emails(recipient_id="u-101")
    follow_up_email = next((e for e in dana_emails if "M-CHECK" in e["mail_id"]), None)

    if follow_up_email:
        print(f"  [SUCCESS] Materialized Scheduled Task Email:")
        print(f"    * Mail ID: {follow_up_email['mail_id']}")
        print(f"    * From: {follow_up_email['sender']}")
        print(f"    * Subject: {follow_up_email['subject']}")
        print(f"    * Body: {follow_up_email['body']}")
        print(f"    * Sent At: {follow_up_email['sent_at']}")
    else:
        print("  [NOTE] No scheduled follow-up email in inbox.")

    print("\n=================================================================")
    print(" Scenario A Workflow Completed Successfully.")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
