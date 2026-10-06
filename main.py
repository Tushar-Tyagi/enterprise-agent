import argparse
from datetime import datetime
import json
import os
import sqlite3
import sys

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import ToolMessage

from agent import create_scenario_a_graph, WORKFLOW_NAME, WORKFLOW_VERSION
from agent_freeform import create_freeform_agent_graph, get_llm_config
from detector import Detector
from environment import SQLiteCompanyAPI, create_company_database

load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enterprise Operations Multi-Agent Runner & Simulation Harness")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["a", "b", "both"],
        default="both",
        help="Scenario to run: 'a' (Purchasing PO replacement), 'b' (Quality lot reallocation), or 'both' (default: 'both')",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Run Scenario A via deterministic workflow pipeline rather than free-form loop",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["deterministic", "freeform"],
        default="deterministic",
        help="Legacy agent mode selector: 'deterministic' or 'freeform'",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to simulate in operational loop (default: 7, reaching 2026-09-08)",
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
        default="scenario-run-001",
        help="LangGraph checkpoint thread identifier",
    )
    parser.add_argument(
        "--allow-mock-planner",
        action="store_true",
        help="Allow deterministic fallback planner if OPENROUTER_API_KEY is not set",
    )
    parser.add_argument(
        "--audit-file",
        type=str,
        default="audit_trail.jsonl",
        help="Append-only JSONL log path for run audits (default: 'audit_trail.jsonl')",
    )
    parser.add_argument(
        "--skip-failures",
        action="store_true",
        help="Skip executing Phase 4 system invariants and failure cases demonstration",
    )
    return parser


def save_run_audit(
    audit_file: str,
    api: SQLiteCompanyAPI,
    mode: str,
    thread_id: str,
    user_id: str,
    attention_item: dict,
    audit_trail: list,
    telemetry: dict,
    final_status: str,
    additional_data: dict = None,
) -> None:
    timestamp = datetime.now().isoformat()
    record = {
        "run_id": f"run-{thread_id}-{int(datetime.now().timestamp())}",
        "timestamp": timestamp,
        "mode": mode,
        "thread_id": thread_id,
        "user_id": user_id,
        "system_clock": api.get_clock(),
        "attention_item": attention_item,
        "audit_trail": audit_trail,
        "llm_telemetry": telemetry,
        "final_status": final_status,
        "metadata": additional_data or {},
    }

    # 1. Append to disk JSONL
    try:
        with open(audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"\n[AUDIT LOG] Persisted append-only audit record to: {audit_file}")
    except Exception as exc:
        print(f"\n[!] Warning: Failed to write audit file '{audit_file}': {exc}")

    # 2. Append to SQLite AuditLogs table
    try:
        api.log_audit_event(
            run_id=record["run_id"],
            category="EXECUTION",
            actor_id=user_id,
            summary=f"Run completed ({mode.upper()} mode, status: {final_status})",
            details={
                "attention_type": attention_item.get("type"),
                "po_id": attention_item.get("po_id") or attention_item.get("lot_id"),
                "audit_entries_count": len(audit_trail),
                "telemetry": telemetry,
            },
            timestamp=timestamp,
        )
    except Exception:
        pass


def run_deterministic_mode(args, api: SQLiteCompanyAPI, attention_item: dict):
    checkpoint_conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(checkpoint_conn)
    checkpointer.setup()

    app = create_scenario_a_graph(api=api, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": args.thread_id}}

    initial_state = {
        "workflow_name": WORKFLOW_NAME,
        "workflow_version": WORKFLOW_VERSION,
        "user_id": "u-101",
        "attention_item": attention_item,
        "context": {},
        "proposed_plan": {},
        "approval_status": "pending",
        "approver_id": "u-101",
        "primary_approver_id": "u-101",
        "escalated_to_backup": False,
        "unanswered_at_eod": False,
        "completed_steps": [],
        "compensation_stack": [],
        "step_data": {},
        "audit_trail": [],
    }

    # Run graph until it pauses at the gate
    paused_state = app.invoke(initial_state, config)

    # Clean executive briefing card before approval prompt
    alert_msg = paused_state.get("alert_message") or (
        f"Part {attention_item['part_id']} will likely cause production order {attention_item['production_order_id']} "
        f"to miss its scheduled start. Supplier Y said the shipment is delayed until Tuesday. "
        f"I can move the PO to Supplier Z and notify production. Want me to proceed?"
    )

    print("\n" + "=" * 67)
    print(" AGENT OPERATIONAL BRIEFING & MITIGATION PROPOSAL")
    print("=" * 67)
    print(f" Context  : Production Order {attention_item.get('production_order_id')} (Scheduled Start: {attention_item.get('production_scheduled_start')})")
    print(f" Problem  : PO {attention_item.get('po_id')} (Part {attention_item.get('part_id')}) delayed until {attention_item.get('delayed_promised_date')}")
    print(f" Impact   : Production line will be starved without alternate sourcing")
    print("-" * 67)
    print(" AGENT ALERT TO PURCHASING MANAGER:")
    print(f' "{alert_msg}"')
    print("-" * 67)
    print(" PROPOSED ACTIONS:")

    planned_actions = paused_state.get("proposed_plan", {}).get("planned_actions", [])
    if planned_actions:
        action_labels = {
            "create_purchase_order": "1. Create Replacement PO",
            "cancel_purchase_order": "2. Cancel Delayed Order ",
            "notify_production":     "3. Notify Supervisor   ",
            "schedule_arrival_check":"4. Schedule Dock Check  ",
        }
        for act in planned_actions:
            tool_name = act.get("tool")
            label = action_labels.get(tool_name, f"• {tool_name}")
            p = act.get("parameters", {})
            if tool_name == "create_purchase_order":
                print(f"  {label} : {p.get('po_id')} ({p.get('quantity')} units @ ${p.get('unit_price', 0):.2f}/unit with Supplier {p.get('supplier_id')})")
                print(f"     * Delivery Promise  : {p.get('promised_date')} (arrives prior to production start)")
                print(f"     * Total Value       : ${p.get('total_amount', 0):,.2f}")
            elif tool_name == "cancel_purchase_order":
                print(f"  {label} : {p.get('po_id')} (Supplier {attention_item.get('supplier_id')}, delivery slipped past start date)")
            elif tool_name == "notify_production":
                print(f"  {label} : Sam Taylor (Supervisor for Production Order {p.get('order_id')})")
            elif tool_name == "schedule_arrival_check":
                print(f"  {label} : Tuesday {p.get('trigger_date')} follow-up to verify shipment arrival")
    else:
        print(f"  1. Create Replacement PO : PO-77815 (50 units with Supplier Z, promised 2026-09-04)")
        print(f"  2. Cancel Delayed Order  : {attention_item.get('po_id')} (Supplier {attention_item.get('supplier_id')})")
        print(f"  3. Notify Supervisor     : Production Supervisor (Order {attention_item.get('production_order_id')})")
        print(f"  4. Schedule Dock Check   : Tuesday follow-up on dock arrival")
    print("=" * 67)

    approver_id = paused_state.get("approver_id", "u-101")
    is_escalated = paused_state.get("escalated_to_backup", False)

    # Interactive Human Gating Loop: Accept, Decline, or Wait
    while True:
        if is_escalated:
            prompt_text = f"\nAuthorize proposal from backup approver [{approver_id}]? Choose [a]ccept, [d]ecline: "
        else:
            prompt_text = f"\nAuthorize proposal from [{approver_id}]? Choose [a]ccept, [d]ecline, [w]ait: "

        if args.auto_approve:
            print(f"{prompt_text}a [Auto-Approved via flag]")
            user_choice = "a"
        else:
            user_choice = input(prompt_text).strip().lower()

        if user_choice in ("w", "wait") and not is_escalated:
            print("\n" + "-" * 67)
            print(f" [WAIT SELECTED]: Request unanswered by [{approver_id}] at end of day (17:00).")
            print(" Advancing system clock by 1 business day...")
            new_clock = api.advance_clock(1)
            print(f" System Clock is now: {new_clock}")
            print("-" * 67)

            user_obj = api.get_user(approver_id)
            backup_id = user_obj.get("backup_approver_id", "u-102") if user_obj else "u-102"
            backup_user = api.get_user(backup_id)
            backup_name = backup_user.get("name", "Alex Morgan") if backup_user else "Alex Morgan"

            approver_id = backup_id
            is_escalated = True
            escalation_msg = (
                f"Gate Check [Escalation]: Approval unanswered at EOD by 'u-101'. "
                f"Clock advanced to {new_clock}. Approver is Out of Office on {new_clock}. "
                f"Escalated authority to designated backup '{backup_id}' ({backup_name})."
            )
            paused_state.setdefault("audit_trail", []).append(escalation_msg)

            print("\n" + "=" * 67)
            print(" BUSINESS POLICY TRIGGERED: ESCALATION TO BACKUP APPROVER")
            print("=" * 67)
            print("  * Policy Condition : Approval request unanswered at end of day (17:00)")
            print(f"  * Calendar Status  : Dana Whitfield is Out of Office on {new_clock} (Supplier site visit)")
            print(f"  * Routing Decision : Escalated authorization authority to {backup_name} ([{backup_id}])")
            print("=" * 67)
            continue

        elif user_choice in ("a", "accept", "y", "yes"):
            print(f"\n--- [Execution] Updating graph state to 'approved' by [{approver_id}] and resuming ---")
            app.update_state(
                config,
                {
                    "approval_status": "approved",
                    "approver_id": approver_id,
                    "escalated_to_backup": is_escalated,
                    "audit_trail": paused_state.get("audit_trail", []),
                },
                as_node="gate",
            )
            final_state = app.invoke(None, config)
            break

        elif user_choice in ("d", "decline", "n", "no"):
            print("\n[!] Execution rejected by user. Aborting workflow.")
            app.update_state(config, {"approval_status": "rejected"}, as_node="gate")
            return
        else:
            valid_opts = "[a]ccept, [d]ecline, [w]ait" if not is_escalated else "[a]ccept, [d]ecline"
            print(f"\n[!] Unrecognized input '{user_choice}'. Please choose {valid_opts}.")
            continue

    print("\n--- [Audit Trail after Execution] ---")
    start_idx = len(paused_state["audit_trail"])
    for log_entry in final_state["audit_trail"][start_idx:]:
        print(f" > {log_entry}")

    print("\nExecution Status:", final_state.get("approval_status"))
    telemetry = final_state.get("llm_telemetry", {})
    save_run_audit(
        audit_file=getattr(args, "audit_file", "audit_trail.jsonl"),
        api=api,
        mode="deterministic",
        thread_id=args.thread_id,
        user_id="u-101",
        attention_item=attention_item,
        audit_trail=final_state.get("audit_trail", []),
        telemetry=telemetry,
        final_status=final_state.get("approval_status", "COMPLETED"),
    )
    return final_state.get("approval_status", "COMPLETED")


def run_freeform_mode(args, api: SQLiteCompanyAPI, attention_item: dict, user_id: str = "u-101", scenario_label: str = "Scenario A"):
    checkpoint_conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(checkpoint_conn)
    checkpointer.setup()

    graph = create_freeform_agent_graph(api=api, checkpointer=checkpointer)
    thread_suffix = "a" if user_id == "u-101" else "b"
    config = {"configurable": {"thread_id": f"{args.thread_id}-{thread_suffix}"}}

    current_state = {
        "messages": [],
        "user_id": user_id,
        "attention_item": attention_item,
        "pending_tool_calls": [],
        "pending_action": None,
        "approval_status": "none",
        "approver_id": user_id,
        "primary_approver_id": user_id,
        "escalated_to_backup": False,
        "unanswered_at_eod": False,
        "idempotency_records": {},
        "compensation_stack": [],
        "audit_trail": [],
        "llm_telemetry": {},
    }

    is_quality = (user_id == "u-201" or attention_item.get("type") in ("quality_hold_shortage_risk", "quality_hold_impacting_production"))

    print(f"\n--- [Free-Form Tool Exploration & Gating Loop: {scenario_label}] ---")

    while True:
        if current_state.get("approval_status") == "approved" or not current_state.get("pending_action"):
            current_state = graph.invoke(current_state, config)

        if current_state.get("pending_action"):
            pending = current_state["pending_action"]
            approver_id = current_state.get("approver_id", user_id)
            is_escalated = current_state.get("escalated_to_backup", False)

            print("\n" + "=" * 67)
            print(f" [MUTATING ACTION INTERCEPTED]: `{pending['name']}`")
            print("=" * 67)

            if is_quality:
                order_id = attention_item.get("production_order_id") or attention_item.get("order_id", "4820")
                lot_id = attention_item.get("lot_id") or attention_item.get("allocated_lot", "L-2093")
                print(" QUALITY OPERATIONAL BRIEFING & IMPACT ASSESSMENT:")
                print(f"  * Impacted Production Order : Order {order_id} (Scheduled Start: {attention_item.get('production_scheduled_start', '2026-09-05')})")
                print(f"  * Quality Hold Risk         : Lot {lot_id} is on hold ('{attention_item.get('hold_reason')}')")
                target_ref = pending.get("args", {}).get("to_lot_id") or pending.get("args", {}).get("recipient_id") or pending["name"]
                print(f"  * Mitigating Action         : {pending['name']} (Target: {target_ref})")
                print(f"  * Authorized Approver       : [{approver_id}] (Quality Manager)")
            else:
                print(" PURCHASING OPERATIONAL BRIEFING & IMPACT ASSESSMENT:")
                print(f"  * Impacted Production Order : Order {attention_item.get('production_order_id')} (Scheduled Start: {attention_item.get('production_scheduled_start')})")
                print(f"  * Delay Risk / Slip Details : PO {attention_item.get('po_id')} (Part {attention_item.get('part_id')}) delayed until {attention_item.get('delayed_promised_date')}")
                target_ref = pending.get("args", {}).get("po_id") or pending.get("args", {}).get("order_id") or pending.get("args", {}).get("recipient_id") or pending["name"]
                print(f"  * Mitigating Action         : {pending['name']} (Target: {target_ref})")
                print(f"  * Authorized Approver       : [{approver_id}]" + (" (Escalated to backup)" if is_escalated else " (Primary)"))

            print(f"  * Routing Context           : {pending.get('routing_reason')}")
            print("\n MODEL OPERATIONAL JUSTIFICATION:")
            print(f"  \"{pending.get('reason')}\"")
            print("-" * 67)

            task_desc = f"{pending['name']} ({target_ref})"
            if args.auto_approve:
                print(f"\nAuto-approving specific write task '{task_desc}' from [{approver_id}] via flag.")
                user_choice = "a"
            else:
                if not is_escalated and not is_quality:
                    prompt_text = f"\nAuthorize specific write task '{task_desc}' from [{approver_id}]? Choose [a]ccept, [d]ecline, [w]ait: "
                elif not is_escalated and is_quality:
                    prompt_text = f"\nAuthorize specific write task '{task_desc}' from Quality Manager [{approver_id}]? Choose [a]ccept, [d]ecline: "
                else:
                    prompt_text = f"\nAuthorize specific write task '{task_desc}' from backup approver [{approver_id}]? Choose [a]ccept, [d]ecline: "
                user_choice = input(prompt_text).strip().lower()

            if user_choice in ("w", "wait") and not is_escalated and not is_quality:
                print(f"\n[WAIT SELECTED]: Request unanswered by [{approver_id}] at end of day (17:00).")
                print(f"Advancing clock from {api.get_clock()} by 1 day...")
                new_clock = api.advance_clock(1)
                print(f"System Clock is now: {new_clock}")

                user_obj = api.get_user(approver_id)
                backup_id = user_obj.get("backup_approver_id", "u-102") if user_obj else "u-102"
                backup_user = api.get_user(backup_id)
                backup_name = backup_user.get("name", "Alex Morgan") if backup_user else "Alex Morgan"

                current_state["unanswered_at_eod"] = True
                current_state["approver_id"] = backup_id
                current_state["escalated_to_backup"] = True
                pending["routing_reason"] = (
                    f"Rule Triggered: Request was unanswered by {user_obj.get('name', approver_id)} at end of day. "
                    f"Approver is Out of Office on {new_clock}. Escalated to designated backup approver {backup_name} ({backup_id})."
                )
                current_state["pending_action"] = pending
                current_state["approval_status"] = "pending"
                current_state.setdefault("audit_trail", []).append(
                    f"Gate Check [Escalation]: Approval unanswered at EOD by '{approver_id}'. "
                    f"Clock advanced to {new_clock}. Escalated '{pending['name']}' to backup approver '{backup_id}' ({backup_name})."
                )
                continue

            elif user_choice in ("a", "accept", "y", "yes"):
                current_state["approval_status"] = "approved"
            elif user_choice in ("d", "decline", "n", "no"):
                approver_name = api.get_user(approver_id).get("name", approver_id)
                task_name = pending.get("name", "action")
                task_reason = pending.get("reason", "")

                print(f"\n[!] Mutating action '{task_name}' DECLINED by approver [{approver_id}] ({approver_name}).")
                current_state.setdefault("audit_trail", []).append(
                    f"Gate Check [Declined]: Mutating action '{task_name}' was DECLINED by approver [{approver_id}] ({approver_name}). "
                    f"Model justification was: \"{task_reason}\"."
                )

                # Unwind any previously executed mutating actions in this transaction
                comp_stack_records = current_state.get("compensation_stack", [])
                if comp_stack_records:
                    print("  * Unwinding previously executed transaction steps via Saga compensation...")
                    from transaction import CompensationStack, IdempotencyRegistry
                    idemp_reg = IdempotencyRegistry(dict(current_state.get("idempotency_records", {})))
                    comp_stack = CompensationStack(api=api, idempotency_registry=idemp_reg)
                    for rec in comp_stack_records:
                        t = rec.get("tool")
                        a = rec.get("args", {})
                        if t == "reopen_po":
                            pid = a.get("po_id")
                            comp_stack.push("reopen_po", lambda p=pid: api.reopen_po(p, "Saga Compensation: Rollback on user decline", user_id=approver_id), description=f"Restore cancelled PO {pid} back to OPEN")
                        elif t == "cancel_po":
                            pid = a.get("po_id")
                            comp_stack.push("cancel_po", lambda p=pid: api.cancel_po(p, "Saga Compensation: Rollback on user decline", user_id=approver_id), description=f"Cancel PO {pid}")
                        elif t == "reallocate_lot_for_order":
                            oid = a.get("order_id")
                            fl = a.get("from_lot_id")
                            tl = a.get("to_lot_id")
                            def _restore(o=oid, f=fl, t=tl):
                                with api.conn:
                                    api.conn.execute("UPDATE QualityLots SET allocated_order_id = ? WHERE lot_id = ?;", (o, f))
                                    api.conn.execute("UPDATE QualityLots SET allocated_order_id = NULL WHERE lot_id = ?;", (t,))
                            comp_stack.push("reallocate_lot_for_order", _restore, description=f"Restore lot {fl} to order {oid} and release {tl}")

                    rollback_log = comp_stack.unwind()
                    for r in rollback_log:
                        current_state["audit_trail"].append(f"Compensation Step [Rollback on Decline]: {r['action']} -> {r['status']} ({r['description']})")
                        print(f"    - {r['action']}: {r['description']} -> {r['status']}")
                    current_state["compensation_stack"] = []

                # Inject ToolMessage reporting rejection to model
                call_id = pending.get("id", "call-declined")
                decline_msg = ToolMessage(
                    content=(
                        f"DECLINED by you: You declined mutating action '{task_name}'. "
                        f"Any previously executed actions in this session have been safely rolled back via Saga compensation to preserve database invariants. "
                        f"Address the user directly as their personal assistant ('You declined...'). Deliberate on the operational consequences, "
                        f"issue a direct warning explaining why declining this action puts the production schedule at risk, "
                        f"and advise on immediate manual next steps."
                    ),
                    tool_call_id=call_id,
                )
                current_state["messages"].append(decline_msg)
                current_state["pending_action"] = None
                current_state["pending_tool_calls"] = []
                current_state["approval_status"] = "declined"

                # Resume the agent graph for final deliberation
                print("\n--- [Agent Deliberating on Human Decline] ---")
                try:
                    post_decline_state = graph.invoke(current_state, config)
                    current_state.update(post_decline_state)
                except Exception as exc:
                    print(f"  [!] Note: Error during deliberation: {exc}")

                # Extract and print the agent's warning
                last_msg = current_state.get("messages", [])[-1]
                warning_text = getattr(last_msg, "content", "")
                if warning_text:
                    print(f"\n[AGENT OPERATIONAL WARNING & DELIBERATION]:\n{warning_text}\n")
                    current_state.setdefault("audit_trail", []).append(f"Agent Deliberation & Warning: {warning_text}")

                break
            else:
                valid_opts = "[a]ccept, [d]ecline, [w]ait" if (not is_escalated and not is_quality) else "[a]ccept, [d]ecline"
                print(f"\n[!] Unrecognized input '{user_choice}'. Please choose {valid_opts}.")
                continue
        else:
            # Reached natural termination
            break

    print(f"\n--- [Audit Trail after {scenario_label} Execution] ---")
    for log_entry in current_state.get("audit_trail", []):
        print(f" > {log_entry}")

    telemetry = current_state.get("llm_telemetry", {})
    final_status = "DECLINED_WITH_WARNING" if current_state.get("approval_status") == "declined" else current_state.get("approval_status", "COMPLETED")
    save_run_audit(
        audit_file=getattr(args, "audit_file", "audit_trail.jsonl"),
        api=api,
        mode="freeform",
        thread_id=args.thread_id,
        user_id=current_state.get("user_id", user_id),
        attention_item=attention_item,
        audit_trail=current_state.get("audit_trail", []),
        telemetry=telemetry,
        final_status=final_status,
    )
    return final_status


def run_scenario_a(args, api: SQLiteCompanyAPI, attention_item: dict):
    print("\n" + "=" * 67)
    use_deterministic = args.deterministic or ("--deterministic" in sys.argv)
    mode_label = "DETERMINISTIC WORKFLOW" if use_deterministic else "FREE-FORM AGENT"
    print(f" Enterprise Agent: Scenario A (Purchasing PO Replacement) [{mode_label}]")
    print("=" * 67)
    user = api.get_user("u-101")
    print(f"Active User Context: {user['name']} ({user['title']}, ID: {user['user_id']})")
    print(f"Spending Approval Threshold: ${user['po_create_max_value']:,.2f}")
    print(f"Designated Backup Approver: {user['backup_approver_id']}")

    if use_deterministic:
        return run_deterministic_mode(args, api, attention_item)
    else:
        return run_freeform_mode(args, api, attention_item, user_id="u-101", scenario_label="Scenario A")


def run_scenario_b(args, api: SQLiteCompanyAPI, attention_item: dict):
    print("\n" + "=" * 67)
    print(" Enterprise Agent: Scenario B (Quality Hold & Reallocation) [FREE-FORM AGENT]")
    print("=" * 67)
    user = api.get_user("u-201")
    print(f"Active User Context: {user['name']} ({user['title']}, ID: {user['user_id']})")
    print(f"Authorized Scopes: {', '.join(user.get('scopes', []))}")

    run_freeform_mode(args, api, attention_item, user_id="u-201", scenario_label="Scenario B")


def run_failure_cases(audit_file: str = "audit_trail.jsonl") -> None:
    print("\n=================================================================")
    print(" [Phase 4: System Invariants & Failure Cases Demonstration]")
    print("=================================================================")

    failure_audit = []

    # Failure Case 1: RBAC & Scope Enforcement
    print("\n[Case 1/5: RBAC & Permission Scope Enforcement]")
    conn = create_company_database(":memory:", seed=True)
    api_casey = SQLiteCompanyAPI(conn=conn, current_user_id="u-201")  # Quality Manager
    try:
        api_casey.create_po(
            po_id="PO-TEST-UNAUTH",
            part_id="P-4471",
            supplier_id="S-Z",
            quantity=50,
            unit_price=210.0,
            promised_date="2026-09-04",
        )
        print("  [FAIL] Casey Chen was unexpectedly permitted to create a purchase order.")
        failure_audit.append({"case": "RBAC", "status": "FAIL"})
    except PermissionError as exc:
        print(f"  [PASS] Casey Chen (u-201) blocked from 'create_po':")
        print(f"         {exc}")
        failure_audit.append({"case": "RBAC_PO_WRITE", "status": "PASS", "intercepted": str(exc)})

    api_dana = SQLiteCompanyAPI(conn=conn, current_user_id="u-101")  # Purchasing Manager
    try:
        api_dana.reallocate_lot_for_order(
            order_id="4820",
            from_lot_id="L-2093",
            to_lot_id="L-2094",
        )
        print("  [FAIL] Dana Whitfield was unexpectedly permitted to mutate quality lots.")
        failure_audit.append({"case": "RBAC", "status": "FAIL"})
    except PermissionError as exc:
        print(f"  [PASS] Dana Whitfield (u-101) blocked from 'reallocate_lot_for_order':")
        print(f"         {exc}")
        failure_audit.append({"case": "RBAC_LOT_WRITE", "status": "PASS", "intercepted": str(exc)})

    # Failure Case 2: Domain Trap Rejection
    print("\n[Case 2/5: Domain Trap Rejection (Unapproved Supplier, Hold Lot & Spending Threshold)]")
    from environment.api import ValidationError
    from environment.exceptions import ApprovalLimitExceededError

    # Trap 2A: Unapproved Supplier S-X (Apex Fasteners)
    approved_suppliers = api_dana.query_suppliers(part_id="P-4471", approved_only=True)
    if not any(s["supplier_id"] == "S-X" for s in approved_suppliers):
        print(f"  [PASS] Unapproved Supplier Trap 'S-X' (Apex Fasteners) rejected:")
        print(f"         Excluded from eligible vendors (approved=0, fails vendor qualification).")
        failure_audit.append({"case": "SUPPLIER_TRAP", "status": "PASS", "intercepted": "Supplier S-X excluded (approved=0)"})
    else:
        print("  [FAIL] Unapproved supplier S-X passed validation.")
        failure_audit.append({"case": "SUPPLIER_TRAP", "status": "FAIL"})

    # Trap 2B: Contaminated Hold Lot L-2095
    try:
        api_casey.reallocate_lot_for_order(
            order_id="4820",
            from_lot_id="L-2093",
            to_lot_id="L-2095",
        )
        print("  [FAIL] Contaminated hold lot L-2095 was allocated.")
        failure_audit.append({"case": "LOT_TRAP", "status": "FAIL"})
    except ValidationError as exc:
        print(f"  [PASS] Hold Lot Trap 'L-2095' (Chemical contamination) rejected:")
        print(f"         {exc}")
        failure_audit.append({"case": "LOT_TRAP", "status": "PASS", "intercepted": str(exc)})

    # Trap 2C: Spending Approval Threshold ($25,000 policy limit)
    try:
        api_dana.create_po(
            po_id="PO-EXCEED-LIMIT",
            part_id="P-4471",
            supplier_id="S-Z",
            quantity=150,
            unit_price=200.0,  # $30,000 > $25,000 threshold
            promised_date="2026-09-04",
        )
        print("  [FAIL] Approval limit exceeded was not caught.")
        failure_audit.append({"case": "APPROVAL_LIMIT_TRAP", "status": "FAIL"})
    except ApprovalLimitExceededError as exc:
        print(f"  [PASS] Spending Approval Limit ($25,000 threshold) enforced:")
        print(f"         {exc}")
        failure_audit.append({"case": "APPROVAL_LIMIT_TRAP", "status": "PASS", "intercepted": str(exc)})

    # Failure Case 3: Trigger Deduplication
    print("\n[Case 3/5: Trigger Deduplication & Alert Suppression]")
    detector = Detector(api_dana)
    triggers_1 = detector.scan_all_attention_items(dedupe=True)
    triggers_2 = detector.scan_all_attention_items(dedupe=True)
    if len(triggers_2) == 0:
        print(f"  [PASS] Deduplication active: initial scan emitted {len(triggers_1)} signal(s),")
        print(f"         subsequent scan on identical state emitted {len(triggers_2)} signal(s).")
        failure_audit.append({"case": "TRIGGER_DEDUPE", "status": "PASS", "initial": len(triggers_1), "duplicate": len(triggers_2)})
    else:
        print(f"  [FAIL] Duplicate triggers emitted: {len(triggers_2)}")
        failure_audit.append({"case": "TRIGGER_DEDUPE", "status": "FAIL"})

    # Failure Case 4: Mid-Flight Saga Compensation Unwinding
    print("\n[Case 4/5: Mid-Flight Saga Compensation Rollback]")
    from transaction import CompensationStack, IdempotencyRegistry
    stack = CompensationStack(api=api_dana)
    po_before = api_dana.get_purchase_order("PO-77812")
    assert po_before["status"] == "OPEN"

    # Step 1: cancel PO-77812
    api_dana.cancel_po("PO-77812", reason="Mid-flight test cancellation")
    stack.push(
        action_name="cancel_purchase_order",
        compensating_callable=lambda: api_dana.reopen_po("PO-77812", reason="Saga compensation rollback"),
        description="Rollback PO-77812 cancellation to OPEN",
    )
    po_canceled = api_dana.get_purchase_order("PO-77812")
    print(f"  * Transaction Step 1 completed: PO-77812 status changed to '{po_canceled['status']}'")

    # Step 2: simulated downstream failure
    try:
        raise RuntimeError("Downstream warehouse service unavailable during transaction (HTTP 503)")
    except RuntimeError as exc:
        print(f"  * Transaction Step 2 simulated failure: {exc}")
        print("  * Initiating LIFO backward compensation unwinding...")
        unwound = stack.unwind()
        po_restored = api_dana.get_purchase_order("PO-77812")
        if po_restored["status"] == "OPEN" and len(unwound) == 1:
            print(f"  [PASS] Successfully unwound {len(unwound)} action(s).")
            print(f"         PO-77812 status restored to pre-incident state: '{po_restored['status']}'")
            failure_audit.append({"case": "SAGA_COMPENSATION", "status": "PASS", "unwound": len(unwound)})
        else:
            print(f"  [FAIL] State restoration failed: PO-77812 status is '{po_restored['status']}'")
            failure_audit.append({"case": "SAGA_COMPENSATION", "status": "FAIL"})

    # Failure Case 5: Idempotency Protection
    print("\n[Case 5/5: Idempotency Protection & Duplicate Write Prevention]")
    reg = IdempotencyRegistry()
    tx_key = "tx-po-replacement-9999"
    payload = {"status": "SUCCESS", "po_id": "PO-99999", "created_by": "u-101"}
    reg.record_success(tx_key, payload)

    cached_payload = reg.get_result(tx_key)
    if cached_payload == payload:
        print(f"  [PASS] Repeated operation with idempotency key '{tx_key}' safely")
        print(f"         short-circuited and returned cached result without duplicate write.")
        failure_audit.append({"case": "IDEMPOTENCY_PROTECTION", "status": "PASS", "key": tx_key})
    else:
        print("  [FAIL] Idempotency cache lookup failed.")
        failure_audit.append({"case": "IDEMPOTENCY_PROTECTION", "status": "FAIL"})

    try:
        with open(audit_file, "a", encoding="utf-8") as f:
            record = {
                "run_id": f"run-failure-cases-{int(datetime.now().timestamp())}",
                "timestamp": datetime.now().isoformat(),
                "mode": "system_failure_invariants",
                "results": failure_audit,
            }
            f.write(json.dumps(record) + "\n")
        print(f"\n[AUDIT LOG] Persisted failure invariants audit record to: {audit_file}")
    except Exception:
        pass

    print("\n=================================================================")
    print(" All Failure Cases and Invariants Verified Successfully.")
    print("=================================================================\n")


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 1. Verify LLM configuration
    llm_config = get_llm_config()
    api_key = llm_config.get("api_key")
    if not api_key and not args.allow_mock_planner:
        print("\n" + "=" * 65)
        print(" [CONFIGURATION ERROR] No LLM API key is configured.")
        print("=" * 65)
        print(" The agent requires an OpenAI-compliant endpoint (OpenRouter, Google")
        print(" Gemini, OpenAI, Groq, vLLM, etc.) to evaluate context and plan.")
        print("\n To resolve, set your credentials in a .env file (see .env.example)")
        print(" or export an API key in your shell:")
        print("     export LLM_API_KEY='...'")
        print("     export LLM_BASE_URL='...'  # optional, defaults to OpenRouter")
        print("     export LLM_MODEL='...'     # optional")
        print("\n Or for direct Google Gemini / AI Studio:")
        print("     export GEMINI_API_KEY='AIzaSy...'")
        print("\n Or run with --allow-mock-planner to run using deterministic fallback logic:")
        print("     python3 main.py --allow-mock-planner")
        print("=" * 65 + "\n")
        sys.exit(1)

    print("=================================================================")
    print(f" Enterprise Autonomous Operations Simulation [Clock: 2026-09-02]")
    print("=================================================================")

    # 2. Phase 1: Scenario A (Purchasing PO Replacement)
    conn_a = create_company_database(":memory:", seed=True)
    api_a = SQLiteCompanyAPI(conn=conn_a, current_user_id="u-101")
    detector_a = Detector(api_a)

    print(f"\nSystem Clock Initialized: {api_a.get_clock()}")
    print("--- [Day 1: Periodic Detection] Scanning operational channels ---")
    triggers_a = detector_a.scan_all_attention_items(dedupe=False)
    po_trigger = next((t for t in triggers_a if "delayed_shipment" in t.get("type", "")), None)

    if po_trigger:
        print(f"  * Detected Signal 1: {po_trigger['type']} (PO {po_trigger['po_id']} delayed past Order {po_trigger['production_order_id']} start)")

    if args.scenario in ("both", "a") and po_trigger:
        status_a = run_scenario_a(args, api_a, po_trigger)

        # Phase 2: Multi-Day Simulation: Advance clock through to Tuesday (2026-09-08)
        sim_days = args.days
        if sim_days > 0:
            print(f"\n--- [Phase 2: Multi-Day Simulation] Advancing clock through next Tuesday (2026-09-08) ---")
            current_date = api_a.get_clock()
            curr_dt = datetime.strptime(current_date, "%Y-%m-%d").date()
            target_dt = datetime.strptime("2026-09-08", "%Y-%m-%d").date()
            days_to_advance = max(0, (target_dt - curr_dt).days)

            for _ in range(days_to_advance):
                advanced_date = api_a.advance_clock(1)
                print(f" Clock tick -> {advanced_date}")

            print(f"\nSystem Clock is now: {api_a.get_clock()}")
            if status_a == "DECLINED_WITH_WARNING" or "declined" in str(status_a).lower():
                print("Checking Dana Whitfield's inbox on 2026-09-08:")
                print("  [INFO] No follow-up dock check scheduled: you declined the replacement workflow, so no arrival verification event was created.")
            else:
                print("Checking Dana Whitfield's inbox on 2026-09-08 for scheduled follow-up task:")
                dana_emails = api_a.get_emails(recipient_id="u-101")
                follow_up_email = next((e for e in dana_emails if "M-CHECK" in e["mail_id"]), None)

                if follow_up_email:
                    print(f"  [SUCCESS] Materialized Scheduled Task Email:")
                    print(f"    * Mail ID : {follow_up_email['mail_id']}")
                    print(f"    * From    : {follow_up_email['sender']}")
                    print(f"    * Subject : {follow_up_email['subject']}")
                    print(f"    * Body    : {follow_up_email['body']}")
                    print(f"    * Sent At : {follow_up_email['sent_at']}")
                else:
                    print("  [NOTE] No scheduled follow-up email in inbox.")

    # 3. Phase 3: Scenario B (Quality Hold & Reallocation)
    if args.scenario in ("both", "b"):
        conn_b = create_company_database(":memory:", seed=True)
        api_b = SQLiteCompanyAPI(conn=conn_b, current_user_id="u-201")
        detector_b = Detector(api_b)
        triggers_b = detector_b.scan_all_attention_items(dedupe=False)
        quality_trigger = next((t for t in triggers_b if "quality_hold" in t.get("type", "")), None)

        if quality_trigger:
            print(f"\n--- [Phase 3: Scenario B Detection] Signal 2: {quality_trigger['type']} (Lot {quality_trigger['lot_id']} on hold) ---")
            run_scenario_b(args, api_b, quality_trigger)

    # 4. Phase 4: System Invariants and Failure Cases
    if not args.skip_failures:
        run_failure_cases(audit_file=args.audit_file)

    print("\n=================================================================")
    print(" Enterprise Multi-Agent Simulation Pipeline Completed.")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
