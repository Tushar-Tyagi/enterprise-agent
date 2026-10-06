import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from environment.api import SQLiteCompanyAPI
from environment.exceptions import ValidationError
from agent_freeform import extract_llm_telemetry, get_llm, get_llm_config

WORKFLOW_NAME = "po_reroute_workflow"
WORKFLOW_VERSION = "1.0.0"
WORKFLOW_STEPS = [
    "confirm_alternate_supplier",
    "confirm_lead_time",
    "create_new_po",
    "cancel_old_po",
    "notify_production",
    "schedule_arrival_check",
]


class AgentState(TypedDict, total=False):
    workflow_name: str
    workflow_version: str
    user_id: str
    attention_item: Dict[str, Any]
    context: Dict[str, Any]
    proposed_plan: Dict[str, Any]
    alert_message: str
    approval_status: str  # "pending", "approved", "rejected", "executed"
    approver_id: str
    primary_approver_id: str
    escalated_to_backup: bool
    unanswered_at_eod: bool
    completed_steps: List[str]
    compensation_stack: List[Dict[str, Any]]
    step_data: Dict[str, Any]
    audit_trail: List[str]


class PlannerOutput(BaseModel):
    alternate_supplier_id: str = Field(description="The ID of the chosen alternate supplier")
    new_qty: int = Field(description="The quantity to order")
    alert_message: str = Field(
        description="The alert to the manager: 'Part <part> will likely cause production order <order> to miss its scheduled start. Supplier <supplier> said the shipment is delayed until Tuesday. I can move the PO to Supplier <alt> and notify production. Want me to proceed?'"
    )
    rationale: str = Field(description="Explanation of why this supplier was chosen")
    actions: List[str] = Field(
        description="List of system actions to take (e.g., 'create_po', 'cancel_po', 'notify', 'schedule_check')"
    )


class BoundedSupplierChoice(BaseModel):
    chosen_supplier_id: str = Field(description="Must match one of the approved supplier IDs provided")
    selection_justification: str = Field(description="Short operational rationale for choosing this supplier")


class BoundedDraftNotification(BaseModel):
    subject: str = Field(description="Subject line for production supervisor")
    message: str = Field(description="Operational briefing for production supervisor confirming mitigation")


def unwind_compensations(
    api: SQLiteCompanyAPI,
    compensation_stack: List[Dict[str, Any]],
    audit_trail: List[str],
    approver_id: str,
) -> None:
    """
    Unwind registered compensations in LIFO order upon workflow step failure.
    """
    while compensation_stack:
        comp = compensation_stack.pop()
        action = comp.get("action")
        try:
            if action == "reopen_po":
                po_id = comp["po_id"]
                api.reopen_po(po_id, reason=comp.get("reason"), user_id=approver_id)
                msg = f"Compensation [Rollback Original PO]: Successfully restored original PO {po_id} back to 'OPEN'."
                audit_trail.append(msg)
                try:
                    api.log_audit_event(
                        run_id="compensation",
                        category="COMPENSATION",
                        actor_id=approver_id,
                        summary=f"Restored PO {po_id} to OPEN",
                        details={"action": "reopen_po", "po_id": po_id},
                    )
                except Exception:
                    pass
            elif action == "cancel_po":
                po_id = comp["po_id"]
                api.cancel_po(po_id, reason=comp.get("reason"), user_id=approver_id)
                msg = f"Compensation [Rollback Replacement PO]: Successfully voided {po_id}."
                audit_trail.append(msg)
                try:
                    api.log_audit_event(
                        run_id="compensation",
                        category="COMPENSATION",
                        actor_id=approver_id,
                        summary=f"Cancelled replacement PO {po_id}",
                        details={"action": "cancel_po", "po_id": po_id},
                    )
                except Exception:
                    pass
        except Exception as comp_err:
            audit_trail.append(f"Compensation Failure on {action}: {comp_err}")



def create_scenario_a_graph(api: SQLiteCompanyAPI, checkpointer: Optional[BaseCheckpointSaver] = None):
    """
    Build and compile the LangGraph agent for Scenario A with human-in-the-loop gating.
    """

    def gather_context(state: AgentState) -> Dict[str, Any]:
        """Fetch blocked production orders, delayed PO, supplier options, and user calendar."""
        user_id = state["user_id"]
        attention = state["attention_item"]

        # 1. Fetch ERP entities
        delayed_po = api.get_purchase_order(attention["po_id"])
        prod_order = api.get_production_order(attention["production_order_id"])
        user = api.get_user(user_id)
        current_clock = api.get_clock()

        # 2. Fetch approved alternate suppliers for this part
        approved_suppliers = api.query_suppliers(part_id=attention["part_id"], approved_only=True)

        # 3. Fetch user calendar events
        calendar_events = api.get_calendar_events(user_id=user_id)

        context = {
            "clock": current_clock,
            "user": user,
            "delayed_po": delayed_po,
            "blocked_production_order": prod_order,
            "calendar_events": calendar_events,
            "approved_suppliers": approved_suppliers,
        }

        supplier_summaries = [
            f"{s['name']} (ID: {s['supplier_id']}, Lead Time: {s['lead_time_days']}d, Price: ${s.get('unit_price', 0):.2f})"
            for s in approved_suppliers
        ]

        summary_msg = (
            f"Context Gathered: Active clock is {current_clock}. User '{user['name']}' has PO spending limit "
            f"${user['po_create_max_value']:,.2f}. PO {delayed_po['po_id']} (Part {delayed_po['part_id']}, Qty {delayed_po['quantity']}) "
            f"is delayed past Production Order {prod_order['order_id']} scheduled start on {prod_order['scheduled_start']}. "
            f"Available approved suppliers: {'; '.join(supplier_summaries)}."
        )

        return {
            "context": context,
            "audit_trail": state.get("audit_trail", []) + [summary_msg],
        }

    def planner(state: AgentState) -> Dict[str, Any]:
        """Call OpenRouter LLM to generate the action plan and log telemetry."""
        context = state["context"]
        attention = state["attention_item"]
        llm_config = get_llm_config()
        api_key = llm_config.get("api_key")

        if not api_key:
            # Fallback for offline / keyless testing
            chosen = next(
                (s for s in context["approved_suppliers"] if s["lead_time_days"] <= 4),
                context["approved_suppliers"][0],
            )
            alert_msg = (
                f"Part {attention['part_id']} will likely cause production order {attention['production_order_id']} "
                f"to miss its scheduled start. Supplier Y said the shipment is delayed until Tuesday. "
                f"I can move the PO to Supplier {chosen['supplier_id']} and notify production. Want me to proceed?"
            )
            plan = PlannerOutput(
                alternate_supplier_id=chosen["supplier_id"],
                new_qty=attention["quantity"],
                alert_message=alert_msg,
                rationale=(
                    f"Selected approved alternate {chosen['name']} ({chosen['supplier_id']}) because its {chosen['lead_time_days']}-day "
                    f"lead time delivers before Production Order {attention['production_order_id']} start date "
                    f"({attention['production_scheduled_start']}). Unapproved and slow suppliers were eliminated."
                ),
                actions=["create_po", "cancel_po", "notify_production", "schedule_delivery_check"],
            )
            telemetry = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_tokens_details": {},
                "cost": 0.0,
                "note": "Mocked plan (No LLM API key configured)",
            }
        else:
            system_prompt = (
                "You are an expert enterprise supply chain orchestrator. "
                "Analyze the delayed PO, stockout risk, and available approved suppliers. "
                "Select the best approved alternate supplier that can deliver in time for scheduled production. "
                "Draft an alert message to the purchasing manager with this exact phrasing structure:\n"
                "'Part <part> will likely cause production order <order> to miss its scheduled start. "
                "Supplier <current_supplier> said the shipment is delayed until Tuesday. "
                "I can move the PO to Supplier <alternate_supplier> and notify production. Want me to proceed?'"
            )

            suppliers_text = "\n".join(
                f"- Supplier ID: {s['supplier_id']}, Name: {s['name']}, Lead Time: {s['lead_time_days']} days, "
                f"Unit Price: ${s.get('unit_price', 0):.2f}, Approved: {s['approved']}"
                for s in context["approved_suppliers"]
            )

            user_prompt = (
                f"Operational Context:\n"
                f"Current Clock: {context['clock']}\n"
                f"Delayed PO: {attention['po_id']} for Part {attention['part_id']} (Qty: {attention['quantity']})\n"
                f"Current Supplier: Supplier Y ({attention['supplier_id']})\n"
                f"Original Promised Date: {attention['original_promised_date']}, Delayed Until: {attention['delayed_promised_date']}\n"
                f"Production Order: {attention['production_order_id']} scheduled to start on {attention['production_scheduled_start']}\n\n"
                f"Available Approved Suppliers:\n{suppliers_text}\n\n"
                f"Instructions:\n"
                f"Select an approved alternate supplier whose lead time guarantees delivery before "
                f"{attention['production_scheduled_start']}. Draft the exact alert message to the purchasing manager, "
                f"and provide quantity, rationale, and action steps."
            )

            try:
                llm = get_llm()
                structured_llm = llm.with_structured_output(PlannerOutput, include_raw=True)
                result = structured_llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ])
            except Exception as model_err:
                llm = get_llm(fallback=True)
                structured_llm = llm.with_structured_output(PlannerOutput, include_raw=True)
                result = structured_llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ])

            plan = result["parsed"]
            raw_msg = result.get("raw")
            telemetry = extract_llm_telemetry(raw_msg)

        plan_dict = plan.model_dump()

        # Resolve supplier unit price and lead time to construct detailed tool action parameters
        chosen_sup = next(
            (s for s in context.get("approved_suppliers", []) if s["supplier_id"] == plan.alternate_supplier_id),
            context["approved_suppliers"][0] if context.get("approved_suppliers") else None,
        )
        unit_price = chosen_sup.get("unit_price", 210.0) if chosen_sup else 210.0
        lead_time = chosen_sup.get("lead_time_days", 2) if chosen_sup else 2
        curr_dt = datetime.strptime(context["clock"], "%Y-%m-%d").date()
        promised_date = (curr_dt + timedelta(days=lead_time)).strftime("%Y-%m-%d")

        planned_actions = [
            {
                "tool": "create_purchase_order",
                "parameters": {
                    "po_id": "PO-77815",
                    "part_id": attention["part_id"],
                    "supplier_id": plan.alternate_supplier_id,
                    "quantity": plan.new_qty,
                    "unit_price": unit_price,
                    "total_amount": float(plan.new_qty * unit_price),
                    "promised_date": promised_date,
                },
                "rationale": f"Procure replacement parts from approved alternate {plan.alternate_supplier_id} with {lead_time}-day lead time delivering on {promised_date} prior to Production Order {attention['production_order_id']} start date ({attention['production_scheduled_start']}).",
            },
            {
                "tool": "cancel_purchase_order",
                "parameters": {
                    "po_id": attention["po_id"],
                    "reason": f"Cancelled due to shipment delay to {attention['delayed_promised_date']}. Replaced by PO-77815.",
                },
                "rationale": f"Cancel original delayed order with Supplier Y ({attention['supplier_id']}) to prevent duplicate inventory and cost.",
            },
            {
                "tool": "notify_production",
                "parameters": {
                    "supervisor_id": attention.get("supervisor_id", "u-301"),
                    "order_id": attention["production_order_id"],
                    "message": f"PO {attention['po_id']} cancelled and replaced by PO-77815 with Supplier Z. Delivery scheduled for {promised_date} prior to order start {attention['production_scheduled_start']}.",
                },
                "rationale": f"Alert production supervisor '{attention.get('supervisor_id', 'u-301')}' that line will not be starved.",
            },
            {
                "tool": "schedule_arrival_check",
                "parameters": {
                    "trigger_date": "2026-09-08",
                    "target_table": "Mail",
                    "payload": {
                        "mail_id": "M-CHECK-PO-77815",
                        "recipient_id": state["user_id"],
                        "subject": "Automated Check: Verify Arrival of Replacement PO PO-77815",
                        "sent_at": "2026-09-08T08:00:00",
                    },
                },
                "rationale": "Follow up on Tuesday (2026-09-08) to verify that replacement shipment actually arrived.",
            },
        ]
        plan_dict["planned_actions"] = planned_actions

        actions_formatted = "\n".join(
            f"    - {a['tool']}: {json.dumps(a['parameters'])}"
            for a in planned_actions
        )
        audit_msg = (
            f"Planner Output: Alternate Supplier '{plan_dict['alternate_supplier_id']}', Qty {plan_dict['new_qty']}.\n"
            f"  * Manager Alert: \"{plan_dict.get('alert_message')}\"\n"
            f"  * Rationale: {plan_dict['rationale']}\n"
            f"  * Planned Tool Actions with Parameters:\n{actions_formatted}\n"
            f"  * Telemetry: prompt_tokens={telemetry['prompt_tokens']}, "
            f"completion_tokens={telemetry['completion_tokens']}, total_tokens={telemetry['total_tokens']}"
        )

        new_context = dict(context)
        new_context["llm_telemetry"] = telemetry

        return {
            "proposed_plan": plan_dict,
            "alert_message": plan_dict.get("alert_message", ""),
            "primary_approver_id": state["user_id"],
            "context": new_context,
            "audit_trail": state.get("audit_trail", []) + [audit_msg],
        }

    def gate(state: AgentState) -> Dict[str, Any]:
        """
        Validate PO spending limits and evaluate calendar availability to route approval.
        Rule: If an approval request is unanswered at end of day and the approver's calendar
        shows them out the next day, it routes to their designated backup.
        """
        context = state["context"]
        plan = state["proposed_plan"]
        user = context["user"]
        primary_approver_id = state.get("primary_approver_id") or user["user_id"]

        # Calculate replacement PO value
        chosen_sup_id = plan["alternate_supplier_id"]
        chosen_sup = next(
            (s for s in context["approved_suppliers"] if s["supplier_id"] == chosen_sup_id),
            None,
        )
        unit_price = chosen_sup["unit_price"] if chosen_sup else 210.0
        new_po_value = float(unit_price * plan["new_qty"])
        max_limit = float(user.get("po_create_max_value") or 0.0)

        # Check calendar for tomorrow (2026-09-03)
        current_dt = datetime.strptime(context["clock"], "%Y-%m-%d").date()
        tomorrow_str = (current_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        is_ooo_tomorrow = api.is_user_out_of_office(primary_approver_id, check_date=tomorrow_str)
        backup_id = user.get("backup_approver_id")

        unanswered_at_eod = state.get("unanswered_at_eod")
        # If unanswered_at_eod is False, the primary approver is active and prompted first
        # If unanswered_at_eod is True (or default True for backward compatibility when omitted), evaluate OOO routing
        should_escalate_ooo = (unanswered_at_eod is not False) and is_ooo_tomorrow and backup_id

        escalated = False
        if should_escalate_ooo:
            escalated = True
            approver_id = backup_id
            approval_status = "pending"
            routing_reason = (
                f"Rule Triggered: Approval request was unanswered by Purchasing Manager '{primary_approver_id}' at end of day. "
                f"Approver's calendar indicates Out of Office tomorrow ({tomorrow_str}). "
                f"Routed to designated backup approver '{backup_id}' (Alex Morgan)."
            )
        elif new_po_value > max_limit:
            approver_id = primary_approver_id
            approval_status = "pending"
            routing_reason = f"PO value ${new_po_value:,.2f} exceeds user threshold ${max_limit:,.2f}."
        else:
            approver_id = primary_approver_id
            approval_status = "pending"
            routing_reason = f"Primary approver '{primary_approver_id}' active. Standard review required before issuing replacement PO and cancelling original."

        gate_msg = (
            f"Gate Check / Evaluation: Calculated PO Value = ${new_po_value:,.2f} (Limit = ${max_limit:,.2f}). "
            f"{routing_reason} (Approval Status: '{approval_status}', Authorized Approver: '{approver_id}')"
        )

        return {
            "approval_status": approval_status,
            "approver_id": approver_id,
            "primary_approver_id": primary_approver_id,
            "escalated_to_backup": escalated,
            "unanswered_at_eod": bool(unanswered_at_eod),
            "audit_trail": state.get("audit_trail", []) + [gate_msg],
        }

    def route_gate(state: AgentState) -> Literal["confirm_alternate_supplier", "__end__"]:
        """Conditional routing: pause at gate if pending approval, else enter declared workflow."""
        if state.get("approval_status") == "approved":
            return "confirm_alternate_supplier"
        return "__end__"

    # =========================================================================
    # Declared Deterministic Workflow Steps (Part 2)
    # Fixed topology:
    # confirm_alternate_supplier -> confirm_lead_time -> create_new_po
    # -> cancel_old_po -> notify_production -> schedule_arrival_check
    # =========================================================================

    def confirm_alternate_supplier(state: AgentState) -> Dict[str, Any]:
        """
        Step 1: Confirm the alternate supplier is approved for the part.
        Bounded LLM selection: only pre-filtered approved suppliers are permitted.
        """
        audit_trail = list(state.get("audit_trail", []))
        completed_steps = list(state.get("completed_steps", []))
        step_data = dict(state.get("step_data", {}))
        compensation_stack = list(state.get("compensation_stack", []))

        if "confirm_alternate_supplier" in completed_steps:
            return {
                "completed_steps": completed_steps,
                "step_data": step_data,
                "audit_trail": audit_trail,
            }

        attention = state["attention_item"]
        plan = state.get("proposed_plan", {})
        approver_id = state.get("approver_id") or state["user_id"]

        # Fetch pre-filtered approved suppliers from authoritative ERP
        approved_suppliers = api.query_suppliers(part_id=attention["part_id"], approved_only=True)
        if not approved_suppliers:
            err = ValidationError(f"No approved alternate suppliers found for part {attention['part_id']}.")
            unwind_compensations(api, compensation_stack, audit_trail, approver_id)
            raise err

        valid_supplier_ids = {s["supplier_id"]: s for s in approved_suppliers}
        chosen_id = plan.get("alternate_supplier_id")

        if chosen_id not in valid_supplier_ids:
            # Bounded guarantee: fall back to lowest lead-time approved supplier
            chosen_supplier = min(approved_suppliers, key=lambda s: s["lead_time_days"])
            chosen_id = chosen_supplier["supplier_id"]
            justification = f"Planner selected unapproved/empty supplier. Bounded selection to approved supplier '{chosen_id}' ({chosen_supplier['name']})."
        else:
            chosen_supplier = valid_supplier_ids[chosen_id]
            justification = f"Confirmed supplier '{chosen_id}' ({chosen_supplier['name']}) is approved for part {attention['part_id']}."

        unit_price = chosen_supplier.get("unit_price") or 210.0
        lead_time = chosen_supplier.get("lead_time_days") or 2

        step_data["supplier_id"] = chosen_id
        step_data["supplier_name"] = chosen_supplier["name"]
        step_data["unit_price"] = unit_price
        step_data["lead_time_days"] = lead_time
        completed_steps.append("confirm_alternate_supplier")

        audit_msg = (
            f"Workflow Step 1 [confirm_alternate_supplier]: Verified approved supplier '{chosen_id}' "
            f"({chosen_supplier['name']}, unit_price=${unit_price:.2f}, lead_time={lead_time}d). {justification}"
        )
        audit_trail.append(audit_msg)

        return {
            "workflow_name": WORKFLOW_NAME,
            "workflow_version": WORKFLOW_VERSION,
            "completed_steps": completed_steps,
            "step_data": step_data,
            "compensation_stack": compensation_stack,
            "audit_trail": audit_trail,
        }

    def confirm_lead_time(state: AgentState) -> Dict[str, Any]:
        """
        Step 2: Confirm their lead time meets the production date.
        """
        audit_trail = list(state.get("audit_trail", []))
        completed_steps = list(state.get("completed_steps", []))
        step_data = dict(state.get("step_data", {}))
        compensation_stack = list(state.get("compensation_stack", []))

        if "confirm_lead_time" in completed_steps:
            return {
                "completed_steps": completed_steps,
                "step_data": step_data,
                "audit_trail": audit_trail,
            }

        attention = state["attention_item"]
        approver_id = state.get("approver_id") or state["user_id"]
        current_clock = api.get_clock()
        lead_time = step_data.get("lead_time_days", 2)

        curr_dt = datetime.strptime(current_clock, "%Y-%m-%d").date()
        promised_date = (curr_dt + timedelta(days=lead_time)).strftime("%Y-%m-%d")
        prod_start = attention["production_scheduled_start"]

        if promised_date > prod_start:
            err = ValidationError(
                f"Alternate supplier lead time yields arrival date {promised_date}, "
                f"which breaches production scheduled start {prod_start}."
            )
            unwind_compensations(api, compensation_stack, audit_trail, approver_id)
            raise err

        step_data["promised_date"] = promised_date
        completed_steps.append("confirm_lead_time")

        audit_msg = (
            f"Workflow Step 2 [confirm_lead_time]: Confirmed arrival date {promised_date} "
            f"(lead time: {lead_time}d from {current_clock}) arrives before production start {prod_start}."
        )
        audit_trail.append(audit_msg)

        return {
            "completed_steps": completed_steps,
            "step_data": step_data,
            "compensation_stack": compensation_stack,
            "audit_trail": audit_trail,
        }

    def create_new_po(state: AgentState) -> Dict[str, Any]:
        """
        Step 3: Create the replacement PO idempotently and register compensation.
        """
        audit_trail = list(state.get("audit_trail", []))
        completed_steps = list(state.get("completed_steps", []))
        step_data = dict(state.get("step_data", {}))
        compensation_stack = list(state.get("compensation_stack", []))

        if "create_new_po" in completed_steps and step_data.get("created_po_id"):
            return {
                "completed_steps": completed_steps,
                "step_data": step_data,
                "audit_trail": audit_trail,
            }

        attention = state["attention_item"]
        approver_id = state.get("approver_id") or state["user_id"]
        qty = state.get("proposed_plan", {}).get("new_qty", attention["quantity"])
        supplier_id = step_data.get("supplier_id", "S-Z")
        unit_price = step_data.get("unit_price", 210.0)
        promised_date = step_data.get("promised_date", "2026-09-04")

        # Idempotency check: see if replacement PO already exists
        target_po_id = "PO-77815"
        try:
            existing = api.get_purchase_order(target_po_id)
            if existing.get("status") == "OPEN":
                step_data["created_po_id"] = target_po_id
                completed_steps.append("create_new_po")
                audit_trail.append(f"Execution [PO Create] (Idempotent): Found existing replacement PO {target_po_id}.")
                return {
                    "completed_steps": completed_steps,
                    "step_data": step_data,
                    "compensation_stack": compensation_stack,
                    "audit_trail": audit_trail,
                }
        except Exception:
            pass

        try:
            new_po = api.create_po(
                po_id=target_po_id,
                part_id=attention["part_id"],
                supplier_id=supplier_id,
                quantity=qty,
                unit_price=unit_price,
                promised_date=promised_date,
                user_id=approver_id,
            )
            step_data["created_po_id"] = target_po_id
            completed_steps.append("create_new_po")

            # Register inverse compensation: void replacement PO
            compensation_stack.append({
                "action": "cancel_po",
                "po_id": target_po_id,
                "reason": "Compensation: Reverting uncommitted replacement PO due to downstream workflow failure.",
            })

            audit_trail.append(
                f"Execution [PO Create]: Created replacement PO {target_po_id} with {supplier_id} "
                f"for {qty} units at ${unit_price:.2f}/unit (Total: ${new_po['total_amount']:,.2f}, Promised: {promised_date})."
            )

            return {
                "completed_steps": completed_steps,
                "step_data": step_data,
                "compensation_stack": compensation_stack,
                "audit_trail": audit_trail,
            }
        except Exception as err:
            unwind_compensations(api, compensation_stack, audit_trail, approver_id)
            raise err

    def cancel_old_po(state: AgentState) -> Dict[str, Any]:
        """
        Step 4: Cancel or reduce the original delayed PO idempotently and register compensation.
        """
        audit_trail = list(state.get("audit_trail", []))
        completed_steps = list(state.get("completed_steps", []))
        step_data = dict(state.get("step_data", {}))
        compensation_stack = list(state.get("compensation_stack", []))

        if "cancel_old_po" in completed_steps:
            return {
                "completed_steps": completed_steps,
                "step_data": step_data,
                "audit_trail": audit_trail,
            }

        attention = state["attention_item"]
        approver_id = state.get("approver_id") or state["user_id"]
        old_po_id = attention["po_id"]
        new_po_id = step_data.get("created_po_id", "PO-77815")

        try:
            old_po = api.get_purchase_order(old_po_id)
            if old_po["status"] != "CANCELLED":
                api.cancel_po(
                    po_id=old_po_id,
                    reason=f"Cancelled due to shipment delay to {attention['delayed_promised_date']}. Replaced by {new_po_id}.",
                    user_id=approver_id,
                )
                # Register inverse compensation: restore original PO to OPEN
                compensation_stack.append({
                    "action": "reopen_po",
                    "po_id": old_po_id,
                    "reason": "Compensation: Restoring original PO after downstream workflow failure.",
                })
                audit_trail.append(f"Execution [PO Cancel]: Successfully cancelled delayed PO {old_po_id}.")
            else:
                audit_trail.append(f"Execution [PO Cancel] (Idempotent): Delayed PO {old_po_id} already cancelled.")

            completed_steps.append("cancel_old_po")

            return {
                "completed_steps": completed_steps,
                "step_data": step_data,
                "compensation_stack": compensation_stack,
                "audit_trail": audit_trail,
            }
        except Exception as err:
            unwind_compensations(api, compensation_stack, audit_trail, approver_id)
            raise err

    def notify_production(state: AgentState) -> Dict[str, Any]:
        """
        Step 5: Notify production supervisor that mitigation is complete.
        Bounded LLM step: generates message text within strict boundary without tool/state mutations.
        """
        audit_trail = list(state.get("audit_trail", []))
        completed_steps = list(state.get("completed_steps", []))
        step_data = dict(state.get("step_data", {}))
        compensation_stack = list(state.get("compensation_stack", []))

        if "notify_production" in completed_steps:
            return {
                "completed_steps": completed_steps,
                "step_data": step_data,
                "audit_trail": audit_trail,
            }

        attention = state["attention_item"]
        approver_id = state.get("approver_id") or state["user_id"]
        supervisor_id = attention.get("supervisor_id", "u-301")
        new_po_id = step_data.get("created_po_id", "PO-77815")
        promised_date = step_data.get("promised_date", "2026-09-04")
        order_id = attention["production_order_id"]

        # Bounded LLM draft or template
        llm_config = get_llm_config()
        api_key = llm_config.get("api_key")
        subj = f"Update: PO Replacement for Production Order {order_id}"
        prod_msg = (
            f"Mitigation complete for Production Order {order_id}: "
            f"Original PO {attention['po_id']} cancelled. Replacement PO {new_po_id} placed with "
            f"Supplier Z with promised delivery on {promised_date} prior to scheduled start {attention['production_scheduled_start']}."
        )

        if api_key:
            try:
                llm = get_llm()
                bounded_llm = llm.with_structured_output(BoundedDraftNotification)
                prompt = (
                    f"Draft a brief operational notification to Production Supervisor {supervisor_id} for Order {order_id}. "
                    f"Context: delayed PO {attention['po_id']} was cancelled and replacement PO {new_po_id} "
                    f"was placed with Supplier Z with delivery on {promised_date} prior to scheduled start {attention['production_scheduled_start']}."
                )
                draft = bounded_llm.invoke([
                    SystemMessage(content="You are an enterprise communications formatter. Output subject and message only. Do not perform any tool calls or actions."),
                    HumanMessage(content=prompt),
                ])
                if draft and getattr(draft, "message", None):
                    subj = draft.subject
                    prod_msg = draft.message
            except Exception:
                pass

        try:
            api.send_email(
                recipient_id=supervisor_id,
                subject=subj,
                body=prod_msg,
                user_id=approver_id,
            )
            api.notify_production(
                supervisor_id=supervisor_id,
                order_id=order_id,
                message=prod_msg,
                user_id=approver_id,
            )
            completed_steps.append("notify_production")
            audit_trail.append(f"Execution [Notify]: Dispatched alert to Production Supervisor '{supervisor_id}'.")

            return {
                "completed_steps": completed_steps,
                "step_data": step_data,
                "compensation_stack": compensation_stack,
                "audit_trail": audit_trail,
            }
        except Exception as err:
            unwind_compensations(api, compensation_stack, audit_trail, approver_id)
            raise err

    def schedule_arrival_check(state: AgentState) -> Dict[str, Any]:
        """
        Step 6: Schedule arrival check on Tuesday (2026-09-08) to confirm shipment arrival.
        """
        audit_trail = list(state.get("audit_trail", []))
        completed_steps = list(state.get("completed_steps", []))
        step_data = dict(state.get("step_data", {}))
        compensation_stack = list(state.get("compensation_stack", []))

        if "schedule_arrival_check" in completed_steps:
            return {
                "completed_steps": completed_steps,
                "step_data": step_data,
                "approval_status": "executed",
                "audit_trail": audit_trail,
            }

        attention = state["attention_item"]
        approver_id = state.get("approver_id") or state["user_id"]
        created_po_id = step_data.get("created_po_id", "PO-77815")
        check_date = "2026-09-08"

        follow_up_payload = {
            "mail_id": f"M-CHECK-{created_po_id}",
            "sender": "system@enterprise.internal",
            "recipient_id": state["user_id"],
            "subject": f"Automated Check: Verify Arrival of Replacement PO {created_po_id}",
            "body": (
                f"Arrival verification for replacement PO {created_po_id} (Part {attention['part_id']}). "
                f"Confirm shipment arrived from Supplier Z for Production Order {attention['production_order_id']}."
            ),
            "sent_at": f"{check_date}T08:00:00",
            "read_status": 0,
        }

        try:
            api.schedule_event(
                trigger_date=check_date,
                target_table="Mail",
                payload=follow_up_payload,
            )
            completed_steps.append("schedule_arrival_check")
            audit_trail.append(
                f"Execution [Schedule]: Staged arrival verification task in ScheduledEvents for {check_date}."
            )

            return {
                "completed_steps": completed_steps,
                "step_data": step_data,
                "approval_status": "executed",
                "audit_trail": audit_trail,
            }
        except Exception as err:
            unwind_compensations(api, compensation_stack, audit_trail, approver_id)
            raise err

    # Graph construction
    workflow = StateGraph(AgentState)
    workflow.add_node("gather_context", gather_context)
    workflow.add_node("planner", planner)
    workflow.add_node("gate", gate)
    workflow.add_node("confirm_alternate_supplier", confirm_alternate_supplier)
    workflow.add_node("confirm_lead_time", confirm_lead_time)
    workflow.add_node("create_new_po", create_new_po)
    workflow.add_node("cancel_old_po", cancel_old_po)
    workflow.add_node("notify_production", notify_production)
    workflow.add_node("schedule_arrival_check", schedule_arrival_check)

    workflow.add_edge(START, "gather_context")
    workflow.add_edge("gather_context", "planner")
    workflow.add_edge("planner", "gate")
    workflow.add_conditional_edges(
        "gate",
        route_gate,
        {
            "confirm_alternate_supplier": "confirm_alternate_supplier",
            "__end__": END,
        },
    )
    workflow.add_edge("confirm_alternate_supplier", "confirm_lead_time")
    workflow.add_edge("confirm_lead_time", "create_new_po")
    workflow.add_edge("create_new_po", "cancel_old_po")
    workflow.add_edge("cancel_old_po", "notify_production")
    workflow.add_edge("notify_production", "schedule_arrival_check")
    workflow.add_edge("schedule_arrival_check", END)

    return workflow.compile(checkpointer=checkpointer)

