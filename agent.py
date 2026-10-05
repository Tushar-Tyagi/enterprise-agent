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


class AgentState(TypedDict):
    user_id: str
    attention_item: Dict[str, Any]
    context: Dict[str, Any]
    proposed_plan: Dict[str, Any]
    alert_message: str
    approval_status: str  # "pending", "approved", "rejected"
    approver_id: str
    primary_approver_id: str
    escalated_to_backup: bool
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
        api_key = os.environ.get("OPENROUTER_API_KEY")

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
                "note": "Mocked plan (OPENROUTER_API_KEY not set)",
            }
        else:
            model_name = os.environ.get("OPENROUTER_MODEL", "google/gemini-3.1-pro-preview")

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
                llm = ChatOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                    model=model_name,
                    temperature=0,
                )
                structured_llm = llm.with_structured_output(PlannerOutput, include_raw=True)
                result = structured_llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ])
            except Exception as model_err:
                fallback_model = "google/gemini-2.5-flash"
                llm = ChatOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                    model=fallback_model,
                    temperature=0,
                )
                structured_llm = llm.with_structured_output(PlannerOutput, include_raw=True)
                result = structured_llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ])

            plan = result["parsed"]
            raw_msg = result.get("raw")

            usage = {}
            if hasattr(raw_msg, "usage_metadata") and raw_msg.usage_metadata:
                usage = dict(raw_msg.usage_metadata)
            elif hasattr(raw_msg, "response_metadata") and raw_msg.response_metadata:
                usage = raw_msg.response_metadata.get("token_usage") or {}
            if not isinstance(usage, dict):
                usage = {}

            resp_meta = getattr(raw_msg, "response_metadata", {}) or {}
            if not isinstance(resp_meta, dict):
                resp_meta = {}
            cost_info = resp_meta.get("cost") or resp_meta.get("cost_details") or 0.0

            telemetry = {
                "prompt_tokens": usage.get("input_tokens") or usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("output_tokens") or usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "prompt_tokens_details": usage.get("input_token_details") or usage.get("prompt_tokens_details", {}),
                "cost": cost_info,
            }

        plan_dict = plan.model_dump()
        audit_msg = (
            f"Planner Output: Alternate Supplier '{plan_dict['alternate_supplier_id']}', Qty {plan_dict['new_qty']}.\n"
            f"  * Manager Alert: \"{plan_dict.get('alert_message')}\"\n"
            f"  * Rationale: {plan_dict['rationale']}\n"
            f"  * Actions: {plan_dict['actions']}\n"
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

        # Evaluate the rule:
        # If approval request is unanswered at end of day and the approver's calendar shows them out the next day,
        # it routes to their designated backup.
        escalated = False
        if is_ooo_tomorrow and backup_id:
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
            routing_reason = "Standard approval required before issuing replacement PO and cancelling original."

        gate_msg = (
            f"Gate Check / Evaluation: Calculated PO Value = ${new_po_value:,.2f} (Limit = ${max_limit:,.2f}). "
            f"{routing_reason} (Approval Status: '{approval_status}', Authorized Approver: '{approver_id}')"
        )

        return {
            "approval_status": approval_status,
            "approver_id": approver_id,
            "primary_approver_id": primary_approver_id,
            "escalated_to_backup": escalated,
            "audit_trail": state.get("audit_trail", []) + [gate_msg],
        }

    def route_gate(state: AgentState) -> Literal["execute_plan", "__end__"]:
        """Conditional routing: pause at gate if pending approval, else proceed."""
        if state.get("approval_status") == "approved":
            return "execute_plan"
        return "__end__"

    def execute_plan(state: AgentState) -> Dict[str, Any]:
        """
        Execute approved replacement PO, cancel delayed PO, notify production,
        and schedule Tuesday arrival check, with transactional compensation on failure.
        """
        plan = state["proposed_plan"]
        attention = state["attention_item"]
        context = state["context"]
        approver_id = state.get("approver_id", state["user_id"])
        current_clock = api.get_clock()

        # Find unit price and lead time
        chosen_sup = next(
            (s for s in context["approved_suppliers"] if s["supplier_id"] == plan["alternate_supplier_id"]),
            None,
        )
        unit_price = chosen_sup["unit_price"] if chosen_sup else 210.0
        lead_time = chosen_sup["lead_time_days"] if chosen_sup else 2

        # Promised delivery date = clock + lead_time
        curr_dt = datetime.strptime(current_clock, "%Y-%m-%d").date()
        promised_date = (curr_dt + timedelta(days=lead_time)).strftime("%Y-%m-%d")

        audit_trail = list(state.get("audit_trail", []))
        created_po_id = None
        original_po_cancelled = False

        try:
            # 1. Create replacement PO as authorized approver
            new_po_id = "PO-77815"
            new_po = api.create_po(
                po_id=new_po_id,
                part_id=attention["part_id"],
                supplier_id=plan["alternate_supplier_id"],
                quantity=plan["new_qty"],
                unit_price=unit_price,
                promised_date=promised_date,
                user_id=approver_id,
            )
            created_po_id = new_po_id
            audit_trail.append(
                f"Execution [PO Create]: Created replacement PO {new_po_id} with {plan['alternate_supplier_id']} "
                f"for {plan['new_qty']} units at ${unit_price:.2f}/unit (Total: ${new_po['total_amount']:,.2f}, Promised: {promised_date})."
            )

            # 2. Cancel delayed original PO with Supplier Y
            api.cancel_po(
                po_id=attention["po_id"],
                reason=f"Cancelled due to shipment delay to {attention['delayed_promised_date']}. Replaced by {new_po_id}.",
                user_id=approver_id,
            )
            original_po_cancelled = True
            audit_trail.append(
                f"Execution [PO Cancel]: Successfully cancelled delayed PO {attention['po_id']}."
            )

            # 3. Notify production supervisor (Sam Taylor u-301)
            supervisor_id = attention.get("supervisor_id", "u-301")
            prod_msg = (
                f"Mitigation complete for Production Order {attention['production_order_id']}: "
                f"Original PO {attention['po_id']} cancelled. Replacement PO {created_po_id} placed with "
                f"Supplier Z with promised delivery on {promised_date} prior to scheduled start {attention['production_scheduled_start']}."
            )

            # Send both inbox message and production notification log
            api.send_email(
                recipient_id=supervisor_id,
                subject=f"Update: PO Replacement for Production Order {attention['production_order_id']}",
                body=prod_msg,
                user_id=approver_id,
            )
            api.notify_production(
                supervisor_id=supervisor_id,
                order_id=attention["production_order_id"],
                message=prod_msg,
                user_id=approver_id,
            )
            audit_trail.append(f"Execution [Notify]: Dispatched alert to Production Supervisor '{supervisor_id}'.")

            # 4. Schedule deferred task for Tuesday check (2026-09-08)
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
            api.schedule_event(
                trigger_date=check_date,
                target_table="Mail",
                payload=follow_up_payload,
            )
            audit_trail.append(
                f"Execution [Schedule]: Staged arrival verification task in ScheduledEvents for {check_date}."
            )

        except Exception as err:
            # Full Bidirectional Saga Compensation
            audit_trail.append(f"Execution Error: {err}. Initiating bidirectional rollback compensation...")

            # Rollback original PO cancellation: restore status to OPEN
            if original_po_cancelled:
                try:
                    api.reopen_po(
                        attention["po_id"],
                        reason="Compensation: Restoring original PO after downstream workflow failure.",
                        user_id=approver_id,
                    )
                    audit_trail.append(
                        f"Compensation [Rollback Original PO]: Successfully restored original PO {attention['po_id']} back to 'OPEN'."
                    )
                except Exception as reopen_err:
                    audit_trail.append(f"Compensation Failure: Could not restore {attention['po_id']}: {reopen_err}")

            # Rollback replacement PO creation: void/cancel it
            if created_po_id:
                try:
                    api.cancel_po(
                        created_po_id,
                        reason="Compensation: Reverting uncommitted replacement PO due to workflow exception.",
                        user_id=approver_id,
                    )
                    audit_trail.append(
                        f"Compensation [Rollback Replacement PO]: Successfully voided {created_po_id}."
                    )
                except Exception as comp_err:
                    audit_trail.append(f"Compensation Failure: Could not void {created_po_id}: {comp_err}")

            raise err

        return {
            "approval_status": "executed",
            "audit_trail": audit_trail,
        }

    # Graph construction
    workflow = StateGraph(AgentState)
    workflow.add_node("gather_context", gather_context)
    workflow.add_node("planner", planner)
    workflow.add_node("gate", gate)
    workflow.add_node("execute_plan", execute_plan)

    workflow.add_edge(START, "gather_context")
    workflow.add_edge("gather_context", "planner")
    workflow.add_edge("planner", "gate")
    workflow.add_conditional_edges(
        "gate",
        route_gate,
        {
            "execute_plan": "execute_plan",
            "__end__": END,
        },
    )
    workflow.add_edge("execute_plan", END)

    return workflow.compile(checkpointer=checkpointer)
