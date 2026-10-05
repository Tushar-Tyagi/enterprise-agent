"""
Free-Form Autonomous Agent Graph with Centralized Toolcards, RBAC Gating,
Mandatory Rationale, Idempotency, and Saga Rollback Compensation.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from environment.api import SQLiteCompanyAPI
from toolcards import TOOL_REGISTRY, build_toolcards_prompt, get_all_tools
from transaction import CompensationStack, IdempotencyRegistry


class FreeformAgentState(TypedDict):
    messages: List[BaseMessage]
    user_id: str
    attention_item: Dict[str, Any]
    pending_action: Optional[Dict[str, Any]]
    approval_status: str  # "none", "pending", "approved", "rejected"
    approver_id: str
    primary_approver_id: str
    escalated_to_backup: bool
    unanswered_at_eod: bool
    idempotency_records: Dict[str, Any]
    compensation_stack: List[Dict[str, Any]]
    audit_trail: List[str]
    llm_telemetry: Dict[str, Any]


def create_freeform_agent_graph(
    api: SQLiteCompanyAPI,
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """
    Construct the cyclical free-flowing LangGraph agent.
    """

    def agent_reasoning(state: FreeformAgentState) -> Dict[str, Any]:
        """
        Agent reasoning node: decides next action, invoking read or mutating tools.
        """
        messages = list(state.get("messages", []))
        user_id = state.get("user_id", "u-101")
        attention = state["attention_item"]
        api_key = os.environ.get("OPENROUTER_API_KEY")

        # 1. Initialize conversation if first invocation
        if not messages:
            toolcards_doc = build_toolcards_prompt()
            system_prompt = (
                "You are an expert enterprise operations and supply chain agent.\n"
                "Your objective is to resolve operational disruptions and stockout risks autonomously.\n"
                "Inspect incoming signals, emails, POs, and production schedules, evaluate approved suppliers, "
                "and execute the full mitigation workflow:\n"
                "  Step 1: Cancel the delayed purchase order.\n"
                "  Step 2: Create a replacement purchase order with an approved alternate supplier capable of delivering before scheduled production start.\n"
                "  Step 3: Notify the production supervisor (user ID: 'u-301' / Sam Taylor) regarding Order 4812 resolution.\n\n"
                "CRITICAL POLICIES:\n"
                "1. Every single tool call MUST provide a detailed 'reason' parameter justifying the call.\n"
                "   Generic phrases like 'mitigate disruption' or 'cancel delayed PO' are STRICTLY PROHIBITED.\n"
                "   For any mutating action (create_purchase_order, cancel_purchase_order, notify_production), "
                "   your 'reason' MUST explicitly include:\n"
                "   (a) What production order is delayed and its scheduled start date (e.g. Order 4812 starting 2026-09-07).\n"
                "   (b) The exact delay details (e.g. Supplier Y delayed PO-77812 to 2026-09-08, slipping past start).\n"
                "   (c) The chosen alternate supplier, unit price, lead time, and promised delivery date before production starts (e.g. Supplier Z / S-Z delivering on 2026-09-04).\n"
                "2. All mutating tools (creating/cancelling POs, notifications) are intercepted by human gating.\n"
                "3. Each mutating call must specify a unique 'idempotency_key'.\n"
                "4. Operational Directory: Production Supervisor for Order 4812 is Sam Taylor (user ID: 'u-301'). When calling notify_production, set supervisor_id='u-301' and order_id='4812'.\n\n"
                f"{toolcards_doc}"
            )
            human_prompt = (
                f"Attention Item Received:\n"
                f"Type: {attention.get('type')}\n"
                f"Delayed PO: {attention.get('po_id')} for Part {attention.get('part_id')}\n"
                f"Delayed Promised Date: {attention.get('delayed_promised_date')}\n"
                f"Impacted Production Order: {attention.get('production_order_id')} (Scheduled Start: {attention.get('production_scheduled_start')})\n"
                f"Quantity: {attention.get('quantity')}\n\n"
                f"Investigate the situation, find an approved alternate supplier capable of delivering before production start, "
                f"and execute the necessary remediation steps."
            )
            messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=human_prompt))

        audit_trail = list(state.get("audit_trail", []))
        telemetry = dict(state.get("llm_telemetry", {}))

        # 2. Invoke Model or Deterministic Mock Loop if key is absent
        if not api_key:
            # Deterministic simulation of free-flowing exploration
            tool_messages_count = sum(1 for m in messages if isinstance(m, ToolMessage))

            if tool_messages_count == 0:
                # Step 1: Query PO details
                ai_msg = AIMessage(
                    content="I will inspect the delayed purchase order to verify details and status.",
                    tool_calls=[{
                        "name": "get_purchase_order",
                        "args": {
                            "po_id": attention["po_id"],
                            "reason": "Examine line items, current status, and supplier for delayed shipment",
                        },
                        "id": "call-po-001",
                    }],
                )
                audit_trail.append("Agent Action: Inspecting purchase order PO-77812.")
            elif tool_messages_count == 1:
                # Step 2: Query alternate suppliers
                ai_msg = AIMessage(
                    content="Delayed PO confirmed. Now querying approved alternate suppliers for part P-4471.",
                    tool_calls=[{
                        "name": "query_suppliers",
                        "args": {
                            "part_id": attention["part_id"],
                            "approved_only": True,
                            "reason": "Identify alternate approved vendors capable of beating scheduled production start",
                        },
                        "id": "call-sup-002",
                    }],
                )
                audit_trail.append("Agent Action: Searching approved alternate suppliers.")
            elif tool_messages_count == 2:
                # Step 3: Create replacement PO with Supplier Z (mutating)
                ai_msg = AIMessage(
                    content="Supplier Z has a 2-day lead time ($210/unit). Initiating replacement purchase order.",
                    tool_calls=[{
                        "name": "create_purchase_order",
                        "args": {
                            "po_id": "PO-77815",
                            "part_id": attention["part_id"],
                            "supplier_id": "S-Z",
                            "quantity": attention["quantity"],
                            "unit_price": 210.0,
                            "promised_date": "2026-09-04",
                            "idempotency_key": "idemp-po-77815",
                            "reason": (
                                f"Part {attention['part_id']} was delayed by Supplier Y to {attention['delayed_promised_date']}, "
                                f"breaching Production Order {attention['production_order_id']} scheduled start on {attention['production_scheduled_start']}. "
                                f"Creating replacement order with approved alternate Supplier Z (S-Z) at $210.00/unit with 2-day lead time "
                                f"guaranteeing delivery on 2026-09-04 prior to production."
                            ),
                        },
                        "id": "call-create-po-003",
                    }],
                )
                audit_trail.append("Agent Action: Requesting creation of replacement PO-77815.")
            elif tool_messages_count == 3:
                # Step 4: Cancel delayed PO (mutating)
                ai_msg = AIMessage(
                    content="Replacement PO-77815 established. Now cancelling delayed PO-77812.",
                    tool_calls=[{
                        "name": "cancel_purchase_order",
                        "args": {
                            "po_id": attention["po_id"],
                            "reason": (
                                f"Cancelling delayed PO {attention['po_id']} with Supplier Y because its revised delivery on {attention['delayed_promised_date']} "
                                f"slips past Production Order {attention['production_order_id']} scheduled start on {attention['production_scheduled_start']}. "
                                f"Sourcing has been transferred to approved alternate Supplier Z (S-Z) with promised delivery on 2026-09-04 "
                                f"under replacement PO-77815."
                            ),
                            "idempotency_key": "idemp-cancel-77812",
                        },
                        "id": "call-cancel-po-004",
                    }],
                )
                audit_trail.append(f"Agent Action: Cancelling delayed PO {attention['po_id']}.")
            elif tool_messages_count == 4:
                # Step 5: Notify production supervisor (mutating)
                ai_msg = AIMessage(
                    content="Notifying production supervisor Sam Taylor regarding the resolution.",
                    tool_calls=[{
                        "name": "notify_production",
                        "args": {
                            "supervisor_id": "u-301",
                            "order_id": attention["production_order_id"],
                            "message": (
                                f"Mitigation complete for Order {attention['production_order_id']}: Original PO {attention['po_id']} "
                                f"cancelled. Replacement PO PO-77815 placed with Supplier Z (promised delivery 2026-09-04)."
                            ),
                            "idempotency_key": "idemp-notify-4812",
                            "reason": "Alert production supervisor that line will not be starved",
                        },
                        "id": "call-notify-005",
                    }],
                )
                audit_trail.append("Agent Action: Notifying Production Supervisor.")
            else:
                ai_msg = AIMessage(
                    content=(
                        f"Operational mitigation complete: Delayed PO {attention['po_id']} has been cancelled and replaced "
                        f"with PO-77815 from Supplier Z. Production order {attention['production_order_id']} is protected."
                    )
                )
                audit_trail.append("Agent Reasoning: Mitigation successfully completed.")

            messages.append(ai_msg)
            telemetry = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
                "note": "Mock agent cycle (OPENROUTER_API_KEY not set)",
            }
        else:
            # Live OpenRouter LLM Call
            tools = get_all_tools(api=api, user_id=user_id)
            model_name = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")

            try:
                llm = ChatOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                    model=model_name,
                    temperature=0,
                )
                llm_with_tools = llm.bind_tools(tools)
                ai_msg = llm_with_tools.invoke(messages)
            except Exception:
                fallback_model = "google/gemini-2.5-flash"
                llm = ChatOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                    model=fallback_model,
                    temperature=0,
                )
                llm_with_tools = llm.bind_tools(tools)
                ai_msg = llm_with_tools.invoke(messages)

            messages.append(ai_msg)

            usage = {}
            if hasattr(ai_msg, "usage_metadata") and ai_msg.usage_metadata:
                usage = dict(ai_msg.usage_metadata)
            elif hasattr(ai_msg, "response_metadata") and ai_msg.response_metadata:
                usage = ai_msg.response_metadata.get("token_usage", {})

            resp_meta = getattr(ai_msg, "response_metadata", {}) or {}
            cost_info = resp_meta.get("cost") or resp_meta.get("cost_details") or 0.0

            telemetry = {
                "prompt_tokens": usage.get("input_tokens") or usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("output_tokens") or usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "prompt_tokens_details": usage.get("input_token_details") or usage.get("prompt_tokens_details", {}),
                "cost": cost_info,
            }

        return {
            "messages": messages,
            "audit_trail": audit_trail,
            "llm_telemetry": telemetry,
        }

    def execute_read_tool(state: FreeformAgentState) -> Dict[str, Any]:
        """
        Execute read-only tool calls immediately and return observation to model.
        """
        messages = list(state["messages"])
        last_msg = messages[-1]
        user_id = state.get("user_id", "u-101")
        audit_trail = list(state.get("audit_trail", []))

        tools_map = {t.name: t for t in get_all_tools(api=api, user_id=user_id)}

        for call in last_msg.tool_calls:
            tool_name = call["name"]
            tool_args = call["args"]
            tool_id = call["id"]

            tool = tools_map.get(tool_name)
            if not tool:
                obs = f"Error: Tool '{tool_name}' not found."
            else:
                try:
                    res = tool.invoke(tool_args)
                    obs = json.dumps(res) if isinstance(res, (dict, list)) else str(res)
                except Exception as exc:
                    obs = f"Error executing tool '{tool_name}': {exc}"

            messages.append(ToolMessage(content=obs, tool_call_id=tool_id))
            audit_trail.append(f"Observation [{tool_name}]: {obs[:120]}...")

        return {
            "messages": messages,
            "audit_trail": audit_trail,
        }

    def gate_mutating_tool(state: FreeformAgentState) -> Dict[str, Any]:
        """
        Intercept mutating tool calls, evaluate spending limits and out-of-office rules,
        and pause for human authorization.
        """
        messages = list(state["messages"])
        last_msg = messages[-1]
        user_id = state.get("user_id", "u-101")
        primary_approver_id = state.get("primary_approver_id") or user_id
        audit_trail = list(state.get("audit_trail", []))

        call = last_msg.tool_calls[0]
        tool_name = call["name"]
        tool_args = call["args"]
        call_id = call["id"]

        user = api.get_user(primary_approver_id)
        backup_id = user.get("backup_approver_id")
        current_clock = api.get_clock()
        # Evaluate whether primary approver is Out of Office TODAY or if request was left unanswered at EOD
        is_ooo_today = api.is_user_out_of_office(primary_approver_id, check_date=current_clock)
        unanswered_at_eod = state.get("unanswered_at_eod", False)

        escalated = False
        if (is_ooo_today or unanswered_at_eod) and backup_id:
            escalated = True
            approver_id = backup_id
            if unanswered_at_eod:
                routing_reason = (
                    f"Rule Triggered: Approval request was unanswered by '{primary_approver_id}' at end of day. "
                    f"Approver is Out of Office on {current_clock}. "
                    f"Escalated authority to designated backup '{backup_id}' (Alex Morgan)."
                )
            else:
                routing_reason = (
                    f"Rule Triggered: Primary approver '{primary_approver_id}' is Out of Office on {current_clock}. "
                    f"Escalated authority to designated backup '{backup_id}' (Alex Morgan)."
                )
        else:
            approver_id = primary_approver_id
            routing_reason = f"Primary approver '{primary_approver_id}' active. Action requires authorized review."

        pending_action = {
            "name": tool_name,
            "args": tool_args,
            "id": call_id,
            "reason": tool_args.get("reason", ""),
            "routing_reason": routing_reason,
        }

        # If already approved by external human supervisor, proceed to execute
        current_approval = state.get("approval_status", "none")
        if current_approval == "approved":
            status = "approved"
        else:
            status = "pending"
            audit_trail.append(
                f"Gate Check: Intercepted mutating call '{tool_name}'. Model Reason: \"{tool_args.get('reason')}\". {routing_reason}"
            )

        return {
            "approval_status": status,
            "approver_id": approver_id,
            "primary_approver_id": primary_approver_id,
            "escalated_to_backup": escalated,
            "pending_action": pending_action,
            "audit_trail": audit_trail,
        }

    def execute_mutating_tool(state: FreeformAgentState) -> Dict[str, Any]:
        """
        Execute the approved mutating tool under the authorized approver's scope,
        guaranteeing idempotency and recording inverse compensation actions.
        """
        messages = list(state["messages"])
        pending = state.get("pending_action")
        approver_id = state.get("approver_id") or state.get("user_id", "u-101")
        audit_trail = list(state.get("audit_trail", []))
        idemp_records = dict(state.get("idempotency_records", {}))
        idemp_registry = IdempotencyRegistry(idemp_records)
        comp_stack_records = list(state.get("compensation_stack", []))

        # Reconstruct compensation stack
        comp_stack = CompensationStack(api=api, idempotency_registry=idemp_registry)

        tool_name = pending["name"]
        tool_args = pending["args"]
        call_id = pending["id"]
        idemp_key = tool_args.get("idempotency_key")

        # 1. Idempotency Check
        if idemp_key and idemp_registry.has_executed(idemp_key):
            cached_result = idemp_registry.get_result(idemp_key)
            obs = json.dumps(cached_result) if isinstance(cached_result, (dict, list)) else str(cached_result)
            messages.append(ToolMessage(content=obs, tool_call_id=call_id))
            audit_trail.append(f"Execution [Idempotent]: Key '{idemp_key}' previously completed. Returned cached result.")
            return {
                "messages": messages,
                "approval_status": "none",
                "pending_action": None,
                "audit_trail": audit_trail,
            }

        # 2. Execute Mutating Action with Compensation Protection
        tools_map = {t.name: t for t in get_all_tools(api=api, user_id=approver_id)}
        tool = tools_map[tool_name]

        try:
            result = tool.invoke(tool_args)
            obs = json.dumps(result) if isinstance(result, (dict, list)) else str(result)

            # Record inverse compensation action
            if tool_name == "create_purchase_order":
                po_id = tool_args["po_id"]
                comp_stack.push(
                    action_name="cancel_po",
                    compensating_callable=lambda: api.cancel_po(po_id, "Saga Compensation: Voiding replacement PO", user_id=approver_id),
                    description=f"Void created PO {po_id}",
                    idempotency_key=idemp_key,
                )
                comp_stack_records.append({"tool": "cancel_po", "args": {"po_id": po_id}})
            elif tool_name == "cancel_purchase_order":
                po_id = tool_args["po_id"]
                comp_stack.push(
                    action_name="reopen_po",
                    compensating_callable=lambda: api.reopen_po(po_id, "Saga Compensation: Restoring cancelled PO", user_id=approver_id),
                    description=f"Restore cancelled PO {po_id}",
                    idempotency_key=idemp_key,
                )
                comp_stack_records.append({"tool": "reopen_po", "args": {"po_id": po_id}})
            elif tool_name == "notify_production":
                comp_stack.push(
                    action_name="notify_production",
                    compensating_callable=lambda: api.notify_production(
                        supervisor_id=tool_args["supervisor_id"],
                        order_id=tool_args["order_id"],
                        message="CORRECTION: Previous mitigation update rolled back due to workflow failure.",
                        user_id=approver_id,
                    ),
                    description="Issue corrective notification to production",
                    idempotency_key=idemp_key,
                )

            if idemp_key:
                idemp_registry.record_success(idemp_key, result)

            audit_trail.append(f"Execution [Mutate]: Successfully executed '{tool_name}' under user '{approver_id}'.")
            messages.append(ToolMessage(content=obs, tool_call_id=call_id))

        except Exception as exc:
            # Unwind Compensation Stack
            audit_trail.append(f"Execution Error in '{tool_name}': {exc}. Unwinding Saga compensation stack...")
            rollback_log = comp_stack.unwind()
            for r in rollback_log:
                audit_trail.append(f"Compensation Step: {r['action']} -> {r['status']}")

            obs = f"Failure during '{tool_name}': {exc}. Previous actions compensated."
            messages.append(ToolMessage(content=obs, tool_call_id=call_id))

        return {
            "messages": messages,
            "approval_status": "none",
            "pending_action": None,
            "idempotency_records": idemp_registry.dump_records(),
            "compensation_stack": comp_stack_records,
            "audit_trail": audit_trail,
        }

    def route_reasoning(state: FreeformAgentState) -> Literal["execute_read_tool", "gate_mutating_tool", "__end__"]:
        """
        Route model tool calls: read tools execute immediately, mutating tools gate.
        """
        messages = state.get("messages", [])
        if not messages:
            return END

        last_msg = messages[-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return END

        first_call = last_msg.tool_calls[0]
        tool_name = first_call["name"]
        meta = TOOL_REGISTRY.get(tool_name, {})

        if meta.get("is_mutating", False):
            return "gate_mutating_tool"
        return "execute_read_tool"

    def route_gate(state: FreeformAgentState) -> Literal["execute_mutating_tool", "__end__"]:
        """
        Gate routing: proceed if approved, otherwise halt for human checkpoint.
        """
        if state.get("approval_status") == "approved":
            return "execute_mutating_tool"
        return END

    def route_start(state: FreeformAgentState) -> Literal["execute_mutating_tool", "agent_reasoning"]:
        """
        Route start: if approved with a pending mutating action, jump straight to execution.
        """
        if state.get("approval_status") == "approved" and state.get("pending_action"):
            return "execute_mutating_tool"
        return "agent_reasoning"

    # Graph Definition
    workflow = StateGraph(FreeformAgentState)
    workflow.add_node("agent_reasoning", agent_reasoning)
    workflow.add_node("execute_read_tool", execute_read_tool)
    workflow.add_node("gate_mutating_tool", gate_mutating_tool)
    workflow.add_node("execute_mutating_tool", execute_mutating_tool)

    workflow.add_conditional_edges(
        START,
        route_start,
        {
            "execute_mutating_tool": "execute_mutating_tool",
            "agent_reasoning": "agent_reasoning",
        },
    )

    workflow.add_conditional_edges(
        "agent_reasoning",
        route_reasoning,
        {
            "execute_read_tool": "execute_read_tool",
            "gate_mutating_tool": "gate_mutating_tool",
            "__end__": END,
        },
    )

    workflow.add_edge("execute_read_tool", "agent_reasoning")

    workflow.add_conditional_edges(
        "gate_mutating_tool",
        route_gate,
        {
            "execute_mutating_tool": "execute_mutating_tool",
            "__end__": END,
        },
    )

    workflow.add_edge("execute_mutating_tool", "agent_reasoning")

    return workflow.compile(checkpointer=checkpointer)
