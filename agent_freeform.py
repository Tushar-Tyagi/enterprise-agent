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
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from environment.api import SQLiteCompanyAPI
from context_quality import gather_quality_context
from toolcards import TOOL_REGISTRY, build_toolcards_prompt, get_all_tools, get_tools_for_user
from transaction import CompensationStack, IdempotencyRegistry

load_dotenv()


def get_llm_config() -> Dict[str, Any]:
    """
    Resolve LLM provider configuration from environment variables or .env file.
    Supports OpenRouter, Google Gemini direct (AI Studio OpenAI endpoint),
    OpenAI, Local vLLM/Ollama, Groq, and any OpenAI-compliant provider.
    """
    load_dotenv()

    api_key = (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )

    base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        if (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")) and not (
            os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY")
        ):
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        else:
            base_url = "https://openrouter.ai/api/v1"

    model = (
        os.environ.get("LLM_MODEL")
        or os.environ.get("OPENROUTER_MODEL")
        or os.environ.get("OPENAI_MODEL")
    )
    if not model:
        if "generativelanguage.googleapis.com" in base_url or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            model = "gemini-2.5-flash"
        else:
            model = "google/gemini-3.1-flash-lite"

    fallback_model = os.environ.get("LLM_FALLBACK_MODEL")
    if not fallback_model:
        if "generativelanguage.googleapis.com" in base_url or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            fallback_model = "gemini-2.5-pro"
        else:
            fallback_model = "google/gemini-2.5-flash"

    try:
        temperature = float(os.environ.get("LLM_TEMPERATURE", "0"))
    except ValueError:
        temperature = 0.0

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "fallback_model": fallback_model,
        "temperature": temperature,
    }


def get_llm(
    model: Optional[str] = None,
    fallback: bool = False,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """
    Centralized factory for OpenAI-compliant LLM instances.
    Compatible with OpenRouter, Google Gemini (via OpenAI endpoint),
    OpenAI, Local vLLM/Ollama, Groq, Mistral, DeepSeek, etc.
    All endpoint, key, and model settings are driven by the environment (.env).
    """
    config = get_llm_config()
    resolved_key = api_key or config["api_key"]
    if not resolved_key:
        raise ValueError(
            "No LLM API key configured. Provide LLM_API_KEY, GEMINI_API_KEY, "
            "OPENROUTER_API_KEY, or OPENAI_API_KEY in your environment or .env file."
        )

    resolved_base_url = base_url or config["base_url"]

    if model is None:
        selected_model = config["fallback_model"] if fallback else config["model"]
    else:
        selected_model = model

    resolved_temp = config["temperature"] if temperature is None else temperature

    return ChatOpenAI(
        base_url=resolved_base_url,
        api_key=resolved_key,
        model=selected_model,
        temperature=resolved_temp,
        **kwargs,
    )


def extract_llm_telemetry(raw_msg: Any) -> Dict[str, Any]:
    """
    Parse and persist exact telemetry directly from the model response payload.
    Captures prompt_tokens, completion_tokens, total_tokens, prompt_tokens_details,
    cost, and cost_details according to global telemetry capture rules.
    """
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
    token_usage = resp_meta.get("token_usage") or {}
    if not isinstance(token_usage, dict):
        token_usage = {}

    cost_info = (
        token_usage.get("cost")
        or resp_meta.get("cost")
        or 0.0
    )
    cost_details = token_usage.get("cost_details") or resp_meta.get("cost_details") or {}

    prompt_details = usage.get("input_token_details") or usage.get("prompt_tokens_details") or {}
    if not isinstance(prompt_details, dict):
        prompt_details = {}

    return {
        "prompt_tokens": usage.get("input_tokens") or usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("output_tokens") or usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "prompt_tokens_details": prompt_details,
        "cost": float(cost_info) if isinstance(cost_info, (int, float)) else 0.0,
        "cost_details": cost_details,
    }


class FreeformAgentState(TypedDict):
    messages: List[BaseMessage]
    user_id: str
    attention_item: Dict[str, Any]
    pending_tool_calls: List[Dict[str, Any]]
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
        attention = state["attention_item"]
        attention_type = attention.get("type", "")
        is_quality_scenario = attention_type in ("quality_hold_shortage_risk", "quality_hold_impacting_production")

        if is_quality_scenario and (not state.get("user_id") or state.get("user_id") == "u-101"):
            user_id = "u-201"
        else:
            user_id = state.get("user_id", "u-101")

        llm_config = get_llm_config()
        api_key = llm_config.get("api_key")

        # 1. Initialize conversation if first invocation
        if not messages:
            if is_quality_scenario:
                quality_ctx = gather_quality_context(api, attention, user_id=user_id)
                toolcards_doc = quality_ctx["toolcards_doc"]
                supervisor_name = quality_ctx["supervisor_name"]
                supervisor_id = quality_ctx["supervisor_id"]
                order_id = quality_ctx["order_id"]
                part_id = quality_ctx["part_id"]
                sched_start = quality_ctx["scheduled_start"]

                user_obj = api.get_user(user_id)
                user_name = user_obj.get("name", "Casey Chen")
                system_prompt = (
                    f"You are the personal AI operational assistant for {user_name} ({user_id}), Quality Manager at Northfield Manufacturing.\n"
                    f"You work directly for {user_name}. Always speak directly to them in the second person ('you', 'your order', 'you declined').\n"
                    "Your objective is to protect manufacturing schedules from quality defects and stockouts autonomously.\n"
                    "You are scoped to 'erp:quality:*', 'erp:production:read', 'mail:*', and 'production:notify'.\n"
                    "You do NOT have authority to create or cancel purchase orders.\n\n"
                    "OPERATIONAL WORKFLOW FOR QUALITY SHORTAGE:\n"
                    f"1. Inspect quality lots for part {part_id} using get_quality_lots.\n"
                    "2. Determine if an inspected, available lot covers the production requirement.\n"
                    "   - CRITICAL: Never select lots that are on 'hold' (beware of hold traps with defects).\n"
                    f"3. If an available candidate lot exists:\n"
                    f"   (a) Reallocate the lot to Production Order {order_id} using reallocate_lot_for_order.\n"
                    f"   (b) Notify Production Supervisor {supervisor_name} ({supervisor_id}) using notify_production.\n"
                    f"4. If NO available lot exists:\n"
                    f"   - Immediately flag the critical shortage to Purchasing Manager Dana Whitfield (u-101) using send_email.\n\n"
                    "CRITICAL POLICIES:\n"
                    "1. Every tool call MUST provide a detailed 'reason' parameter explaining your operational justification.\n"
                    "2. Mutating actions (reallocating lots, notifications, sending emails) are intercepted by human gating.\n"
                    "3. Each mutating call must specify a unique 'idempotency_key'.\n"
                    "4. Human Authorization Feedback: You work directly for {user_name}. If they decline a mutating action, you will receive a ToolMessage beginning with 'DECLINED by you'. "
                    "Do NOT attempt to re-execute the rejected mutating tool. Address them directly in the second person ('You declined...'). Deliberate on the operational consequences, explain clearly how declining risks production line starvation, and advise on manual next steps.\n\n"
                    f"{toolcards_doc}"
                )
                human_prompt = (
                    f"Attention Item Received:\n"
                    f"Type: quality_hold_shortage_risk\n"
                    f"Production Order: {order_id} (Scheduled Start: {sched_start})\n"
                    f"Part Required: {part_id} (Quantity: {quality_ctx.get('order_quantity')})\n"
                    f"Current Lot on Hold: {attention.get('allocated_lot')} (Reason: {attention.get('hold_reason', 'Quality specification failure')})\n\n"
                    f"Investigate quality lots for part {part_id}. If an inspected good lot is available, reallocate it to Order {order_id} "
                    f"and notify Supervisor {supervisor_name} ({supervisor_id}). If no good lot covers it, flag a shortage to purchasing."
                )
            else:
                user_obj = api.get_user(user_id)
                user_name = user_obj.get("name", "Dana Whitfield")
                toolcards_doc = build_toolcards_prompt()
                system_prompt = (
                    f"You are the personal AI operational assistant for {user_name} ({user_id}), Purchasing Manager at Northfield Manufacturing.\n"
                    f"You work directly for {user_name}. Always speak directly to her in the second person ('you', 'your order', 'you declined').\n"
                    "Your objective is to resolve operational disruptions and stockout risks autonomously.\n"
                    "Inspect incoming signals, emails, POs, and production schedules, evaluate approved suppliers, "
                    "and execute the full mitigation workflow:\n"
                    "  Step 1: Cancel the delayed purchase order (PO-77812) with Supplier Y.\n"
                    "  Step 2: Create a replacement purchase order with approved alternate Supplier Z (supplier_id: 'S-Z') for 50 units of P-4471 at $210.00/unit, promising delivery on 2026-09-04 (2-day lead time). Assign new PO ID 'PO-77815' (avoiding existing PO-77812, PO-77813, PO-77814).\n"
                    "  Step 3: Notify the production supervisor (user ID: 'u-301' / Sam Taylor) regarding Production Order 4812 resolution.\n"
                    "  Step 4: Schedule a deferred task for next Tuesday (2026-09-08) to verify shipment arrival using schedule_event:\n"
                    "          trigger_date='2026-09-08', target_table='Mail', payload={'mail_id': 'M-CHECK-PO-77815', 'sender': 'system@enterprise.internal', 'recipient_id': 'u-101', 'subject': 'Automated Check: Verify Arrival of Replacement PO PO-77815', 'body': 'Confirm shipment arrived from Supplier Z for Production Order 4812.', 'sent_at': '2026-09-08T08:00:00', 'read_status': 0}.\n\n"
                    "CRITICAL POLICIES:\n"
                    "1. Every single tool call MUST provide a detailed 'reason' parameter justifying the call.\n"
                    "   Generic phrases like 'mitigate disruption' or 'cancel delayed PO' are STRICTLY PROHIBITED.\n"
                    "   For any mutating action (create_purchase_order, cancel_purchase_order, notify_production, schedule_event), "
                    "   your 'reason' MUST explicitly include:\n"
                    "   (a) What production order is delayed and its scheduled start date (e.g. Order 4812 starting 2026-09-07).\n"
                    "   (b) The exact delay details (e.g. Supplier Y delayed PO-77812 to 2026-09-08, slipping past start).\n"
                    "   (c) The chosen alternate supplier, unit price, lead time, and promised delivery date before production starts (e.g. Supplier Z / S-Z delivering on 2026-09-04).\n"
                    "2. All mutating tools (creating/cancelling POs, notifications, scheduling) are intercepted by human gating.\n"
                    "3. Each mutating call must specify a unique 'idempotency_key'.\n"
                    "4. Operational Directory: Production Supervisor for Order 4812 is Sam Taylor (user ID: 'u-301'). When calling notify_production, set supervisor_id='u-301' and order_id='4812'.\n"
                    f"5. Human Authorization Feedback: You work directly for {user_name}. If she declines a mutating action, you will receive a ToolMessage beginning with 'DECLINED by you'. "
                    "Do NOT attempt to re-execute the rejected mutating tool. Address her directly in the second person ('You declined...'). Deliberate on the operational consequences, explain clearly why declining this action risks production line starvation or missed start dates, and recommend concrete manual actions she should take.\n\n"
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

        # 2. Check if human approver declined a tool call or state is marked declined
        has_declined = (
            state.get("approval_status") == "declined"
            or any(isinstance(m, ToolMessage) and "DECLINED" in str(m.content) for m in messages)
        )
        if has_declined:
            current_hold_lot = attention.get("allocated_lot") or attention.get("lot_id", "L-2093")
            order_id = attention.get("order_id") or attention.get("production_order_id", "4820")
            part_id = attention.get("part_id", "P-1180")
            po_id = attention.get("po_id", "PO-77812")
            prod_id = attention.get("production_order_id", "4812")
            user_obj = api.get_user(user_id)
            user_name = user_obj.get("name", "Dana Whitfield") if user_obj else "Dana Whitfield"

            warning_content = ""
            if api_key:
                try:
                    llm = get_llm()
                    decline_prompt = (
                        f"You are the personal AI operational assistant for {user_name} ({user_id}) at Northfield Manufacturing.\n"
                        f"Address them directly in the second person ('You declined...').\n"
                        f"They have declined a proposed operational action. All partial database mutations were safely rolled back via Saga compensation.\n"
                        f"CRITICAL INSTRUCTION: Do NOT emit any tool calls or function calls. Output only plain conversational text.\n"
                        f"Deliver a direct, urgent operational warning explaining the immediate consequences to the production schedule, "
                        f"how declining this action leaves the manufacturing line at risk of starvation or delay, "
                        f"and recommend concrete manual next steps they should take."
                    )
                    prompt_messages = [
                        SystemMessage(content=decline_prompt),
                        *messages[1:],
                    ]
                    ai_res = llm.invoke(prompt_messages)
                    if hasattr(ai_res, "content") and isinstance(ai_res.content, str) and ai_res.content.strip():
                        warning_content = ai_res.content.strip()

                    turn_telem = extract_llm_telemetry(ai_res)
                    telemetry["prompt_tokens"] = telemetry.get("prompt_tokens", 0) + turn_telem["prompt_tokens"]
                    telemetry["completion_tokens"] = telemetry.get("completion_tokens", 0) + turn_telem["completion_tokens"]
                    telemetry["total_tokens"] = telemetry.get("total_tokens", 0) + turn_telem["total_tokens"]
                    telemetry["prompt_tokens_details"] = turn_telem.get("prompt_tokens_details", {})
                    telemetry["cost"] = telemetry.get("cost", 0.0) + turn_telem["cost"]
                    telemetry["cost_details"] = turn_telem.get("cost_details", {})
                except Exception:
                    warning_content = ""

            if not warning_content:
                if is_quality_scenario:
                    warning_content = (
                        f"Warning: You declined the lot reallocation for Production Order {order_id}. "
                        f"Because defective lot {current_hold_lot} was removed and no replacement lot was assigned, "
                        f"Order {order_id} will be short of the 25 required units of {part_id} when scheduled to start on {attention.get('scheduled_start', '2026-09-05')}. "
                        f"I have rolled back all partial database changes via Saga compensation so your records remain clean. "
                        f"You will need to manually expedite alternative lots or work with production to reschedule this run."
                    )
                else:
                    warning_content = (
                        f"Warning: You declined the mitigating purchase order action for Production Order {prod_id}. "
                        f"Because Supplier Y delayed PO {po_id} to {attention.get('delayed_promised_date', '2026-09-08')}, "
                        f"Production Order {prod_id} will run out of Part {attention.get('part_id', 'P-4471')} and miss its scheduled start on {attention.get('production_scheduled_start', '2026-09-07')}. "
                        f"I have rolled back all partial changes via Saga compensation, restoring {po_id} to OPEN status with Supplier Y. "
                        f"You will need to manually expedite delivery or adjust the production schedule for Line 2 immediately."
                    )

            ai_msg = AIMessage(content=warning_content)
            ai_msg.tool_calls = []
            audit_trail.append("Agent Reasoning: Deliberated on user decline and issued personal operational warning.")
            messages.append(ai_msg)
            return {
                "messages": messages,
                "pending_tool_calls": [],
                "pending_action": None,
                "approval_status": "declined",
                "audit_trail": audit_trail,
                "llm_telemetry": telemetry,
            }

        # 3. Invoke Model or Deterministic Mock Loop if key is absent
        if not api_key:

            # Deterministic simulation of free-flowing exploration
            tool_messages_count = sum(1 for m in messages if isinstance(m, ToolMessage))

            if is_quality_scenario:
                order_id = attention.get("order_id") or attention.get("production_order_id", "4820")
                part_id = attention.get("part_id", "P-1180")
                current_hold_lot = attention.get("allocated_lot") or attention.get("lot_id", "L-2093")
                supervisor_id = attention.get("supervisor_id", "u-301")

                lots = api.get_quality_lots(part_id=part_id)
                allocated_to_order = [l for l in lots if l.get("allocated_order_id") == order_id]
                reallocated_good_lot = any(l["lot_id"] != current_hold_lot and l["status"] == "available" for l in allocated_to_order)
                notified_supervisor = len(api.get_production_notifications(order_id)) > 0

                emails = api.get_emails(recipient_id="u-101")
                shortage_emailed = any("Shortage" in e.get("subject", "") and order_id in e.get("subject", "") for e in emails)
                available_candidates = [l for l in lots if l["status"] == "available" and not l.get("allocated_order_id")]

                if (reallocated_good_lot and notified_supervisor) or shortage_emailed or tool_messages_count >= 5:
                    if reallocated_good_lot:
                        target_lot = allocated_to_order[0]["lot_id"]
                        ai_msg = AIMessage(
                            content=(
                                f"Quality mitigation complete: Production Order {order_id} has been reallocated from "
                                f"defective lot {current_hold_lot} to verified lot {target_lot}. Supervisor {supervisor_id} has been notified."
                            )
                        )
                    else:
                        ai_msg = AIMessage(
                            content=(
                                f"Quality shortage flagged: No available lot could cover Order {order_id}. "
                                f"Purchasing Manager Dana Whitfield (u-101) has been alerted via email."
                            )
                        )
                    audit_trail.append("Agent Reasoning: Quality mitigation successfully completed.")
                elif tool_messages_count == 0:
                    ai_msg = AIMessage(
                        content=f"Inspecting all quality lots for part {part_id} to evaluate candidate coverage and identify hold status.",
                        tool_calls=[{
                            "name": "get_quality_lots",
                            "args": {
                                "part_id": part_id,
                                "reason": f"Inspect quality status of all inventory lots for part {part_id} needed by Production Order {order_id}",
                            },
                            "id": "call-qual-lots-001",
                        }],
                    )
                    audit_trail.append(f"Agent Action: Inspecting quality lots for part {part_id}.")
                elif tool_messages_count == 1:
                    if available_candidates:
                        candidate_lot_id = available_candidates[0]["lot_id"]
                        ai_msg = AIMessage(
                            content=f"Found available lot {candidate_lot_id}. Initiating reallocation to Production Order {order_id}.",
                            tool_calls=[{
                                "name": "reallocate_lot_for_order",
                                "args": {
                                    "order_id": order_id,
                                    "from_lot_id": current_hold_lot,
                                    "to_lot_id": candidate_lot_id,
                                    "idempotency_key": f"idemp-realloc-{order_id}-{candidate_lot_id}",
                                    "reason": (
                                        f"Current lot {current_hold_lot} is on quality hold ({attention.get('hold_reason', 'Defect')}), "
                                        f"threatening Production Order {order_id} scheduled for {attention.get('scheduled_start', '2026-09-05')}. "
                                        f"Reallocating verified available lot {candidate_lot_id} to prevent line starvation."
                                    ),
                                },
                                "id": "call-realloc-002",
                            }],
                        )
                        audit_trail.append(f"Agent Action: Proposing lot reallocation from {current_hold_lot} to {candidate_lot_id}.")
                    else:
                        ai_msg = AIMessage(
                            content=f"No available lot found for part {part_id}. Flagging shortage to purchasing manager Dana Whitfield (u-101).",
                            tool_calls=[{
                                "name": "send_email",
                                "args": {
                                    "recipient_id": "u-101",
                                    "subject": f"URGENT: Material Shortage for Order {order_id} (Part {part_id})",
                                    "body": (
                                        f"Production Order {order_id} requires 25 units of {part_id} on {attention.get('scheduled_start', '2026-09-05')}. "
                                        f"Current lot {current_hold_lot} is on quality hold and no alternate lots are available. Please expedite purchase order."
                                    ),
                                    "idempotency_key": f"idemp-shortage-{order_id}",
                                    "reason": f"No available lots exist for part {part_id} to replace hold lot {current_hold_lot}.",
                                },
                                "id": "call-email-shortage-002",
                            }],
                        )
                        audit_trail.append("Agent Action: Flagging material shortage to Purchasing Manager.")
                elif tool_messages_count == 2:
                    if reallocated_good_lot or available_candidates:
                        target_lot = allocated_to_order[0]["lot_id"] if allocated_to_order else available_candidates[0]["lot_id"]
                        ai_msg = AIMessage(
                            content=f"Notifying Production Supervisor {supervisor_id} regarding lot reallocation.",
                            tool_calls=[{
                                "name": "notify_production",
                                "args": {
                                    "supervisor_id": supervisor_id,
                                    "order_id": order_id,
                                    "message": (
                                        f"Quality resolution for Order {order_id}: Defective lot {current_hold_lot} released. "
                                        f"Reallocated inspected lot {target_lot}. Order start date {attention.get('scheduled_start', '2026-09-05')} is protected."
                                    ),
                                    "idempotency_key": f"idemp-notify-qual-{order_id}",
                                    "reason": f"Inform supervisor that Production Order {order_id} has a valid inspected lot allocated.",
                                },
                                "id": "call-notify-qual-003",
                            }],
                        )
                        audit_trail.append(f"Agent Action: Notifying Production Supervisor for Order {order_id}.")
                    else:
                        ai_msg = AIMessage(
                            content=f"Mitigation complete: Purchasing manager notified of shortage for Order {order_id}."
                        )
                        audit_trail.append("Agent Reasoning: Mitigation completed.")
                else:
                    ai_msg = AIMessage(
                        content=f"Quality mitigation workflow completed for Production Order {order_id}."
                    )
                    audit_trail.append("Agent Reasoning: Mitigation completed.")
            else:
                # Scenario A: Purchasing
                try:
                    cancelled = api.get_purchase_order(attention.get("po_id", ""))["status"] == "CANCELLED"
                except Exception:
                    cancelled = False
                try:
                    created = api.get_purchase_order("PO-77815")["status"] == "OPEN"
                except Exception:
                    created = False
                try:
                    notified = len(api.get_production_notifications(attention.get("production_order_id", ""))) > 0
                except Exception:
                    notified = False

                if (cancelled and created and notified) or tool_messages_count >= 5:
                    ai_msg = AIMessage(
                        content=(
                            f"Operational mitigation complete: Delayed PO {attention['po_id']} has been cancelled and replaced "
                            f"with PO-77815 from Supplier Z. Production order {attention['production_order_id']} is protected."
                        )
                    )
                    audit_trail.append("Agent Reasoning: Mitigation successfully completed.")
                elif tool_messages_count == 0:
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
                "note": "Mock agent cycle (No LLM API key configured)",
            }
        else:
            # Live LLM Call (OpenAI-compliant endpoint configured via .env)
            tools = get_tools_for_user(api=api, user_id=user_id)

            try:
                llm = get_llm()
                llm_with_tools = llm.bind_tools(tools)
                ai_msg = llm_with_tools.invoke(messages)
            except Exception:
                llm = get_llm(fallback=True)
                llm_with_tools = llm.bind_tools(tools)
                ai_msg = llm_with_tools.invoke(messages)

            messages.append(ai_msg)

            turn_telem = extract_llm_telemetry(ai_msg)

            # Accumulate across multiple tool-calling turns in free-form mode
            prev_telemetry = dict(state.get("llm_telemetry", {}))
            prev_prompt = prev_telemetry.get("prompt_tokens", 0)
            prev_completion = prev_telemetry.get("completion_tokens", 0)
            prev_total = prev_telemetry.get("total_tokens", 0)
            prev_cost = prev_telemetry.get("cost", 0.0)

            telemetry = {
                "prompt_tokens": prev_prompt + turn_telem["prompt_tokens"],
                "completion_tokens": prev_completion + turn_telem["completion_tokens"],
                "total_tokens": prev_total + turn_telem["total_tokens"],
                "prompt_tokens_details": turn_telem.get("prompt_tokens_details", {}),
                "cost": prev_cost + turn_telem["cost"],
                "cost_details": turn_telem.get("cost_details", {}),
            }

        tool_calls = list(ai_msg.tool_calls) if getattr(ai_msg, "tool_calls", None) else []

        return {
            "messages": messages,
            "pending_tool_calls": tool_calls,
            "audit_trail": audit_trail,
            "llm_telemetry": telemetry,
        }

    def execute_read_tool(state: FreeformAgentState) -> Dict[str, Any]:
        """
        Execute read-only tool calls immediately and return observation to model.
        Processes consecutive read calls from the front of pending_tool_calls.
        """
        messages = list(state["messages"])
        user_id = state.get("user_id", "u-101")
        audit_trail = list(state.get("audit_trail", []))
        tools_map = {t.name: t for t in get_all_tools(api=api, user_id=user_id)}

        pending_calls = state.get("pending_tool_calls")
        if pending_calls is None:
            if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
                pending_calls = list(messages[-1].tool_calls)
            else:
                pending_calls = []
        else:
            pending_calls = list(pending_calls)

        while pending_calls:
            first_call = pending_calls[0]
            tool_name = first_call["name"]
            meta = TOOL_REGISTRY.get(tool_name, {})
            if meta.get("is_mutating", False):
                break

            call = pending_calls.pop(0)
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
            "pending_tool_calls": pending_calls,
            "audit_trail": audit_trail,
        }

    def gate_mutating_tool(state: FreeformAgentState) -> Dict[str, Any]:
        """
        Intercept mutating tool calls, evaluate spending limits and out-of-office rules,
        and pause for human authorization.
        """
        messages = list(state["messages"])
        user_id = state.get("user_id", "u-101")
        primary_approver_id = state.get("primary_approver_id") or user_id
        audit_trail = list(state.get("audit_trail", []))

        pending_calls = state.get("pending_tool_calls")
        if pending_calls is None:
            if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
                pending_calls = list(messages[-1].tool_calls)
            else:
                pending_calls = []
        else:
            pending_calls = list(pending_calls)

        if not pending_calls:
            return {
                "approval_status": "none",
                "pending_action": None,
                "pending_tool_calls": [],
            }

        call = pending_calls[0]
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
            "pending_tool_calls": pending_calls,
            "audit_trail": audit_trail,
        }

    def execute_mutating_tool(state: FreeformAgentState) -> Dict[str, Any]:
        """
        Execute the approved mutating tool under the authorized approver's scope,
        guaranteeing idempotency and recording inverse compensation actions.
        """
        messages = list(state["messages"])
        pending = state.get("pending_action")
        if not pending:
            return {
                "approval_status": "none",
                "pending_action": None,
                "pending_tool_calls": list(state.get("pending_tool_calls") or []),
            }

        approver_id = state.get("approver_id") or state.get("user_id", "u-101")
        audit_trail = list(state.get("audit_trail", []))
        idemp_records = dict(state.get("idempotency_records", {}))
        idemp_registry = IdempotencyRegistry(idemp_records)
        comp_stack_records = list(state.get("compensation_stack", []))
        # Reconstruct compensation stack from persisted records
        comp_stack = CompensationStack(api=api, idempotency_registry=idemp_registry)
        for rec in comp_stack_records:
            t = rec.get("tool")
            a = rec.get("args", {})
            if t == "cancel_po":
                pid = a.get("po_id")
                comp_stack.push("cancel_po", lambda p=pid: api.cancel_po(p, "Saga Compensation", user_id=approver_id), description=f"Void PO {pid}")
            elif t == "reopen_po":
                pid = a.get("po_id")
                comp_stack.push("reopen_po", lambda p=pid: api.reopen_po(p, "Saga Compensation", user_id=approver_id), description=f"Reopen PO {pid}")
            elif t == "reallocate_lot_for_order":
                oid = a.get("order_id")
                fl = a.get("from_lot_id")
                tl = a.get("to_lot_id")
                def _restore(o=oid, f=fl, t=tl):
                    with api.conn:
                        api.conn.execute("UPDATE QualityLots SET allocated_order_id = ? WHERE lot_id = ?;", (o, t))
                        api.conn.execute("UPDATE QualityLots SET allocated_order_id = NULL WHERE lot_id = ?;", (f,))
                comp_stack.push("reallocate_lot_for_order", _restore, description=f"Restore lot {tl} to order {oid}")
            elif t == "notify_production":
                comp_stack.push("notify_production", lambda: api.notify_production(supervisor_id=a.get("supervisor_id", "u-301"), order_id=a.get("order_id", "4820"), message="CORRECTION: Mitigation rolled back.", user_id=approver_id), description="Corrective notify")


        tool_name = pending["name"]
        tool_args = pending["args"]
        call_id = pending["id"]
        idemp_key = tool_args.get("idempotency_key")

        # Pop completed action from pending_tool_calls
        pending_calls = list(state.get("pending_tool_calls") or [])
        if pending_calls:
            if pending_calls[0].get("id") == call_id:
                pending_calls.pop(0)
            else:
                pending_calls = [c for c in pending_calls if c.get("id") != call_id]

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
                "pending_tool_calls": pending_calls,
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
                created_po_id = result.get("po_id") if isinstance(result, dict) else tool_args["po_id"]
                comp_stack.push(
                    action_name="cancel_po",
                    compensating_callable=lambda pid=created_po_id: api.cancel_po(pid, "Saga Compensation: Voiding replacement PO", user_id=approver_id),
                    description=f"Void created PO {created_po_id}",
                    idempotency_key=idemp_key,
                )
                comp_stack_records.append({"tool": "cancel_po", "args": {"po_id": created_po_id}})
            elif tool_name == "cancel_purchase_order":
                cancelled_po_id = tool_args["po_id"]
                comp_stack.push(
                    action_name="reopen_po",
                    compensating_callable=lambda pid=cancelled_po_id: api.reopen_po(pid, "Saga Compensation: Restoring cancelled PO", user_id=approver_id),
                    description=f"Restore cancelled PO {cancelled_po_id}",
                    idempotency_key=idemp_key,
                )
                comp_stack_records.append({"tool": "reopen_po", "args": {"po_id": cancelled_po_id}})
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
            elif tool_name == "schedule_event":
                event_id = result if isinstance(result, int) else None
                if event_id:
                    comp_stack.push(
                        action_name="cancel_scheduled_event",
                        compensating_callable=lambda eid=event_id: api.conn.execute("DELETE FROM ScheduledEvents WHERE event_id = ?;", (eid,)),
                        description=f"Remove scheduled event {event_id}",
                        idempotency_key=idemp_key,
                    )
            elif tool_name == "reallocate_lot_for_order":
                oid = tool_args["order_id"]
                orig_lot = tool_args["from_lot_id"]
                alt_lot = tool_args["to_lot_id"]

                def _restore_lot_allocation(o=oid, orig=orig_lot, alt=alt_lot):
                    with api.conn:
                        api.conn.execute("UPDATE QualityLots SET allocated_order_id = ? WHERE lot_id = ?;", (o, orig))
                        api.conn.execute("UPDATE QualityLots SET allocated_order_id = NULL WHERE lot_id = ?;", (alt,))

                comp_stack.push(
                    action_name="reallocate_lot_for_order",
                    compensating_callable=_restore_lot_allocation,
                    description=f"Restore lot {orig_lot} to order {oid} and release {alt_lot}",
                    idempotency_key=idemp_key,
                )
                comp_stack_records.append({
                    "tool": "reallocate_lot_for_order",
                    "args": {
                        "order_id": oid,
                        "from_lot_id": alt_lot,
                        "to_lot_id": orig_lot,
                    },
                })
            elif tool_name == "send_email":
                comp_stack_records.append({"tool": "send_email", "args": {"recipient_id": tool_args["recipient_id"]}})

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
            pending_calls = []

        return {
            "messages": messages,
            "approval_status": "none",
            "pending_action": None,
            "pending_tool_calls": pending_calls,
            "idempotency_records": idemp_registry.dump_records(),
            "compensation_stack": comp_stack_records,
            "audit_trail": audit_trail,
        }

    def route_reasoning(state: FreeformAgentState) -> Literal["execute_read_tool", "gate_mutating_tool", "__end__"]:
        """
        Route model tool calls: read tools execute immediately, mutating tools gate.
        """
        pending_calls = state.get("pending_tool_calls")
        if pending_calls is None:
            messages = state.get("messages", [])
            if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
                pending_calls = list(messages[-1].tool_calls)
            else:
                pending_calls = []
        else:
            pending_calls = list(pending_calls)

        if not pending_calls:
            return END

        first_call = pending_calls[0]
        tool_name = first_call["name"]
        meta = TOOL_REGISTRY.get(tool_name, {})

        if meta.get("is_mutating", False):
            return "gate_mutating_tool"
        return "execute_read_tool"

    def route_after_read(state: FreeformAgentState) -> Literal["execute_read_tool", "gate_mutating_tool", "agent_reasoning"]:
        """
        Route after read tools: if there are remaining pending calls (e.g. mutating tools), gate them;
        otherwise return to agent reasoning.
        """
        pending_calls = list(state.get("pending_tool_calls") or [])
        if not pending_calls:
            return "agent_reasoning"
        first = pending_calls[0]
        meta = TOOL_REGISTRY.get(first["name"], {})
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

    def route_after_mutate(state: FreeformAgentState) -> Literal["execute_read_tool", "gate_mutating_tool", "agent_reasoning"]:
        """
        Route after mutating action execution: if more calls are queued in this turn,
        gate the next mutating call or execute next read tool; otherwise return to agent reasoning.
        """
        pending_calls = list(state.get("pending_tool_calls") or [])
        if not pending_calls:
            return "agent_reasoning"
        first = pending_calls[0]
        meta = TOOL_REGISTRY.get(first["name"], {})
        if meta.get("is_mutating", False):
            return "gate_mutating_tool"
        return "execute_read_tool"

    def route_start(state: FreeformAgentState) -> Literal["execute_mutating_tool", "gate_mutating_tool", "execute_read_tool", "agent_reasoning"]:
        """
        Route start: if approved with a pending mutating action, jump straight to execution.
        If pending_action is already staged, jump to gate.
        If pending_tool_calls has unfulfilled actions from an AIMessage, resume them.
        Otherwise proceed to agent reasoning.
        """
        if state.get("approval_status") == "approved" and state.get("pending_action"):
            return "execute_mutating_tool"
        if state.get("pending_action"):
            return "gate_mutating_tool"

        pending_calls = state.get("pending_tool_calls")
        if pending_calls is None:
            messages = state.get("messages", [])
            if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
                pending_calls = list(messages[-1].tool_calls)
            else:
                pending_calls = []
        else:
            pending_calls = list(pending_calls)

        if pending_calls:
            first = pending_calls[0]
            meta = TOOL_REGISTRY.get(first["name"], {})
            if meta.get("is_mutating", False):
                return "gate_mutating_tool"
            return "execute_read_tool"

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
            "gate_mutating_tool": "gate_mutating_tool",
            "execute_read_tool": "execute_read_tool",
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

    workflow.add_conditional_edges(
        "execute_read_tool",
        route_after_read,
        {
            "execute_read_tool": "execute_read_tool",
            "gate_mutating_tool": "gate_mutating_tool",
            "agent_reasoning": "agent_reasoning",
        },
    )

    workflow.add_conditional_edges(
        "gate_mutating_tool",
        route_gate,
        {
            "execute_mutating_tool": "execute_mutating_tool",
            "__end__": END,
        },
    )

    workflow.add_conditional_edges(
        "execute_mutating_tool",
        route_after_mutate,
        {
            "execute_read_tool": "execute_read_tool",
            "gate_mutating_tool": "gate_mutating_tool",
            "agent_reasoning": "agent_reasoning",
        },
    )

    return workflow.compile(checkpointer=checkpointer)
