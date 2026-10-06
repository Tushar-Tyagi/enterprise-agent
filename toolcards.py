"""
Centralized Toolcards and Schema Registry for Enterprise Agent.
Defines all tool schemas, RBAC scope requirements, metadata, and prompt representation.
"""

from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from environment.api import SQLiteCompanyAPI


# ---------------------------------------------------------------------------
# Base Schema Requiring Model Rationale
# ---------------------------------------------------------------------------

class BaseToolInput(BaseModel):
    reason: str = Field(
        ...,
        description=(
            "Comprehensive operational justification for this tool call. "
            "For mutating actions (creating/cancelling purchase orders or notifications), the reason MUST explicitly include: "
            "(1) the impacted production order ID and scheduled start date, "
            "(2) the delayed supplier name, original date, and delayed date showing how much delay/slip occurs, and "
            "(3) the alternate supplier chosen, unit price, lead time, and promised delivery date."
        ),
    )


# ---------------------------------------------------------------------------
# Read Tool Schemas
# ---------------------------------------------------------------------------

class ReadEmailsInput(BaseToolInput):
    recipient_id: Optional[str] = Field(
        default=None,
        description="User ID whose inbox to check. If omitted, checks active user's inbox.",
    )
    unread_only: bool = Field(
        default=True,
        description="Whether to fetch only unread messages.",
    )


class GetPurchaseOrderInput(BaseToolInput):
    po_id: str = Field(
        ...,
        description="Unique identifier of the purchase order (e.g. 'PO-77812').",
    )


class GetProductionOrdersInput(BaseToolInput):
    part_id: Optional[str] = Field(
        default=None,
        description="Optional part ID filter to find production runs consuming a specific part.",
    )


class QuerySuppliersInput(BaseToolInput):
    part_id: Optional[str] = Field(
        default=None,
        description="Filter suppliers by approved parts they can supply.",
    )
    approved_only: bool = Field(
        default=True,
        description="If True, only return approved suppliers.",
    )


class GetCalendarEventsInput(BaseToolInput):
    user_id: Optional[str] = Field(
        default=None,
        description="User ID whose calendar events to query. Defaults to active user.",
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Start date filter in YYYY-MM-DD format.",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date filter in YYYY-MM-DD format.",
    )


class GetQualityLotsInput(BaseToolInput):
    part_id: Optional[str] = Field(
        default=None,
        description="Filter quality lots by part identifier (e.g. 'P-1180').",
    )
    status: Optional[str] = Field(
        default=None,
        description="Filter quality lots by status ('available' or 'hold').",
    )
    allocated_order_id: Optional[str] = Field(
        default=None,
        description="Filter by allocated production order ID.",
    )


class GetQualityLotInput(BaseToolInput):
    lot_id: str = Field(
        ...,
        description="Unique lot identifier (e.g. 'L-2093', 'L-2094').",
    )


# ---------------------------------------------------------------------------
# Mutating Tool Schemas
# ---------------------------------------------------------------------------

class CreatePurchaseOrderInput(BaseToolInput):
    po_id: str = Field(
        default="PO-77815",
        description="Unique PO identifier to assign. Recommended: 'PO-77815' (avoids collision with existing PO-77812, PO-77813, PO-77814).",
    )
    part_id: str = Field(..., description="Part identifier being ordered (e.g. 'P-4471').")
    supplier_id: str = Field(..., description="Target supplier identifier (e.g. 'S-Z').")
    quantity: int = Field(..., description="Quantity of parts to order.")
    unit_price: float = Field(..., description="Negotiated unit price.")
    promised_date: str = Field(..., description="Promised arrival date in YYYY-MM-DD format.")
    idempotency_key: str = Field(..., description="Unique idempotency client key to prevent duplicate creation.")


class CancelPurchaseOrderInput(BaseToolInput):
    po_id: str = Field(..., description="PO identifier to cancel (e.g. 'PO-77812').")
    idempotency_key: str = Field(..., description="Unique idempotency client key.")


class SendEmailInput(BaseToolInput):
    recipient_id: str = Field(..., description="Target recipient user ID (e.g. 'u-301' for Production Supervisor Sam Taylor).")
    subject: str = Field(..., description="Subject of the email.")
    body: str = Field(..., description="Body text of the email message.")
    idempotency_key: str = Field(..., description="Unique idempotency client key.")


class NotifyProductionInput(BaseToolInput):
    supervisor_id: str = Field(..., description="Supervisor user ID to notify (e.g. 'u-301' for Production Supervisor Sam Taylor).")
    order_id: str = Field(..., description="Production order identifier impacted (e.g. '4812').")
    message: str = Field(..., description="Detailed operational alert message.")
    idempotency_key: str = Field(..., description="Unique idempotency client key.")


class ScheduleEventInput(BaseToolInput):
    trigger_date: str = Field(..., description="Date the event should mature (YYYY-MM-DD), e.g. '2026-09-08'.")
    target_table: str = Field(..., description="Target table to materialize into (e.g. 'Mail').")
    payload: Dict[str, Any] = Field(
        ...,
        description=(
            "Event payload dictionary. For target_table='Mail', provide: 'mail_id' (e.g. 'M-CHECK-PO-77815'), "
            "'recipient_id' ('u-101'), 'sender' ('system@enterprise.internal'), 'subject', and 'body'."
        ),
    )
    idempotency_key: str = Field(..., description="Unique idempotency client key.")


class ReallocateLotForOrderInput(BaseToolInput):
    order_id: str = Field(..., description="Target production order ID (e.g. '4820').")
    from_lot_id: str = Field(..., description="Current lot ID on hold to release (e.g. 'L-2093').")
    to_lot_id: str = Field(..., description="Available candidate lot ID to allocate (e.g. 'L-2094').")
    idempotency_key: str = Field(..., description="Unique idempotency client key.")


# ---------------------------------------------------------------------------
# Tool Registry Definitions
# ---------------------------------------------------------------------------

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "read_emails": {
        "schema": ReadEmailsInput,
        "is_mutating": False,
        "required_scope": "mail:read",
        "description": "Read email messages from the user's inbox, optionally filtering by unread status.",
    },
    "get_purchase_order": {
        "schema": GetPurchaseOrderInput,
        "is_mutating": False,
        "required_scope": "erp:po:read",
        "description": "Inspect purchase order details, status, promised dates, and line item amounts.",
    },
    "get_production_orders": {
        "schema": GetProductionOrdersInput,
        "is_mutating": False,
        "required_scope": "erp:production:read",
        "description": "Inspect scheduled manufacturing runs and check required parts and start dates.",
    },
    "query_suppliers": {
        "schema": QuerySuppliersInput,
        "is_mutating": False,
        "required_scope": "erp:po:read",
        "description": "Search active suppliers, approved catalogs, unit pricing, and lead time days.",
    },
    "get_calendar_events": {
        "schema": GetCalendarEventsInput,
        "is_mutating": False,
        "required_scope": "calendar:read",
        "description": "Query calendar schedules and out-of-office dates for a given user.",
    },
    "get_quality_lots": {
        "schema": GetQualityLotsInput,
        "is_mutating": False,
        "required_scope": "erp:quality:read",
        "description": "Query inspected lots, part associations, quality hold status, and allocated orders.",
    },
    "get_quality_lot": {
        "schema": GetQualityLotInput,
        "is_mutating": False,
        "required_scope": "erp:quality:read",
        "description": "Inspect specific quality lot details, hold reasons, and test timestamps.",
    },
    "create_purchase_order": {
        "schema": CreatePurchaseOrderInput,
        "is_mutating": True,
        "required_scope": "erp:po:create",
        "description": "Create a new purchase order with an alternate supplier. Requires human approval if spending thresholds apply.",
    },
    "cancel_purchase_order": {
        "schema": CancelPurchaseOrderInput,
        "is_mutating": True,
        "required_scope": "erp:po:cancel",
        "description": "Cancel an open purchase order due to shipping delays or alternate sourcing.",
    },
    "send_email": {
        "schema": SendEmailInput,
        "is_mutating": True,
        "required_scope": "mail:send",
        "description": "Send an email message to a company colleague or supervisor.",
    },
    "notify_production": {
        "schema": NotifyProductionInput,
        "is_mutating": True,
        "required_scope": "production:notify",
        "description": "Issue an operational notification to the production supervisor regarding a schedule or part update.",
    },
    "reallocate_lot_for_order": {
        "schema": ReallocateLotForOrderInput,
        "is_mutating": True,
        "required_scope": "erp:quality:write",
        "description": "Atomically release a hold lot and reallocate an available replacement lot for a production order.",
    },
    "schedule_event": {
        "schema": ScheduleEventInput,
        "is_mutating": True,
        "required_scope": "mail:send",
        "description": "Schedule a deferred event to materialize when the system clock reaches trigger_date.",
    },
}


# ---------------------------------------------------------------------------
# Tool Factory & Prompts
# ---------------------------------------------------------------------------

def get_all_tools(api: SQLiteCompanyAPI, user_id: str) -> List[StructuredTool]:
    """
    Construct LangChain StructuredTool instances bound to the given API instance and active user context.
    Every call checks user permissions directly against the SQLite database.
    """
    tools = []

    def make_read_emails(api_inst: SQLiteCompanyAPI, uid: str) -> Callable:
        def _read_emails(reason: str, recipient_id: Optional[str] = None, unread_only: bool = True) -> Any:
            target = recipient_id or uid
            with api_inst.user_context(uid):
                return api_inst.get_emails(recipient_id=target, unread_only=unread_only)
        return _read_emails

    def make_get_po(api_inst: SQLiteCompanyAPI, uid: str) -> Callable:
        def _get_po(po_id: str, reason: str) -> Any:
            with api_inst.user_context(uid):
                return api_inst.get_purchase_order(po_id=po_id)
        return _get_po

    def make_get_prod(api_inst: SQLiteCompanyAPI, uid: str) -> Callable:
        def _get_prod(reason: str, part_id: Optional[str] = None) -> Any:
            with api_inst.user_context(uid):
                return api_inst.get_production_orders(part_id=part_id)
        return _get_prod

    def make_query_sup(api_inst: SQLiteCompanyAPI, uid: str) -> Callable:
        def _query_sup(reason: str, part_id: Optional[str] = None, approved_only: bool = True) -> Any:
            with api_inst.user_context(uid):
                return api_inst.query_suppliers(part_id=part_id, approved_only=approved_only)
        return _query_sup

    def make_get_cal(api_inst: SQLiteCompanyAPI, uid: str) -> Callable:
        def _get_cal(reason: str, user_id: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Any:
            target = user_id or uid
            with api_inst.user_context(uid):
                return api_inst.get_calendar_events(user_id=target)
        return _get_cal

    def make_get_quality_lots(api_inst: SQLiteCompanyAPI, uid: str) -> Callable:
        def _get_quality_lots(reason: str, part_id: Optional[str] = None, status: Optional[str] = None, allocated_order_id: Optional[str] = None) -> Any:
            with api_inst.user_context(uid):
                return api_inst.get_quality_lots(part_id=part_id, status=status, allocated_order_id=allocated_order_id)
        return _get_quality_lots

    def make_get_quality_lot(api_inst: SQLiteCompanyAPI, uid: str) -> Callable:
        def _get_quality_lot(lot_id: str, reason: str) -> Any:
            with api_inst.user_context(uid):
                return api_inst.get_quality_lot(lot_id=lot_id)
        return _get_quality_lot

    def make_create_po(api_inst: SQLiteCompanyAPI, uid: str) -> Callable:
        def _create_po(
            po_id: str,
            part_id: str,
            supplier_id: str,
            quantity: int,
            unit_price: float,
            promised_date: str,
            idempotency_key: str,
            reason: str,
        ) -> Any:
            with api_inst.user_context(uid):
                api_inst.check_permission("erp:po:create", user_id=uid)
                target_po_id = po_id
                cursor = api_inst.conn.cursor()
                cursor.execute("SELECT 1 FROM PurchaseOrders WHERE po_id = ?;", (target_po_id,))
                if cursor.fetchone():
                    # Collision detected: find max numerical suffix or allocate next available
                    cursor.execute("SELECT po_id FROM PurchaseOrders WHERE po_id LIKE 'PO-%';")
                    rows = cursor.fetchall()
                    existing_nums = [77814]
                    for (r_id,) in rows:
                        parts = r_id.split("-")
                        if len(parts) == 2 and parts[1].isdigit():
                            existing_nums.append(int(parts[1]))
                    target_po_id = f"PO-{max(existing_nums) + 1}"

                return api_inst.create_po(
                    po_id=target_po_id,
                    part_id=part_id,
                    supplier_id=supplier_id,
                    quantity=quantity,
                    unit_price=unit_price,
                    promised_date=promised_date,
                )
        return _create_po

    def make_cancel_po(api_inst: SQLiteCompanyAPI, uid: str) -> Callable:
        def _cancel_po(po_id: str, reason: str, idempotency_key: str) -> Any:
            with api_inst.user_context(uid):
                return api_inst.cancel_po(po_id=po_id, reason=reason)
        return _cancel_po

    def make_send_email(api_inst: SQLiteCompanyAPI, uid: str) -> Callable:
        def _send_email(recipient_id: str, subject: str, body: str, idempotency_key: str, reason: str) -> Any:
            with api_inst.user_context(uid):
                return api_inst.send_email(recipient_id=recipient_id, subject=subject, body=body)
        return _send_email

    def make_notify_prod(api_inst: SQLiteCompanyAPI, uid: str) -> Callable:
        def _notify_prod(supervisor_id: str, order_id: str, message: str, idempotency_key: str, reason: str) -> Any:
            with api_inst.user_context(uid):
                return api_inst.notify_production(supervisor_id=supervisor_id, order_id=order_id, message=message)
        return _notify_prod

    def make_reallocate_lot_for_order(api_inst: SQLiteCompanyAPI, uid: str) -> Callable:
        def _reallocate_lot_for_order(order_id: str, from_lot_id: str, to_lot_id: str, idempotency_key: str, reason: str) -> Any:
            with api_inst.user_context(uid):
                return api_inst.reallocate_lot_for_order(order_id=order_id, from_lot_id=from_lot_id, to_lot_id=to_lot_id)
        return _reallocate_lot_for_order

    def make_schedule_event(api_inst: SQLiteCompanyAPI, uid: str) -> Callable:
        def _schedule_event(trigger_date: str, target_table: str, payload: Dict[str, Any], idempotency_key: str, reason: str) -> Any:
            with api_inst.user_context(uid):
                return api_inst.schedule_event(trigger_date=trigger_date, target_table=target_table, payload=payload)
        return _schedule_event

    factory_map = {
        "read_emails": make_read_emails,
        "get_purchase_order": make_get_po,
        "get_production_orders": make_get_prod,
        "query_suppliers": make_query_sup,
        "get_calendar_events": make_get_cal,
        "get_quality_lots": make_get_quality_lots,
        "get_quality_lot": make_get_quality_lot,
        "create_purchase_order": make_create_po,
        "cancel_purchase_order": make_cancel_po,
        "send_email": make_send_email,
        "notify_production": make_notify_prod,
        "reallocate_lot_for_order": make_reallocate_lot_for_order,
        "schedule_event": make_schedule_event,
    }

    for name, meta in TOOL_REGISTRY.items():
        fn = factory_map[name](api, user_id)
        tool = StructuredTool.from_function(
            func=fn,
            name=name,
            description=meta["description"],
            args_schema=meta["schema"],
        )
        # Attach custom metadata
        tool.metadata = {
            "is_mutating": meta["is_mutating"],
            "required_scope": meta["required_scope"],
        }
        tools.append(tool)

    return tools


def get_tools_for_user(api: SQLiteCompanyAPI, user_id: str) -> List[StructuredTool]:
    """
    Construct only the subset of StructuredTools that the specified user
    is authorized to invoke under their RBAC permissions.
    """
    user_info = api.get_user(user_id)
    user_scopes = set(user_info.get("scopes", []))
    all_tools = get_all_tools(api, user_id)
    return [t for t in all_tools if t.metadata.get("required_scope") in user_scopes]


def build_toolcards_prompt(allowed_tools: Optional[List[str]] = None) -> str:
    """
    Format tool definitions into an explicit toolcard guide for the agent's initial prompt.
    If allowed_tools is provided, filters to only those tools.
    """
    lines = [
        "## OPERATIONAL TOOLCARDS & REGISTRY",
        "You have access to the following operational tools. Each invocation MUST include a 'reason' explaining your intent.",
        "Mutating actions will be intercepted by the supervisor approval gate before executing in the ERP.",
        "",
    ]

    for name, meta in TOOL_REGISTRY.items():
        if allowed_tools is not None and name not in allowed_tools:
            continue
        category = "MUTATING ACTION [GATED]" if meta["is_mutating"] else "READ QUERY [IMMEDIATE]"
        lines.append(f"### `{name}` ({category})")
        lines.append(f"- **Description:** {meta['description']}")
        lines.append(f"- **Required RBAC Scope:** `{meta['required_scope']}`")
        schema_props = meta["schema"].model_json_schema().get("properties", {})
        param_list = [f"`{p}`" for p in schema_props.keys()]
        lines.append(f"- **Parameters:** {', '.join(param_list)}")
        lines.append("")

    return "\n".join(lines)

