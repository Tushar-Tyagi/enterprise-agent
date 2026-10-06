"""
Quality Context Gathering & Persona Scoping Module for Scenario B.
Provides scoped operational context for Quality Manager Casey Chen (u-201),
including production order requirements, quality lots inspection, supervisor routing,
and RBAC-filtered tool availability.
"""

from typing import Any, Dict, List, Optional
from environment.api import SQLiteCompanyAPI
from toolcards import build_toolcards_prompt, get_tools_for_user


def gather_quality_context(
    api: SQLiteCompanyAPI,
    attention_item: Dict[str, Any],
    user_id: str = "u-201",
) -> Dict[str, Any]:
    """
    Gathers end-to-end operational context for Scenario B:
    - User identity & RBAC permissions (Casey Chen, Quality Manager).
    - Production order details & schedule constraint.
    - Lot inventory: currently allocated hold lot, candidate lots, trap lots.
    - Production supervisor directory routing.
    - RBAC-authorized toolcards prompt.
    """
    with api.user_context(user_id):
        # 1. Resolve active user persona & permissions
        user_info = api.get_user(user_id)
        user_scopes = set(user_info.get("scopes", []))
        clock_date = api.get_clock()

        order_id = attention_item.get("order_id") or attention_item.get("production_order_id", "4820")
        part_id = attention_item.get("part_id", "P-1180")
        hold_lot_id = attention_item.get("allocated_lot") or attention_item.get("lot_id")

        # 2. Query production order schedule and requirements
        prod_order = api.get_production_order(order_id)
        scheduled_start = prod_order.get("scheduled_start")
        supervisor_id = prod_order.get("supervisor_id", "u-301")
        supervisor_info = api.get_user(supervisor_id)

        # 3. Query all lots for the target part to evaluate coverage and expose traps
        all_lots = api.get_quality_lots(part_id=part_id)

        current_lot = None
        available_lots = []
        hold_lots = []

        for lot in all_lots:
            if lot.get("allocated_order_id") == order_id:
                current_lot = lot
            elif lot.get("status") == "available" and not lot.get("allocated_order_id"):
                available_lots.append(lot)
            elif lot.get("status") == "hold":
                hold_lots.append(lot)

        # Fallback if current lot not flagged with allocated_order_id directly
        if not current_lot and hold_lot_id:
            try:
                current_lot = api.get_quality_lot(hold_lot_id)
            except Exception:
                current_lot = {"lot_id": hold_lot_id, "status": "hold"}

        # 4. Formulate tool list and prompt filtered strictly to Casey's authorized scopes
        authorized_tools = get_tools_for_user(api, user_id)
        authorized_tool_names = [t.name for t in authorized_tools]
        toolcards_doc = build_toolcards_prompt(allowed_tools=authorized_tool_names)

        return {
            "user_id": user_id,
            "user_name": user_info.get("name", "Casey Chen"),
            "user_title": user_info.get("title", "Quality Manager"),
            "authorized_scopes": list(user_scopes),
            "clock_date": clock_date,
            "order_id": order_id,
            "part_id": part_id,
            "scheduled_start": scheduled_start,
            "order_quantity": prod_order.get("quantity"),
            "supervisor_id": supervisor_id,
            "supervisor_name": supervisor_info.get("name", "Sam Taylor"),
            "current_lot": current_lot,
            "available_lots": available_lots,
            "hold_lots": hold_lots,
            "has_covering_lot": len(available_lots) > 0,
            "recommended_lot": available_lots[0]["lot_id"] if available_lots else None,
            "authorized_tool_names": authorized_tool_names,
            "toolcards_doc": toolcards_doc,
        }
