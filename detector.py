import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from environment.api import SQLiteCompanyAPI


class Detector:
    """
    Monitors company operational channels (inbox, ERP signals) to detect
    critical attention items requiring agent intervention.
    """

    def __init__(self, api: SQLiteCompanyAPI):
        self.api = api

    def _extract_po_id(self, text: str) -> Optional[str]:
        """Extract PO identifier pattern (e.g., PO-77812)."""
        match = re.search(r"\b(PO-\d+)\b", text, re.IGNORECASE)
        return match.group(1).upper() if match else None

    def _parse_delayed_date(self, text: str, reference_year: int = 2026) -> Optional[str]:
        """
        Parse delayed date references from message text.
        Supports ISO formats (YYYY-MM-DD) as well as phrases like 'Tuesday 9/8' or '9/8'.
        """
        # Match ISO format YYYY-MM-DD
        iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        if iso_match:
            return iso_match.group(1)

        # Match MM/DD or M/D (e.g., 9/8, 09/08)
        slash_match = re.search(r"\b(\d{1,2})/(\d{1,2})\b", text)
        if slash_match:
            month = int(slash_match.group(1))
            day = int(slash_match.group(2))
            return f"{reference_year:04d}-{month:02d}-{day:02d}"

        return None

    def scan_for_attention_items(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Scan unread mail for delayed shipments and cross-reference against
        upcoming production order schedules.

        Returns a structured attention item dictionary if a production order
        is blocked by a shipment delay; otherwise returns None.
        """
        current_clock = self.api.get_clock()
        ref_year = datetime.strptime(current_clock, "%Y-%m-%d").year

        # 1. Fetch unread emails for user
        unread_emails = self.api.get_emails(recipient_id=user_id, unread_only=True)

        for email in unread_emails:
            full_text = f"{email['subject']} {email['body']}"

            # Check if this email indicates a shipment delay
            if not re.search(r"\b(delay|delayed|slips|postponed)\b", full_text, re.IGNORECASE):
                continue

            po_id = self._extract_po_id(full_text)
            if not po_id:
                continue

            try:
                po = self.api.get_purchase_order(po_id)
            except Exception:
                continue

            # Only examine OPEN purchase orders
            if po.get("status") != "OPEN":
                continue

            delayed_date = self._parse_delayed_date(full_text, reference_year=ref_year)
            if not delayed_date:
                continue

            part_id = po["part_id"]

            # 2. Cross-reference with upcoming production orders consuming this part
            prod_orders = self.api.get_production_orders(part_id=part_id)

            for prod in prod_orders:
                prod_start = prod["scheduled_start"]

                # If delayed delivery slips past the scheduled production start, we have a stockout
                if delayed_date > prod_start:
                    return {
                        "type": "delayed_shipment_impacting_production",
                        "severity": "CRITICAL",
                        "mail_id": email["mail_id"],
                        "po_id": po_id,
                        "part_id": part_id,
                        "supplier_id": po["supplier_id"],
                        "quantity": po["quantity"],
                        "unit_price": po["unit_price"],
                        "original_promised_date": po["promised_date"],
                        "delayed_promised_date": delayed_date,
                        "production_order_id": prod["order_id"],
                        "production_scheduled_start": prod_start,
                        "supervisor_id": prod["supervisor_id"],
                        "description": (
                            f"Critical stockout risk: PO {po_id} for part {part_id} "
                            f"is delayed until {delayed_date}, missing scheduled start "
                            f"for Production Order {prod['order_id']} on {prod_start}."
                        ),
                    }

        return None
