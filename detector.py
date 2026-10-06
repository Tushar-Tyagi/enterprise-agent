import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from environment.api import SQLiteCompanyAPI


class Detector:
    """
    Monitors company operational channels (inbox, ERP signals) to detect
    critical attention items requiring agent intervention.
    Includes persistent and in-memory trigger deduplication so duplicate
    events are not emitted repeatedly.
    """

    def __init__(self, api: SQLiteCompanyAPI):
        self.api = api
        self._local_seen_triggers = set()

    def is_trigger_seen(self, trigger_id: str) -> bool:
        """Check if trigger has been recorded locally or in persistent database."""
        if trigger_id in self._local_seen_triggers:
            return True
        if hasattr(self.api, "is_trigger_processed") and self.api.is_trigger_processed(trigger_id):
            return True
        return False

    def mark_trigger_processed(
        self,
        trigger_id: str,
        trigger_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark trigger as processed in memory and persist in database."""
        self._local_seen_triggers.add(trigger_id)
        if hasattr(self.api, "record_processed_trigger"):
            self.api.record_processed_trigger(trigger_id, trigger_type, metadata)

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

    def scan_for_attention_items(self, user_id: str, dedupe: bool = True) -> Optional[Dict[str, Any]]:
        """
        Scan unread mail for delayed shipments and cross-reference against
        upcoming production order schedules.

        When dedupe=True (default), skips triggers that have already been detected
        or processed.

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
                    trigger_id = f"delayed_shipment:{email['mail_id']}:{po_id}:{prod['order_id']}"
                    if dedupe and self.is_trigger_seen(trigger_id):
                        continue

                    attention_item = {
                        "type": "delayed_shipment_impacting_production",
                        "severity": "CRITICAL",
                        "trigger_id": trigger_id,
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

                    if dedupe:
                        self.mark_trigger_processed(
                            trigger_id=trigger_id,
                            trigger_type="delayed_shipment_impacting_production",
                            metadata={
                                "mail_id": email["mail_id"],
                                "po_id": po_id,
                                "order_id": prod["order_id"],
                                "delayed_date": delayed_date,
                            },
                        )

                    return attention_item

        return None

    def scan_for_quality_hold_shortages(
        self,
        user_id: str = "u-201",
        horizon_days: int = 3,
        dedupe: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Scan ERP for lot-tracked parts placed on quality hold that are allocated
        to production orders scheduled to start within horizon_days (default: 3 days).

        Cross-references with unallocated available lots for the same part to
        determine whether an alternate good lot covers the shortage.
        """
        current_clock = self.api.get_clock()
        curr_dt = datetime.strptime(current_clock, "%Y-%m-%d").date()
        cutoff_dt = curr_dt + timedelta(days=horizon_days)

        with self.api.user_context(user_id):
            # Query lots on hold
            hold_lots = self.api.get_quality_lots(status="hold")

            for lot in hold_lots:
                allocated_order_id = lot.get("allocated_order_id")
                if not allocated_order_id:
                    continue

                try:
                    order = self.api.get_production_order(allocated_order_id)
                except Exception:
                    continue

                order_start_str = order["scheduled_start"]
                try:
                    order_start_dt = datetime.strptime(order_start_str, "%Y-%m-%d").date()
                except Exception:
                    continue

                # Check if order scheduled start is within the horizon window [curr_dt, cutoff_dt]
                if curr_dt <= order_start_dt <= cutoff_dt:
                    trigger_id = f"quality_hold:{lot['lot_id']}:{order['order_id']}"
                    if dedupe and self.is_trigger_seen(trigger_id):
                        continue

                    part_id = lot["part_id"]
                    # Find whether another good lot of the same part can cover it
                    available_lots = self.api.get_quality_lots(
                        part_id=part_id,
                        status="available",
                        allocated_order_id="",  # unallocated
                    )
                    alternate_lot = available_lots[0] if available_lots else None
                    days_until_start = (order_start_dt - curr_dt).days

                    item = {
                        "type": "quality_hold_impacting_production",
                        "severity": "CRITICAL",
                        "trigger_id": trigger_id,
                        "lot_id": lot["lot_id"],
                        "part_id": part_id,
                        "hold_reason": lot.get("hold_reason", "Quality hold placed"),
                        "production_order_id": order["order_id"],
                        "production_scheduled_start": order_start_str,
                        "days_until_start": days_until_start,
                        "supervisor_id": order["supervisor_id"],
                        "covered_by_alternate": alternate_lot is not None,
                        "alternate_lot_id": alternate_lot["lot_id"] if alternate_lot else None,
                        "description": (
                            f"Quality hold alert: Lot {lot['lot_id']} for part {part_id} is on hold "
                            f"('{lot.get('hold_reason')}'). Production Order {order['order_id']} consumes this lot "
                            f"on {order_start_str} (in {days_until_start} days). "
                            + (
                                f"Alternate lot {alternate_lot['lot_id']} is available for reallocation."
                                if alternate_lot
                                else "No available lot covers this part; shortage must be flagged to purchasing."
                            )
                        ),
                    }

                    if dedupe:
                        self.mark_trigger_processed(
                            trigger_id=trigger_id,
                            trigger_type="quality_hold_impacting_production",
                            metadata={
                                "lot_id": lot["lot_id"],
                                "order_id": order["order_id"],
                                "part_id": part_id,
                                "alternate_lot_id": alternate_lot["lot_id"] if alternate_lot else None,
                            },
                        )

                    return item

        return None

    def scan_all_attention_items(
        self,
        dedupe: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Multi-channel daily scan across all operational domains:
        - Purchasing inbox signals (PO delay / supplier stockout)
        - ERP Quality holds impacting scheduled production orders
        Returns an ordered list of active attention items requiring intervention today.
        """
        items: List[Dict[str, Any]] = []

        # 1. Purchasing Domain
        item_po = self.scan_for_attention_items(user_id="u-101", dedupe=dedupe)
        if item_po:
            items.append(item_po)

        # 2. Quality Domain
        item_quality = self.scan_for_quality_hold_shortages(user_id="u-201", horizon_days=5, dedupe=dedupe)
        if item_quality:
            items.append(item_quality)

        return items
