import datetime
import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

from .exceptions import (
    ApprovalLimitExceededError,
    CompanyAPIError,
    EntityNotFoundError,
    UnauthorizedError,
    ValidationError,
)


class SQLiteCompanyAPI:
    """
    Access layer and business logic interface for the company environment,
    backed by a SQLite database with strict scope verification, approval
    limit enforcement, and transactional writes.
    """

    def __init__(
        self,
        conn: Optional[sqlite3.Connection] = None,
        database: str = ":memory:",
        current_user_id: Optional[str] = None,
    ):
        if conn is not None:
            self.conn = conn
        else:
            from .db import create_company_database
            self.conn = create_company_database(database=database, seed=True)

        self.conn.row_factory = sqlite3.Row
        self.current_user_id = current_user_id
        # Materialize any scheduled events matured up to current clock
        self.process_scheduled_events()

    def as_user(self, user_id: str) -> "SQLiteCompanyAPI":
        """Set the active user context for subsequent API calls and return self."""
        self.current_user_id = user_id
        return self

    @contextmanager
    def user_context(self, user_id: str):
        """Temporarily switch active user within a context block."""
        prev_user = self.current_user_id
        self.current_user_id = user_id
        try:
            yield self
        finally:
            self.current_user_id = prev_user

    # -------------------------------------------------------------------------
    # Permission and Scope Verification
    # -------------------------------------------------------------------------

    def _resolve_user(self, user_id: Optional[str]) -> str:
        uid = user_id or self.current_user_id
        if not uid:
            raise UnauthorizedError("anonymous", "authenticated_user", "No authenticated user provided for operation.")
        return uid

    def check_permission(self, required_scope: Union[str, List[str]], user_id: Optional[str] = None) -> str:
        """
        Verify that the user possesses at least one of the required scopes.
        Raises UnauthorizedError if not authorized.
        Returns the resolved user_id.
        """
        uid = self._resolve_user(user_id)
        scopes = [required_scope] if isinstance(required_scope, str) else required_scope

        placeholders = ",".join("?" for _ in scopes)
        query = f"SELECT 1 FROM UserScopes WHERE user_id = ? AND scope IN ({placeholders}) LIMIT 1;"
        cursor = self.conn.cursor()
        cursor.execute(query, [uid] + scopes)
        row = cursor.fetchone()

        if not row:
            needed = " OR ".join(scopes)
            raise UnauthorizedError(uid, needed)
        return uid

    # -------------------------------------------------------------------------
    # Clock / System State
    # -------------------------------------------------------------------------

    def get_clock(self) -> str:
        """Get the current mock date in YYYY-MM-DD format."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM SystemState WHERE key = 'current_date';")
        row = cursor.fetchone()
        return row["value"] if row else "2026-09-02"

    def set_clock(self, new_date: str) -> str:
        """Set the current mock date and materialize all scheduled events due by new_date."""
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO SystemState (key, value) VALUES ('current_date', ?);",
                (new_date,),
            )
        self.process_scheduled_events(new_date)
        return new_date

    def advance_clock(self, days: int = 1) -> str:
        """Advance the mock date by a given number of days."""
        current = self.get_clock()
        dt = datetime.datetime.strptime(current, "%Y-%m-%d").date()
        new_dt = dt + datetime.timedelta(days=days)
        new_date = new_dt.strftime("%Y-%m-%d")
        return self.set_clock(new_date)

    def process_scheduled_events(self, as_of_date: Optional[str] = None) -> int:
        """
        Scan ScheduledEvents staging table and materialize all records whose
        trigger_date has been reached into their respective tables.
        Returns the number of processed events.
        """
        cutoff = as_of_date or self.get_clock()
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT event_id, target_table, payload_json
            FROM ScheduledEvents
            WHERE date(trigger_date) <= date(?) AND processed = 0
            ORDER BY date(trigger_date) ASC, event_id ASC;
            """,
            (cutoff,),
        )
        events = cursor.fetchall()
        if not events:
            return 0

        with self.conn:
            for ev in events:
                event_id = ev["event_id"]
                table = ev["target_table"]
                payload = json.loads(ev["payload_json"])

                # Sanitize / normalize payload for known tables
                if table == "ProductionNotifications":
                    if "sent_by" not in payload:
                        payload["sent_by"] = "u-101"
                    if "created_at" not in payload:
                        payload["created_at"] = f"{cutoff}T12:00:00"
                    if payload.get("supervisor_id") in ("production-supervisor", "production", None):
                        payload["supervisor_id"] = "u-301"
                elif table == "Mail":
                    if "sent_at" not in payload:
                        payload["sent_at"] = f"{cutoff}T12:00:00"
                    if "sender" not in payload:
                        payload["sender"] = "u-101"

                cols = list(payload.keys())
                placeholders = ",".join("?" for _ in cols)
                col_names = ",".join(cols)
                sql = f"INSERT OR REPLACE INTO {table} ({col_names}) VALUES ({placeholders});"
                try:
                    self.conn.execute(sql, list(payload.values()))
                except Exception:
                    pass

                self.conn.execute(
                    "UPDATE ScheduledEvents SET processed = 1, processed_at = ? WHERE event_id = ?;",
                    (cutoff, event_id),
                )

        return len(events)

    def schedule_event(self, trigger_date: str, target_table: str, payload: Dict[str, Any]) -> int:
        """
        Stage a future event to be materialized when the clock reaches trigger_date.
        If trigger_date <= current clock, immediately materializes the event.
        """
        # Ensure required payload fields exist
        if target_table == "ProductionNotifications":
            if "sent_by" not in payload:
                payload["sent_by"] = self.current_user_id or "u-101"
            if "created_at" not in payload:
                payload["created_at"] = f"{self.get_clock()}T12:00:00"
            if payload.get("supervisor_id") in ("production-supervisor", "production", None):
                payload["supervisor_id"] = "u-301"
        elif target_table == "Mail":
            if "sent_at" not in payload:
                payload["sent_at"] = f"{self.get_clock()}T12:00:00"
            if "sender" not in payload:
                payload["sender"] = self.current_user_id or "u-101"

        payload_json = json.dumps(payload)
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO ScheduledEvents (trigger_date, target_table, payload_json, processed)
                VALUES (?, ?, ?, 0);
                """,
                (trigger_date, target_table, payload_json),
            )
            event_id = cursor.lastrowid

        if trigger_date <= self.get_clock():
            self.process_scheduled_events()

        return event_id

    # -------------------------------------------------------------------------
    # Users & Approvers
    # -------------------------------------------------------------------------

    def get_user(self, user_id: str) -> Dict[str, Any]:
        """Fetch user record including scopes and backup approver info."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM Users WHERE user_id = ?;", (user_id,))
        row = cursor.fetchone()
        if not row:
            raise EntityNotFoundError(f"User '{user_id}' not found.")

        user_dict = dict(row)
        cursor.execute("SELECT scope FROM UserScopes WHERE user_id = ?;", (user_id,))
        user_dict["scopes"] = [r["scope"] for r in cursor.fetchall()]
        return user_dict

    def is_user_out_of_office(self, user_id: str, check_date: Optional[str] = None) -> bool:
        """
        Check if a user is out of office on a given date (defaults to current clock).
        Requires 'calendar:read' scope.
        """
        self.check_permission("calendar:read")
        target_date = check_date or self.get_clock()
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT 1 FROM CalendarEvents
            WHERE owner_id = ?
              AND out_of_office = 1
              AND date(start_time) <= date(?)
              AND date(end_time) >= date(?);
            """,
            (user_id, target_date, target_date),
        )
        return cursor.fetchone() is not None

    def get_effective_approver(self, user_id: str, check_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Determine the effective approver. If the requested user is out of office,
        routes to their backup approver. Requires 'calendar:read' scope.
        """
        self.check_permission("calendar:read")
        user = self.get_user(user_id)
        if self.is_user_out_of_office(user_id, check_date):
            backup_id = user.get("backup_approver_id")
            if backup_id:
                backup_user = self.get_user(backup_id)
                backup_user["delegated_for"] = user_id
                backup_user["reason"] = f"Primary approver '{user_id}' is Out of Office."
                return backup_user
        return user

    # -------------------------------------------------------------------------
    # Calendar
    # -------------------------------------------------------------------------

    def get_calendar_events(
        self,
        user_id: Optional[str] = None,
        out_of_office_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Query calendar events. Requires 'calendar:read' scope."""
        self.check_permission("calendar:read")
        cursor = self.conn.cursor()

        query = "SELECT * FROM CalendarEvents WHERE 1=1"
        params: List[Any] = []

        if user_id:
            query += " AND owner_id = ?"
            params.append(user_id)

        if out_of_office_only:
            query += " AND out_of_office = 1"

        query += " ORDER BY start_time ASC;"
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # Mail (Inbox)
    # -------------------------------------------------------------------------

    def get_emails(
        self,
        recipient_id: Optional[str] = None,
        unread_only: bool = False,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve inbox emails. Requires 'mail:read' scope."""
        caller = self.check_permission("mail:read", user_id=user_id)
        target_recipient = recipient_id or caller

        cursor = self.conn.cursor()
        query = "SELECT * FROM Mail WHERE recipient_id = ?"
        params: List[Any] = [target_recipient]

        if unread_only:
            query += " AND read_status = 0"

        query += " ORDER BY sent_at DESC;"
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_email_by_id(self, mail_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Fetch a specific email and mark it as read. Requires 'mail:read'."""
        caller = self.check_permission("mail:read", user_id=user_id)
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM Mail WHERE mail_id = ?;", (mail_id,))
        row = cursor.fetchone()
        if not row:
            raise EntityNotFoundError(f"Email '{mail_id}' not found.")

        # Ensure caller is recipient or sender
        if row["recipient_id"] != caller and row["sender"] != caller:
            raise UnauthorizedError(caller, "mail:read", f"User '{caller}' cannot access email addressed to '{row['recipient_id']}'.")

        with self.conn:
            self.conn.execute("UPDATE Mail SET read_status = 1 WHERE mail_id = ?;", (mail_id,))

        updated_row = self.conn.execute("SELECT * FROM Mail WHERE mail_id = ?;", (mail_id,)).fetchone()
        return dict(updated_row)

    def send_email(
        self,
        recipient_id: str,
        subject: str,
        body: str,
        mail_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send an email. Requires 'mail:send' scope."""
        sender_id = self.check_permission("mail:send", user_id=user_id)
        new_mail_id = mail_id or f"M-{int(datetime.datetime.now().timestamp() * 1000)}"
        sent_at = f"{self.get_clock()}T12:00:00"

        cursor = self.conn.cursor()
        target_recipient_id = recipient_id
        cursor.execute("SELECT user_id FROM Users WHERE user_id = ?;", (recipient_id,))
        if not cursor.fetchone():
            cursor.execute(
                "SELECT user_id FROM Users WHERE name LIKE ? OR title LIKE ? LIMIT 1;",
                (f"%{recipient_id}%", f"%{recipient_id}%"),
            )
            row = cursor.fetchone()
            if row:
                target_recipient_id = row[0]
            else:
                cursor.execute("SELECT user_id FROM Users WHERE title LIKE '%Production%' LIMIT 1;")
                row = cursor.fetchone()
                if row:
                    target_recipient_id = row[0]

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO Mail (mail_id, sender, recipient_id, subject, body, sent_at, read_status)
                VALUES (?, ?, ?, ?, ?, ?, 0);
                """,
                (new_mail_id, sender_id, target_recipient_id, subject, body, sent_at),
            )

        return {
            "mail_id": new_mail_id,
            "sender": sender_id,
            "recipient_id": target_recipient_id,
            "subject": subject,
            "body": body,
            "sent_at": sent_at,
            "read_status": 0,
        }

    # -------------------------------------------------------------------------
    # ERP: Parts & Suppliers
    # -------------------------------------------------------------------------

    def get_parts(self, part_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Retrieve part details. Requires any ERP read scope."""
        self.check_permission(["erp:po:read", "erp:production:read", "erp:quality:read"])
        cursor = self.conn.cursor()

        if part_id:
            cursor.execute("SELECT * FROM Parts WHERE part_id = ?;", (part_id,))
            row = cursor.fetchone()
            if not row:
                raise EntityNotFoundError(f"Part '{part_id}' not found.")
            return dict(row)

        cursor.execute("SELECT * FROM Parts ORDER BY part_id;")
        return [dict(row) for row in cursor.fetchall()]

    def query_suppliers(
        self,
        part_id: Optional[str] = None,
        approved_only: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query suppliers, with optional filters for supported part and approval status.
        Requires 'erp:po:read' scope.
        """
        self.check_permission("erp:po:read")
        cursor = self.conn.cursor()
        params: List[Any] = []

        if part_id:
            query = """
                SELECT s.supplier_id, s.name, s.approved, s.lead_time_days, sap.unit_price
                FROM Suppliers s
                JOIN SupplierApprovedParts sap ON s.supplier_id = sap.supplier_id
                WHERE sap.part_id = ?
            """
            params.append(part_id)
            if approved_only is True:
                query += " AND s.approved = 1"
            elif approved_only is False:
                query += " AND s.approved = 0"
            query += " ORDER BY s.lead_time_days ASC, sap.unit_price ASC;"
        else:
            query = """
                SELECT DISTINCT s.supplier_id, s.name, s.approved, s.lead_time_days
                FROM Suppliers s
                WHERE 1=1
            """
            if approved_only is True:
                query += " AND s.approved = 1"
            elif approved_only is False:
                query += " AND s.approved = 0"
            query += " ORDER BY s.lead_time_days ASC;"

        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]

        # Attach list of approved parts and pricing for each supplier
        for sup in results:
            cursor.execute(
                "SELECT part_id, unit_price FROM SupplierApprovedParts WHERE supplier_id = ?;",
                (sup["supplier_id"],),
            )
            part_rows = cursor.fetchall()
            sup["approved_parts"] = [r["part_id"] for r in part_rows]
            sup["part_pricing"] = {r["part_id"]: r["unit_price"] for r in part_rows}

        return results

    # -------------------------------------------------------------------------
    # ERP: Purchase Orders
    # -------------------------------------------------------------------------

    def get_purchase_orders(
        self,
        status: Optional[str] = None,
        part_id: Optional[str] = None,
        supplier_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query purchase orders. Requires 'erp:po:read' scope."""
        self.check_permission("erp:po:read")
        cursor = self.conn.cursor()

        query = "SELECT * FROM PurchaseOrders WHERE 1=1"
        params: List[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status.upper())

        if part_id:
            query += " AND part_id = ?"
            params.append(part_id)

        if supplier_id:
            query += " AND supplier_id = ?"
            params.append(supplier_id)

        query += " ORDER BY promised_date ASC;"
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_purchase_order(self, po_id: str) -> Dict[str, Any]:
        """Fetch a specific purchase order by ID. Requires 'erp:po:read'."""
        self.check_permission("erp:po:read")
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM PurchaseOrders WHERE po_id = ?;", (po_id,))
        row = cursor.fetchone()
        if not row:
            raise EntityNotFoundError(f"Purchase order '{po_id}' not found.")
        return dict(row)

    def create_po(
        self,
        po_id: str,
        part_id: str,
        supplier_id: str,
        quantity: int,
        unit_price: float,
        promised_date: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a purchase order with approval limit validation.
        Requires 'erp:po:create' scope.
        """
        caller_id = self.check_permission("erp:po:create", user_id=user_id)
        user = self.get_user(caller_id)

        total_amount = float(quantity * unit_price)
        max_limit = float(user.get("po_create_max_value") or 0.0)

        # Check approval limit
        if total_amount > max_limit:
            backup_id = user.get("backup_approver_id")
            raise ApprovalLimitExceededError(
                user_id=caller_id,
                amount=total_amount,
                max_limit=max_limit,
                backup_approver_id=backup_id,
            )

        # Validate supplier and part existence
        cursor = self.conn.cursor()
        cursor.execute("SELECT approved FROM Suppliers WHERE supplier_id = ?;", (supplier_id,))
        sup = cursor.fetchone()
        if not sup:
            raise EntityNotFoundError(f"Supplier '{supplier_id}' does not exist.")

        cursor.execute("SELECT 1 FROM Parts WHERE part_id = ?;", (part_id,))
        if not cursor.fetchone():
            raise EntityNotFoundError(f"Part '{part_id}' does not exist.")

        created_at = f"{self.get_clock()}T10:00:00"

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO PurchaseOrders (
                    po_id, part_id, supplier_id, quantity, unit_price, total_amount,
                    status, promised_date, created_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?);
                """,
                (
                    po_id,
                    part_id,
                    supplier_id,
                    quantity,
                    unit_price,
                    total_amount,
                    promised_date,
                    caller_id,
                    created_at,
                ),
            )

        return self.get_purchase_order(po_id)

    def cancel_po(self, po_id: str, reason: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Cancel a purchase order. Requires 'erp:po:cancel' scope."""
        self.check_permission("erp:po:cancel", user_id=user_id)
        po = self.get_purchase_order(po_id)

        if po["status"] == "CANCELLED":
            return po

        with self.conn:
            self.conn.execute("UPDATE PurchaseOrders SET status = 'CANCELLED' WHERE po_id = ?;", (po_id,))

        return self.get_purchase_order(po_id)

    def reopen_po(self, po_id: str, reason: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Restore/reopen a cancelled purchase order back to OPEN. Requires 'erp:po:create' or 'erp:po:cancel'."""
        self.check_permission(["erp:po:create", "erp:po:cancel"], user_id=user_id)
        po = self.get_purchase_order(po_id)

        if po["status"] == "OPEN":
            return po

        with self.conn:
            self.conn.execute("UPDATE PurchaseOrders SET status = 'OPEN' WHERE po_id = ?;", (po_id,))

        return self.get_purchase_order(po_id)

    # -------------------------------------------------------------------------
    # ERP: Production Orders
    # -------------------------------------------------------------------------

    def get_production_orders(
        self,
        supervisor_id: Optional[str] = None,
        part_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query production orders. Requires 'erp:production:read' scope."""
        self.check_permission("erp:production:read")
        cursor = self.conn.cursor()

        query = "SELECT * FROM ProductionOrders WHERE 1=1"
        params: List[Any] = []

        if supervisor_id:
            query += " AND supervisor_id = ?"
            params.append(supervisor_id)

        if part_id:
            query += " AND part_id = ?"
            params.append(part_id)

        query += " ORDER BY scheduled_start ASC;"
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_production_order(self, order_id: str) -> Dict[str, Any]:
        """Fetch production order by ID. Requires 'erp:production:read'."""
        self.check_permission("erp:production:read")
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM ProductionOrders WHERE order_id = ?;", (order_id,))
        row = cursor.fetchone()
        if not row:
            raise EntityNotFoundError(f"Production order '{order_id}' not found.")
        return dict(row)

    def update_production_order(
        self,
        order_id: str,
        scheduled_start: Optional[str] = None,
        status: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update production order schedule or status. Requires 'erp:production:write' scope."""
        self.check_permission("erp:production:write", user_id=user_id)
        self.get_production_order(order_id)

        fields = []
        params = []
        if scheduled_start is not None:
            fields.append("scheduled_start = ?")
            params.append(scheduled_start)
        if status is not None:
            fields.append("status = ?")
            params.append(status)

        if not fields:
            return self.get_production_order(order_id)

        params.append(order_id)
        with self.conn:
            self.conn.execute(f"UPDATE ProductionOrders SET {', '.join(fields)} WHERE order_id = ?;", params)

        return self.get_production_order(order_id)

    # -------------------------------------------------------------------------
    # ERP: Quality Lots
    # -------------------------------------------------------------------------

    def get_quality_lots(
        self,
        part_id: Optional[str] = None,
        status: Optional[str] = None,
        allocated_order_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query quality lots. Requires 'erp:quality:read' scope."""
        self.check_permission("erp:quality:read")
        cursor = self.conn.cursor()

        query = "SELECT * FROM QualityLots WHERE 1=1"
        params: List[Any] = []

        if part_id:
            query += " AND part_id = ?"
            params.append(part_id)

        if status:
            query += " AND status = ?"
            params.append(status)

        if allocated_order_id is not None:
            if allocated_order_id == "":
                query += " AND allocated_order_id IS NULL"
            else:
                query += " AND allocated_order_id = ?"
                params.append(allocated_order_id)

        query += " ORDER BY lot_id ASC;"
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_quality_lot(self, lot_id: str) -> Dict[str, Any]:
        """Fetch quality lot by ID. Requires 'erp:quality:read'."""
        self.check_permission("erp:quality:read")
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM QualityLots WHERE lot_id = ?;", (lot_id,))
        row = cursor.fetchone()
        if not row:
            raise EntityNotFoundError(f"Quality lot '{lot_id}' not found.")
        return dict(row)

    def update_quality_lot(
        self,
        lot_id: str,
        status: Optional[str] = None,
        allocated_order_id: Optional[str] = None,
        hold_reason: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update quality lot status and allocations. Requires 'erp:quality:write'."""
        self.check_permission("erp:quality:write", user_id=user_id)
        self.get_quality_lot(lot_id)

        fields = []
        params = []
        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if allocated_order_id is not None:
            fields.append("allocated_order_id = ?")
            params.append(None if allocated_order_id == "" else allocated_order_id)
        if hold_reason is not None:
            fields.append("hold_reason = ?")
            params.append(hold_reason)

        if not fields:
            return self.get_quality_lot(lot_id)

        params.append(lot_id)
        with self.conn:
            self.conn.execute(f"UPDATE QualityLots SET {', '.join(fields)} WHERE lot_id = ?;", params)

        return self.get_quality_lot(lot_id)

    def reallocate_lot_for_order(
        self,
        order_id: str,
        from_lot_id: str,
        to_lot_id: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Atomically release one lot from an order and allocate an available lot to it.
        Requires 'erp:production:write' or 'erp:quality:write' scope.
        """
        self.check_permission(["erp:production:write", "erp:quality:write"], user_id=user_id)

        cursor = self.conn.cursor()
        # Verify order exists
        cursor.execute("SELECT * FROM ProductionOrders WHERE order_id = ?;", (order_id,))
        order = cursor.fetchone()
        if not order:
            raise EntityNotFoundError(f"Production order '{order_id}' not found.")

        # Verify old lot
        cursor.execute("SELECT * FROM QualityLots WHERE lot_id = ?;", (from_lot_id,))
        old_lot = cursor.fetchone()
        if not old_lot:
            raise EntityNotFoundError(f"Source lot '{from_lot_id}' not found.")

        # Verify new lot
        cursor.execute("SELECT * FROM QualityLots WHERE lot_id = ?;", (to_lot_id,))
        new_lot = cursor.fetchone()
        if not new_lot:
            raise EntityNotFoundError(f"Target lot '{to_lot_id}' not found.")

        if new_lot["status"] != "available":
            raise ValidationError(f"Target lot '{to_lot_id}' is not available (status: {new_lot['status']}).")

        if new_lot["part_id"] != order["part_id"]:
            raise ValidationError(
                f"Target lot part '{new_lot['part_id']}' does not match order part '{order['part_id']}'."
            )

        with self.conn:
            self.conn.execute(
                "UPDATE QualityLots SET allocated_order_id = NULL WHERE lot_id = ?;",
                (from_lot_id,),
            )
            self.conn.execute(
                "UPDATE QualityLots SET allocated_order_id = ? WHERE lot_id = ?;",
                (order_id, to_lot_id),
            )

        return {
            "order_id": order_id,
            "unallocated_lot": from_lot_id,
            "allocated_lot": to_lot_id,
            "status": "success",
        }

    # -------------------------------------------------------------------------
    # Production Notifications
    # -------------------------------------------------------------------------

    def notify_production(
        self,
        supervisor_id: str,
        message: str,
        order_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Dispatch a notification to production personnel. Requires 'production:notify' scope."""
        sender_id = self.check_permission("production:notify", user_id=user_id)
        created_at = f"{self.get_clock()}T12:00:00"

        cursor = self.conn.cursor()

        # Defensive order_id resolution (e.g. "Order 4812" -> "4812")
        target_order_id = order_id
        if order_id:
            cursor.execute("SELECT order_id, supervisor_id FROM ProductionOrders WHERE order_id = ?;", (str(order_id),))
            order_row = cursor.fetchone()
            if not order_row:
                import re
                match = re.search(r"\d+", str(order_id))
                if match:
                    cursor.execute("SELECT order_id, supervisor_id FROM ProductionOrders WHERE order_id = ?;", (match.group(0),))
                    order_row = cursor.fetchone()
                    if order_row:
                        target_order_id = order_row["order_id"]
            else:
                target_order_id = order_row["order_id"]

        # Defensive supervisor_id resolution (e.g. "production-supervisor" or "Sam Taylor" -> "u-301")
        target_supervisor_id = supervisor_id
        cursor.execute("SELECT user_id FROM Users WHERE user_id = ?;", (supervisor_id,))
        if not cursor.fetchone():
            resolved = None
            if target_order_id:
                cursor.execute("SELECT supervisor_id FROM ProductionOrders WHERE order_id = ?;", (target_order_id,))
                row = cursor.fetchone()
                if row:
                    resolved = row[0]
            if not resolved:
                cursor.execute(
                    "SELECT user_id FROM Users WHERE name LIKE ? OR title LIKE ? LIMIT 1;",
                    (f"%{supervisor_id}%", f"%{supervisor_id}%"),
                )
                row = cursor.fetchone()
                if row:
                    resolved = row[0]
            if not resolved:
                cursor.execute("SELECT user_id FROM Users WHERE title LIKE '%Production%' LIMIT 1;")
                row = cursor.fetchone()
                if row:
                    resolved = row[0]
            if resolved:
                target_supervisor_id = resolved

        with self.conn:
            cursor.execute(
                """
                INSERT INTO ProductionNotifications (supervisor_id, order_id, message, sent_by, created_at)
                VALUES (?, ?, ?, ?, ?);
                """,
                (target_supervisor_id, target_order_id, message, sender_id, created_at),
            )
            notification_id = cursor.lastrowid

        return {
            "notification_id": notification_id,
            "supervisor_id": target_supervisor_id,
            "order_id": target_order_id,
            "message": message,
            "sent_by": sender_id,
            "created_at": created_at,
        }
