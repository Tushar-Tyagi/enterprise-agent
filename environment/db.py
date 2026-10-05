import datetime
import json
import os
import sqlite3
from pathlib import Path
from typing import Optional, Union

SCHEMA_FILE = Path(__file__).parent / "schema.sql"


def init_database(conn: sqlite3.Connection) -> None:
    """Initialize tables and views from schema.sql."""
    with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)


def seed_database(conn: sqlite3.Connection) -> None:
    """Seed the database with initial happy-path and trap/negative scenario data."""
    with conn:
        cursor = conn.cursor()

        # 1. System State (Clock initialized to 2026-09-02)
        cursor.execute("INSERT OR REPLACE INTO SystemState (key, value) VALUES ('current_date', '2026-09-02');")

        # 2. Users (insert u-102 before u-101 to satisfy foreign key constraint)
        users = [
            ("u-102", "Alex Morgan", "Director/Backup", 100000.0, None),
            ("u-101", "Dana Whitfield", "Purchasing Manager", 25000.0, "u-102"),
            ("u-201", "Casey Chen", "Quality Manager", 0.0, None),
            ("u-301", "Sam Taylor", "Production Supervisor", 0.0, None),
        ]
        cursor.executemany(
            """
            INSERT OR REPLACE INTO Users (user_id, name, title, po_create_max_value, backup_approver_id)
            VALUES (?, ?, ?, ?, ?);
            """,
            users,
        )

        # 3. User Scopes
        user_scopes = [
            # u-101
            ("u-101", "erp:po:read"),
            ("u-101", "erp:po:create"),
            ("u-101", "erp:po:cancel"),
            ("u-101", "erp:production:read"),
            ("u-101", "mail:read"),
            ("u-101", "mail:send"),
            ("u-101", "calendar:read"),
            ("u-101", "production:notify"),
            # u-102 (same scopes as u-101)
            ("u-102", "erp:po:read"),
            ("u-102", "erp:po:create"),
            ("u-102", "erp:po:cancel"),
            ("u-102", "erp:production:read"),
            ("u-102", "mail:read"),
            ("u-102", "mail:send"),
            ("u-102", "calendar:read"),
            ("u-102", "production:notify"),
            # u-201
            ("u-201", "erp:quality:read"),
            ("u-201", "erp:quality:write"),
            ("u-201", "erp:production:read"),
            ("u-201", "erp:po:read"),
            ("u-201", "mail:read"),
            ("u-201", "mail:send"),
            ("u-201", "calendar:read"),
            ("u-201", "production:notify"),
            # u-301
            ("u-301", "erp:production:read"),
            ("u-301", "erp:production:write"),
            ("u-301", "mail:read"),
            ("u-301", "calendar:read"),
        ]
        cursor.executemany(
            "INSERT OR REPLACE INTO UserScopes (user_id, scope) VALUES (?, ?);",
            user_scopes,
        )

        # 4. Calendar Events
        events = [
            ("E-002", "u-101", "Out of Office / PTO", "2026-09-03T00:00:00", "2026-09-04T23:59:59", 1),
            ("E-003", "u-101", "Supplier Site Visit", "2026-09-05T00:00:00", "2026-09-05T23:59:59", 0),  # Negative Trap: Not OOO
        ]
        cursor.executemany(
            """
            INSERT OR REPLACE INTO CalendarEvents (event_id, owner_id, title, start_time, end_time, out_of_office)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            events,
        )

        # 5. Mail
        mails = [
            (
                "M-001",
                "Supplier Y",
                "u-101",
                "PO-77812 shipment update",
                "Shipment is delayed until Tuesday 9/8.",
                "2026-09-02T08:30:00",
                0,
            ),
            (
                "M-002",
                "Supplier W",
                "u-101",
                "PO-77813 delay notice",
                "Delay notification for a part that has no upcoming production orders.",
                "2026-09-02T09:15:00",
                0,
            ),
            (
                "M-003",
                "Industrial Supply Weekly",
                "u-101",
                "Global Logistics Trends & Insights",
                "A promotional newsletter covering supply chain technology trends.",
                "2026-09-02T06:00:00",
                0,
            ),
        ]
        cursor.executemany(
            """
            INSERT OR REPLACE INTO Mail (mail_id, sender, recipient_id, subject, body, sent_at, read_status)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            mails,
        )

        # 6. Parts
        parts = [
            ("P-4471", "Turbine Blade Assembly (Scenario A)", 0),
            ("P-1180", "Titanium Fastener Flange (Scenario B)", 1),
            ("P-9999", "Unrelated Legacy Gasket", 0),
        ]
        cursor.executemany(
            "INSERT OR REPLACE INTO Parts (part_id, description, lot_tracked) VALUES (?, ?, ?);",
            parts,
        )

        # 7. Suppliers
        suppliers = [
            ("S-Y", "Supplier Y", 1, 5),
            ("S-Z", "Supplier Z (Valid Alternate)", 1, 2),
            ("S-X", "Supplier X (The Trap)", 0, 1),  # Trap: unapproved
            ("S-W", "Supplier W (Slow)", 1, 14),     # Trap: too slow
        ]
        cursor.executemany(
            """
            INSERT OR REPLACE INTO Suppliers (supplier_id, name, approved, lead_time_days)
            VALUES (?, ?, ?, ?);
            """,
            suppliers,
        )

        # 8. Supplier Approved Parts (with part-specific unit_price)
        supplier_parts = [
            ("S-Y", "P-4471", 200.0),
            ("S-Z", "P-4471", 210.0),
            ("S-X", "P-4471", 150.0),  # Note: supplier itself has approved=0
            ("S-W", "P-4471", 195.0),
            ("S-W", "P-9999", 50.0),
        ]
        cursor.executemany(
            "INSERT OR REPLACE INTO SupplierApprovedParts (supplier_id, part_id, unit_price) VALUES (?, ?, ?);",
            supplier_parts,
        )

        # 9. Purchase Orders
        pos = [
            ("PO-77812", "P-4471", "S-Y", 50, 200.0, 10000.0, "OPEN", "2026-09-04", "u-101", "2026-08-20T10:00:00"),
            ("PO-77813", "P-9999", "S-W", 20, 50.0, 1000.0, "CLOSED", "2026-08-15", "u-101", "2026-08-01T09:00:00"),
            ("PO-77814", "P-4471", "S-Z", 150, 200.0, 30000.0, "OPEN", "2026-09-05", "u-102", "2026-09-01T14:00:00"),
        ]
        cursor.executemany(
            """
            INSERT OR REPLACE INTO PurchaseOrders (
                po_id, part_id, supplier_id, quantity, unit_price, total_amount, status, promised_date, created_by, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            pos,
        )

        # 10. Production Orders
        production_orders = [
            ("4812", "P-4471", 50, "2026-09-07", "u-301", "SCHEDULED"),
            ("4820", "P-1180", 25, "2026-09-05", "u-301", "SCHEDULED"),
            ("4899", "P-4471", 100, "2026-12-01", "u-301", "SCHEDULED"),  # Trap: far future
        ]
        cursor.executemany(
            """
            INSERT OR REPLACE INTO ProductionOrders (order_id, part_id, quantity, scheduled_start, supervisor_id, status)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            production_orders,
        )

        # 11. Quality Lots
        lots = [
            ("L-2093", "P-1180", "hold", "4820", "Surface finish 3.4 Ra vs spec 3.2 Ra", "2026-09-01T11:00:00"),
            ("L-2094", "P-1180", "available", None, None, "2026-09-01T15:30:00"),
            ("L-2095", "P-1180", "hold", None, "Micro-crack detected during dye penetrant inspection", "2026-09-02T09:00:00"),
        ]
        cursor.executemany(
            """
            INSERT OR REPLACE INTO QualityLots (lot_id, part_id, status, allocated_order_id, hold_reason, inspected_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            lots,
        )

        # 12. Scheduled Events (Future scenario events staged until clock advances)
        scheduled_events = [
            (
                "2026-09-03",
                "Mail",
                json.dumps({
                    "mail_id": "M-004",
                    "sender": "Supplier Z",
                    "recipient_id": "u-101",
                    "subject": "Expedited Delivery Confirmation",
                    "body": "Confirmed: 50 units of P-4471 scheduled for dispatch.",
                    "sent_at": "2026-09-03T09:00:00",
                    "read_status": 0,
                }),
                0,
            ),
            (
                "2026-09-04",
                "Mail",
                json.dumps({
                    "mail_id": "M-005",
                    "sender": "Quality Lab",
                    "recipient_id": "u-201",
                    "subject": "Lab Results for Lot L-2093",
                    "body": "Rework analysis complete for surface finish variance.",
                    "sent_at": "2026-09-04T14:00:00",
                    "read_status": 0,
                }),
                0,
            ),
        ]
        cursor.executemany(
            """
            INSERT OR REPLACE INTO ScheduledEvents (trigger_date, target_table, payload_json, processed)
            VALUES (?, ?, ?, ?);
            """,
            scheduled_events,
        )


def create_company_database(
    database: Union[str, Path] = ":memory:",
    seed: bool = True,
    check_same_thread: bool = False,
) -> sqlite3.Connection:
    """Create a configured SQLite connection, initializing tables and seeding if requested."""
    conn = sqlite3.connect(database, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    init_database(conn)
    if seed:
        seed_database(conn)
    return conn
