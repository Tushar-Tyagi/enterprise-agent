-- Enterprise Scenario Schema for SQLite

PRAGMA foreign_keys = ON;

-- Global System State (Mock Clock, etc.)
CREATE TABLE IF NOT EXISTS SystemState (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Users & Organizational Hierarchy
CREATE TABLE IF NOT EXISTS Users (
    user_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    title TEXT NOT NULL,
    po_create_max_value REAL DEFAULT 0.0,
    backup_approver_id TEXT REFERENCES Users(user_id)
);

-- User Scopes / Permissions Junction Table
CREATE TABLE IF NOT EXISTS UserScopes (
    user_id TEXT NOT NULL REFERENCES Users(user_id) ON DELETE CASCADE,
    scope TEXT NOT NULL,
    PRIMARY KEY (user_id, scope)
);

-- Calendar Events
CREATE TABLE IF NOT EXISTS CalendarEvents (
    event_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL REFERENCES Users(user_id),
    title TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    out_of_office INTEGER NOT NULL DEFAULT 0 CHECK (out_of_office IN (0, 1))
);

-- Mail Messages
CREATE TABLE IF NOT EXISTS Mail (
    mail_id TEXT PRIMARY KEY,
    sender TEXT NOT NULL,
    recipient_id TEXT NOT NULL REFERENCES Users(user_id),
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    sent_at TEXT NOT NULL,
    read_status INTEGER NOT NULL DEFAULT 0 CHECK (read_status IN (0, 1))
);

-- ERP: Parts Catalog
CREATE TABLE IF NOT EXISTS Parts (
    part_id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    lot_tracked INTEGER NOT NULL DEFAULT 0 CHECK (lot_tracked IN (0, 1))
);

-- ERP: Suppliers
CREATE TABLE IF NOT EXISTS Suppliers (
    supplier_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    approved INTEGER NOT NULL DEFAULT 0 CHECK (approved IN (0, 1)),
    lead_time_days INTEGER NOT NULL
);

-- Supplier Approved Parts Junction Table
CREATE TABLE IF NOT EXISTS SupplierApprovedParts (
    supplier_id TEXT NOT NULL REFERENCES Suppliers(supplier_id) ON DELETE CASCADE,
    part_id TEXT NOT NULL REFERENCES Parts(part_id) ON DELETE CASCADE,
    unit_price REAL NOT NULL,
    PRIMARY KEY (supplier_id, part_id)
);

-- ERP: Purchase Orders
CREATE TABLE IF NOT EXISTS PurchaseOrders (
    po_id TEXT PRIMARY KEY,
    part_id TEXT NOT NULL REFERENCES Parts(part_id),
    supplier_id TEXT NOT NULL REFERENCES Suppliers(supplier_id),
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    total_amount REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('OPEN', 'CLOSED', 'CANCELLED')),
    promised_date TEXT NOT NULL,
    created_by TEXT NOT NULL REFERENCES Users(user_id),
    created_at TEXT NOT NULL
);

-- ERP: Production Orders
CREATE TABLE IF NOT EXISTS ProductionOrders (
    order_id TEXT PRIMARY KEY,
    part_id TEXT NOT NULL REFERENCES Parts(part_id),
    quantity INTEGER NOT NULL,
    scheduled_start TEXT NOT NULL,
    supervisor_id TEXT NOT NULL REFERENCES Users(user_id),
    status TEXT NOT NULL DEFAULT 'SCHEDULED'
);

-- ERP: Quality Lots
CREATE TABLE IF NOT EXISTS QualityLots (
    lot_id TEXT PRIMARY KEY,
    part_id TEXT NOT NULL REFERENCES Parts(part_id),
    status TEXT NOT NULL CHECK (status IN ('hold', 'available', 'quarantined', 'released')),
    allocated_order_id TEXT REFERENCES ProductionOrders(order_id),
    hold_reason TEXT,
    inspected_at TEXT
);

-- Production Notifications Log
CREATE TABLE IF NOT EXISTS ProductionNotifications (
    notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
    supervisor_id TEXT NOT NULL REFERENCES Users(user_id),
    order_id TEXT REFERENCES ProductionOrders(order_id),
    message TEXT NOT NULL,
    sent_by TEXT NOT NULL REFERENCES Users(user_id),
    created_at TEXT NOT NULL
);

-- Staging Queue for Clock-Driven Scenario Events
CREATE TABLE IF NOT EXISTS ScheduledEvents (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger_date TEXT NOT NULL,
    target_table TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    processed INTEGER NOT NULL DEFAULT 0 CHECK (processed IN (0, 1)),
    processed_at TEXT
);

-- Append-Only Audit Log of Engine Steps, Inputs, Decisions, Approvals, Outcomes
CREATE TABLE IF NOT EXISTS AuditLogs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    category TEXT NOT NULL CHECK (category IN ('INPUT', 'DECISION', 'APPROVAL', 'EXECUTION', 'COMPENSATION')),
    actor_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    details_json TEXT NOT NULL
);

-- Processed Triggers / Out-of-band Detector Deduplication
CREATE TABLE IF NOT EXISTS ProcessedTriggers (
    trigger_id TEXT PRIMARY KEY,
    trigger_type TEXT NOT NULL,
    detected_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);


