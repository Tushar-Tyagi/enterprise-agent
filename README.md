# Enterprise AI Agent Harness

An extensible AI agent harness for realistic enterprise supply chain and operations scenarios, backed by SQLite, fine-grained RBAC permissions, transactional workflows, and LangGraph human-in-the-loop state machines.

---

## 1. Quickstart & Conda Setup

### Create and Activate Environment

```bash
# Create the environment from environment.yml (Python 3.12)
conda env create -f environment.yml

# Activate the conda environment
conda activate enterprise-agent
```

Alternatively, use `pip` with `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Environment Variables

Set your OpenRouter API key for LLM-driven planning:
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
# Optional: specify model override (defaults to google/gemini-2.5-flash fallback or anthropic/claude-3.5-sonnet)
export OPENROUTER_MODEL="google/gemini-2.5-flash"
```

---

## 2. Running Scenarios & Verification

### Run Scenario A CLI Harness (Interactive)
Runs the full end-to-end detection, planning, human gating, and execution pipeline:
```bash
python3 main.py
```

### Run Scenario A CLI Harness (Auto-Approve)
```bash
python3 main.py --auto-approve
```

### Run Scenario A Free-Form Agent Mode
Runs the autonomous tool-calling agent with centralized toolcards, model rationale, RBAC verification, mutating action gating, idempotency, and Saga rollback compensation:
```bash
# Interactive mode (prompts with model reason before each mutating action)
python3 main.py --mode freeform

# Auto-approve mode
python3 main.py --mode freeform --auto-approve
```

### Run Multi-Scenario Environment Demo
Exercises database traps, permission barriers, Scenario A (PO expedite), and Scenario B (quality lot reallocation):
```bash
python3 run_demo.py
```

### Run Test Suite
Executes all 24 unit and integration tests across data layers, trap detection, deterministic pipelines, and free-form agent lifecycles:
```bash
pytest -v
```

---

## 3. Architecture & Implemented Components

```
enterprise-agent/
├── environment/
│   ├── __init__.py           # Public exports
│   ├── schema.sql            # SQLite DDL schema with foreign keys and constraints
│   ├── db.py                 # DB initialization, seeds, and clock-driven staged events
│   ├── api.py                # SQLiteCompanyAPI access layer with RBAC and transactions
│   └── exceptions.py         # Custom exceptions (UnauthorizedError, ApprovalLimitExceededError, etc.)
├── detector.py               # Out-of-band operational detector for stockout and delay risks
├── toolcards.py              # Centralized toolcards, Pydantic schemas with mandatory reason, and RBAC tools
├── transaction.py            # Idempotency registry and Saga compensation rollback stack
├── agent.py                  # LangGraph deterministic pipeline (Scenario A)
├── agent_freeform.py         # LangGraph free-form cyclical agent with tool loop and human gating
├── main.py                   # Scenario A CLI harness with deterministic & free-form modes
├── run_demo.py               # Standalone database and scenario validation script
├── environment.yml           # Conda environment definition
├── requirements.txt          # Python pip dependencies
└── tests/
    ├── test_company_api.py   # 11 tests for RBAC, traps, clock gating, and DB operations
    ├── test_scenario_a_agent.py # End-to-end deterministic LangGraph lifecycle integration test
    ├── test_freeform_agent.py   # 4 tests for free-form lifecycle, gating, idempotency, and compensation
    ├── test_toolcards.py        # 4 tests for toolcards, schema validation, and RBAC enforcement
    ├── test_transaction.py      # 2 tests for idempotency caching and Saga LIFO unwinding
    └── test_main_cli.py         # CLI argument and mode selection test
```

### Data Layer (`environment/`)
- **`schema.sql`**: Relational tables for `SystemState` (clock), `Users`, `UserScopes`, `CalendarEvents`, `Mail`, `Parts`, `Suppliers`, `SupplierApprovedParts`, `PurchaseOrders`, `ProductionOrders`, `QualityLots`, `ProductionNotifications`, and `ScheduledEvents`.
- **`environment/db.py`**: Initializes and seeds baseline positive and negative/trap data:
  - **Users**: Dana Whitfield (`u-101`, limit $25,000, backup `u-102`), Alex Morgan (`u-102`, limit $100,000), Casey Chen (`u-201`, Quality Manager), Sam Taylor (`u-301`, Production Supervisor).
  - **Calendar**: `E-002` (Dana OOO 2026-09-03 to 2026-09-04) and trap `E-003` (Supplier Site Visit, `out_of_office = 0`).
  - **Suppliers**: `S-Y` (delayed, 5d lead time), `S-Z` (approved alternate, 2d lead time, $210/unit), `S-X` (trap: unapproved), `S-W` (trap: 14d lead time, too slow).
  - **Purchase Orders**: Delayed `PO-77812` for `P-4471`, trap closed `PO-77813`, trap over-limit `PO-77814`.
  - **Production Orders**: `4812` (starts 2026-09-07, blocked by delay), `4820` (starts 2026-09-05), trap future run `4899`.
  - **Quality Lots**: Hold lot `L-2093`, available spare `L-2094`, trap hold lot `L-2095`.
- **`environment/api.py` (`SQLiteCompanyAPI`)**:
  - Enforces user scopes before running any SQL read/write.
  - Dynamically calculates approval limits on PO creation, raising `ApprovalLimitExceededError` with designated backup approvers.
  - Manages mock clock and automatically materializes staged events from `ScheduledEvents` when the clock advances.
  - Implements transactional atomic lot reallocations and PO updates.

### The Detector (`detector.py`)
- Scans user inbox for unread shipping delay notices.
- Cross-references delayed PO parts against scheduled production start dates.
- Triggers on `M-001` (`PO-77812` delayed to 2026-09-08, missing Order `4812` start on 2026-09-07).
- Ignores irrelevant traps (`M-002` for inactive part `P-9999` and newsletter noise `M-003`).

### LangGraph Agent (`agent.py`)
- Employs `SqliteSaver` checkpointer to serialize state and pause for human authorization.
- **`gather_context`**: Pulls ERP state, calendars, and candidate suppliers.
- **`planner`**: Invokes OpenRouter LLM with `PlannerOutput` structured schema. Formulates the exact alert message to the purchasing manager:
  > *"Part P-4471 will likely cause production order 4812 to miss its scheduled start. Supplier Y said the shipment is delayed until Tuesday. I can move the PO to Supplier Z and notify production. Want me to proceed?"*
  Captures and logs exact token usage telemetry (`prompt_tokens`, `completion_tokens`, `total_tokens`, `cost`).
- **`gate`**: Evaluates the business rule:
  > *If an approval request is unanswered at end of day and the approver's calendar shows them out the next day, it routes to their designated backup.*
  Detects that Dana's calendar shows Out of Office tomorrow (`2026-09-03`), pauses graph execution (`approval_status = "pending"`), and escalates authorization authority to designated backup approver `u-102` (Alex Morgan).
- **`execute_plan`**: On approval, creates replacement `PO-77815`, cancels original `PO-77812`, dispatches alerts to Production Supervisor `u-301`, and stages arrival verification for next Tuesday (`2026-09-08`). Includes bidirectional Saga compensation: if any downstream action (such as alert dispatch or task staging) fails, the workflow rolls back in reverse order by restoring `PO-77812` back to `'OPEN'` and voiding `PO-77815` (`'CANCELLED'`).

### CLI Runner (`main.py`)
- Orchestrates detection, graph invocation, presents the exact manager alert, simulates the End-of-Day OOO rule escalation to backup approver Alex Morgan, prompts for authorization, executes actions, and advances the clock to next Tuesday (`2026-09-08`), proving arrival verification delivery.
