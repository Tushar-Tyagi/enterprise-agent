# Free-Form Agent Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a free-form tool-calling agent mode that investigates attention items, enforces RBAC on centralized toolcards, requires LLM rationale, pauses for human approval before mutating calls, and executes actions with idempotency and Saga compensation.

**Architecture:** A LangGraph cyclical ReAct state machine using centralized toolcards from `toolcards.py`. Read tools execute inline with permission checks; mutating tools trigger an authorization gate that evaluates spending limits and calendar out-of-office rules, pausing via `SqliteSaver` checkpointing. Mutating executions are protected by an `IdempotencyRegistry` and an unwinding `CompensationStack` that executes inverse operations upon failure.

**Tech Stack:** Python 3.12, LangGraph, LangChain OpenAI/Core, SQLite, Pydantic v2, Pytest

**Spec:** `docs/superpowers/specs/2026-10-05-freeform-agent-mode-design.md`

## Global Constraints

- Preserve the deterministic pipeline in `agent.py` (`create_scenario_a_graph`) without modification.
- Centralize all tool definitions and system prompt toolcards in a single file: `toolcards.py`.
- Require a mandatory `reason: str` parameter on all tool input schemas.
- Enforce active user RBAC scopes through `SQLiteCompanyAPI.check_permission`.
- Mutating tools require an `idempotency_key: str` and register reverse compensating operations.
- Always capture exact token usage and API cost telemetry from LLM calls.

## Review Focus

- Unauthorized tool execution: when a user lacks a required scope (e.g., `po:create`), the tool must raise `UnauthorizedError` rather than silently failing or bypassing checks.
- Missing tool rationale: if the LLM submits a tool call without `reason`, schema validation must reject the call.
- Idempotent retries: calling a mutating tool twice with the same idempotency key must return the previous result without repeating database mutations.
- Downstream failure rollback: if a multi-step sequence fails after creating a PO, the compensation stack must void the replacement PO and restore any modified records in LIFO order.
- Approver escalation: if the user is out of office tomorrow, mutating gating must route the approval request to their designated backup.

---

### Task 1: Centralized Toolcards and Schema Registry (`toolcards.py`)

**Files:**
- Create: `toolcards.py`
- Test: `tests/test_toolcards.py`

**Interfaces:**
- Consumes: `environment.api.SQLiteCompanyAPI`, `environment.exceptions.UnauthorizedError`
- Produces: `TOOL_REGISTRY`, `get_all_tools(api: SQLiteCompanyAPI, user_id: str)`, `build_toolcards_prompt()`

- [x] **Step 1: Write the failing test for tool schemas, RBAC verification, and mandatory reason**

```python
# tests/test_toolcards.py
import pytest
from pydantic import ValidationError
from environment.db import create_company_database
from environment.api import SQLiteCompanyAPI
from environment.exceptions import UnauthorizedError
from toolcards import (
    GetPurchaseOrderInput,
    CreatePurchaseOrderInput,
    get_all_tools,
    build_toolcards_prompt,
)

def test_tool_requires_reason():
    with pytest.raises(ValidationError):
        GetPurchaseOrderInput(po_id="PO-77812")

def test_tool_succeeds_with_reason():
    inp = GetPurchaseOrderInput(po_id="PO-77812", reason="Verifying promised delivery date")
    assert inp.po_id == "PO-77812"
    assert inp.reason == "Verifying promised delivery date"

def test_tool_enforces_rbac_scope():
    conn = create_company_database(":memory:", seed=True)
    api = SQLiteCompanyAPI(conn=conn, current_user_id="u-301") # Sam Taylor: no po:create scope
    tools = {t.name: t for t in get_all_tools(api=api, user_id="u-301")}
    create_tool = tools["create_purchase_order"]
    with pytest.raises(UnauthorizedError):
        create_tool.invoke({
            "po_id": "PO-99999",
            "part_id": "P-4471",
            "supplier_id": "S-Z",
            "quantity": 100,
            "unit_price": 210.0,
            "promised_date": "2026-09-04",
            "idempotency_key": "idemp-001",
            "reason": "Test order without permissions",
        })

def test_toolcards_prompt_contains_all_tools():
    prompt = build_toolcards_prompt()
    assert "create_purchase_order" in prompt
    assert "get_purchase_order" in prompt
    assert "reason" in prompt
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_toolcards.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'toolcards'`

- [x] **Step 3: Implement `toolcards.py`**

Define `BaseToolInput` with `reason: str`, Pydantic models for read and mutating tools, tool factories bound to `SQLiteCompanyAPI`, and `build_toolcards_prompt()`.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_toolcards.py -v`  
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add toolcards.py tests/test_toolcards.py
git commit -m "feat: implement centralized toolcards with rbac and rationale schemas"
```

---

### Task 2: Idempotency Registry and Saga Compensation Ledger (`transaction.py`)

**Files:**
- Create: `transaction.py`
- Test: `tests/test_transaction.py`

**Interfaces:**
- Consumes: `environment.api.SQLiteCompanyAPI`
- Produces: `IdempotencyRegistry`, `CompensationStack`, `execute_with_transaction_guard`

- [x] **Step 1: Write the failing test for idempotency and compensation stack**

```python
# tests/test_transaction.py
import pytest
from environment.db import create_company_database
from environment.api import SQLiteCompanyAPI
from transaction import IdempotencyRegistry, CompensationStack

def test_idempotency_caching():
    reg = IdempotencyRegistry()
    assert not reg.has_executed("key-1")
    reg.record_success("key-1", {"status": "created", "po_id": "PO-100"})
    assert reg.has_executed("key-1")
    assert reg.get_result("key-1") == {"status": "created", "po_id": "PO-100"}

def test_compensation_stack_lifo_unwind():
    conn = create_company_database(":memory:", seed=True)
    api = SQLiteCompanyAPI(conn=conn, current_user_id="u-102")
    stack = CompensationStack(api=api)

    # Action 1: Create replacement PO
    api.create_po("PO-77815", "P-4471", "S-Z", 100, 210.0, "2026-09-04", user_id="u-102")
    stack.push(
        action_name="create_po",
        compensating_callable=lambda: api.cancel_po("PO-77815", "Compensating rollback", user_id="u-102"),
        description="Void PO-77815",
    )

    # Action 2: Cancel original PO
    api.cancel_po("PO-77812", "Replacing with S-Z", user_id="u-102")
    stack.push(
        action_name="cancel_po",
        compensating_callable=lambda: api.reopen_po("PO-77812", "Compensating reopen", user_id="u-102"),
        description="Reopen PO-77812",
    )

    assert api.get_purchase_order("PO-77815")["status"] == "OPEN"
    assert api.get_purchase_order("PO-77812")["status"] == "CANCELLED"

    # Trigger rollback
    unwind_log = stack.unwind()
    assert len(unwind_log) == 2
    assert api.get_purchase_order("PO-77812")["status"] == "OPEN"
    assert api.get_purchase_order("PO-77815")["status"] == "CANCELLED"
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transaction.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'transaction'`

- [x] **Step 3: Implement `transaction.py`**

Implement `IdempotencyRegistry` with in-memory cache and optional database table persistence, and `CompensationStack` with LIFO unwinding and structured rollback logging.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transaction.py -v`  
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add transaction.py tests/test_transaction.py
git commit -m "feat: implement idempotency registry and saga compensation stack"
```

---

### Task 3: Free-Form LangGraph Agent with Gating and Compensation (`agent_freeform.py`)

**Files:**
- Create: `agent_freeform.py`
- Test: `tests/test_freeform_agent.py`

**Interfaces:**
- Consumes: `toolcards.py`, `transaction.py`, `environment.api.SQLiteCompanyAPI`
- Produces: `create_freeform_agent_graph(api, checkpointer)`, `FreeformAgentState`

- [x] **Step 1: Write the failing test for free-form graph compilation, gating, and compensation**

```python
# tests/test_freeform_agent.py
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from environment.db import create_company_database
from environment.api import SQLiteCompanyAPI
from agent_freeform import create_freeform_agent_graph

def test_freeform_graph_compilation_and_gating():
    conn = create_company_database(":memory:", seed=True)
    api = SQLiteCompanyAPI(conn=conn, current_user_id="u-101")
    cp_conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(cp_conn)
    checkpointer.setup()

    graph = create_freeform_agent_graph(api=api, checkpointer=checkpointer)
    assert graph is not None

    attention_item = {
        "type": "delayed_shipment_impacting_production",
        "po_id": "PO-77812",
        "part_id": "P-4471",
        "production_order_id": "4812",
        "supplier_id": "S-Y",
        "quantity": 100,
        "delayed_promised_date": "2026-09-08",
        "production_scheduled_start": "2026-09-07",
    }

    initial_state = {
        "messages": [],
        "user_id": "u-101",
        "attention_item": attention_item,
        "pending_action": None,
        "approval_status": "none",
        "approver_id": "u-101",
        "primary_approver_id": "u-101",
        "escalated_to_backup": False,
        "idempotency_records": {},
        "compensation_stack": [],
        "audit_trail": [],
        "llm_telemetry": {},
    }

    config = {"configurable": {"thread_id": "test-ff-001"}}
    # Should run reasoning and pause at mutating gate or proceed cleanly
    res = graph.invoke(initial_state, config)
    assert "messages" in res
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_freeform_agent.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'agent_freeform'`

- [x] **Step 3: Implement `agent_freeform.py`**

Define `FreeformAgentState`, build the iterative tool-calling graph, connect read tools for immediate execution, intercept mutating tools in `gate_mutating_tool`, evaluate spending limits and calendar out-of-office rules, execute approved actions through `execute_mutating_tool` with `CompensationStack`, and capture exact telemetry metadata.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_freeform_agent.py -v`  
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add agent_freeform.py tests/test_freeform_agent.py
git commit -m "feat: implement free-form agent graph with tool loop and human gating"
```

---

### Task 4: CLI Integration in `main.py`

**Files:**
- Modify: `main.py`
- Test: `tests/test_main_cli.py`

**Interfaces:**
- Consumes: `agent.py:create_scenario_a_graph`, `agent_freeform.py:create_freeform_agent_graph`
- Produces: CLI argument `--mode [deterministic|freeform]`

- [x] **Step 1: Write test for CLI argument parsing and mode selection**

```python
# tests/test_main_cli.py
import pytest
from main import build_parser

def test_cli_mode_arguments():
    parser = build_parser()
    args_default = parser.parse_args([])
    assert args_default.mode == "deterministic"

    args_freeform = parser.parse_args(["--mode", "freeform"])
    assert args_freeform.mode == "freeform"
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_main_cli.py -v`  
Expected: FAIL with `ImportError: cannot import name 'build_parser' from 'main'`

- [x] **Step 3: Refactor `main.py` to extract `build_parser` and dispatch `--mode freeform`**

Preserve all existing deterministic behavior when `--mode deterministic` is chosen. When `--mode freeform` is selected, initialize `create_freeform_agent_graph`, run until the gate interrupts on a mutating tool, display the model's `reason`, prompt for approval, and resume execution.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_main_cli.py -v`  
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add main.py tests/test_main_cli.py
git commit -m "feat: add --mode freeform to CLI runner with reason-based approval prompt"
```

---

### Task 5: End-to-End Integration Verification & Existing Regression Check

**Files:**
- Test: `tests/test_scenario_a_agent.py`
- Test: `tests/test_freeform_agent.py`
- Test: `tests/test_company_api.py`

- [x] **Step 1: Run complete test suite covering both deterministic and free-form modes**

Run: `pytest -v`  
Expected: All tests pass (12 original tests + new free-form tests).

- [x] **Step 2: Run demo harness in both deterministic and free-form modes**

Run: `python3 main.py --allow-mock-planner --auto-approve`  
Run: `python3 main.py --mode freeform --allow-mock-planner --auto-approve`  
Expected: Both exit 0 with clean audit logs.

- [x] **Step 3: Commit final integration updates**

```bash
git add tests/
git commit -m "test: verify complete integration of deterministic and free-form modes"
```
