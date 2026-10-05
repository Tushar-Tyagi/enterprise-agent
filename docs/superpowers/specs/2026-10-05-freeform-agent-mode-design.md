# Free-Form Agent Mode with Toolcards, RBAC, Idempotency, and Compensation

**Date:** 2026-10-05  
**Status:** Draft / Approved Design  

---

## 1. Overview & Objectives

This specification defines a free-flowing, tool-calling agent mode for the enterprise agent harness. When an attention item is detected, the agent autonomously navigates ERP state, unread mail, and vendor catalogs via a LangGraph tool loop. 

Key constraints:
1. **Preservation of Deterministic Pipeline**: The existing fixed state machine in `agent.py` (`create_scenario_a_graph`) remains intact and functional.
2. **Centralized Toolcards**: All tool definitions, parameter schemas, and system prompt toolcards live in a single file: `toolcards.py`.
3. **Mandatory Model Rationale**: Every tool invocation schema requires a `reason: str` field so the model explains why it is invoking that tool.
4. **RBAC Scope Verification**: Every tool executes under the active user's session in `SQLiteCompanyAPI`, enforcing role-based permissions (`mail:read`, `po:create`, etc.).
5. **Human Approval Interception**: Mutating actions are intercepted before execution. The CLI displays the model's supplied `reason`, calculates spending limits and calendar out-of-office rules, and routes to backup approvers when applicable.
6. **Idempotency & Saga Compensation**: Every mutating action accepts an `idempotency_key` and registers an inverse compensating operation on a LIFO stack. If a downstream action fails, compensation unrolls previously executed mutations.
7. **Exact LLM Telemetry**: Every LLM invocation captures exact token usage and cost metadata directly from OpenRouter responses.

---

## 2. File Organization

- `toolcards.py`: Contains Pydantic schemas, tool definitions, toolcard formatting, and permission mappings.
- `transaction.py`: Contains `IdempotencyRegistry` and `CompensationStack` for transactional rollback.
- `agent_freeform.py`: LangGraph implementation of the free-flowing tool-calling agent with human approval interception and Saga compensation.
- `agent.py`: Preserved deterministic Scenario A pipeline (untouched).
- `main.py`: CLI harness updated with `--mode [deterministic|freeform]` support.
- `tests/test_freeform_agent.py`: Complete test coverage for tool execution, RBAC gating, reason validation, idempotency, and compensation unrolling.

---

## 3. Tool Architecture (`toolcards.py`)

### Base Tool Input Schema
```python
class BaseToolInput(BaseModel):
    reason: str = Field(
        ...,
        description="Detailed explanation of why the agent is invoking this specific tool in relation to resolving the attention item.",
    )
```

### Read Tools
- `read_emails(recipient_id: str, unread_only: bool = True, reason: str)` -> Scope: `mail:read`
- `get_purchase_order(po_id: str, reason: str)` -> Scope: `po:read`
- `get_production_orders(part_id: Optional[str] = None, reason: str)` -> Scope: `production:read`
- `query_suppliers(part_id: Optional[str] = None, approved_only: bool = True, reason: str)` -> Scope: `suppliers:read`
- `get_calendar_events(user_id: str, reason: str)` -> Scope: `calendar:read`

### Mutating Tools
- `create_purchase_order(po_id: str, part_id: str, supplier_id: str, quantity: int, unit_price: float, promised_date: str, idempotency_key: str, reason: str)` -> Scope: `po:create`
- `cancel_purchase_order(po_id: str, reason: str, idempotency_key: str)` -> Scope: `po:cancel`
- `send_email(recipient_id: str, subject: str, body: str, idempotency_key: str, reason: str)` -> Scope: `mail:send`
- `notify_production(supervisor_id: str, order_id: str, message: str, idempotency_key: str, reason: str)` -> Scope: `production:notify`
- `schedule_event(trigger_date: str, target_table: str, payload: dict, idempotency_key: str, reason: str)` -> Scope: `system:admin`

Each tool definition exposes `is_mutating: bool` and `required_scope: str`.

---

## 4. Idempotency & Compensation (`transaction.py`)

### Idempotency Registry
Tracks executed mutating calls by `(tool_name, idempotency_key)`.
- If a key has already completed successfully, returns the recorded result immediately.
- If a key is currently in-flight, prevents concurrent duplicate execution.
- If an action is compensated, marks status as `COMPENSATED`.

### Saga Compensation Stack
When a mutating action executes, it registers an inverse action tuple:
- `create_purchase_order` -> inverse: `cancel_purchase_order(po_id=po_id, reason="Rollback compensation")`
- `cancel_purchase_order` -> inverse: `reopen_po(po_id=po_id, reason="Rollback compensation")`
- `send_email` -> inverse: sends corrective follow-up email `CORRECTION: Prior action aborted.`
- `notify_production` -> inverse: sends corrective notification to supervisor.
- `schedule_event` -> inverse: deletes scheduled event from `ScheduledEvents`.

When any downstream action fails with an unhandled exception:
1. Runtime traps error.
2. Unrolls `compensation_stack` in LIFO order.
3. Calls each inverse operation with the authorized user session.
4. Records compensation outcomes in `audit_trail`.
5. Re-raises the error or sets state to `failed`.

---

## 5. Free-Form Agent State Graph (`agent_freeform.py`)

### Agent State
```python
class FreeformAgentState(TypedDict):
    messages: List[Any]
    user_id: str
    attention_item: Dict[str, Any]
    pending_action: Optional[Dict[str, Any]]
    approval_status: str  # "none", "pending", "approved", "rejected"
    approver_id: str
    primary_approver_id: str
    escalated_to_backup: bool
    idempotency_records: Dict[str, Any]
    compensation_stack: List[Dict[str, Any]]
    audit_trail: List[str]
    llm_telemetry: Dict[str, Any]
```

### Graph Topology
1. `START` -> `agent_reasoning`
2. `agent_reasoning` -> conditional edge:
   - If AI response has no tool calls -> `END`
   - If tool call is read-only -> `execute_read_tool` -> `agent_reasoning`
   - If tool call is mutating -> `gate_mutating_tool`
3. `gate_mutating_tool`:
   - Checks spending threshold and calendar OOO rules.
   - Sets `approval_status = "pending"`, stores `pending_action`.
   - Conditional edge:
     - If `approval_status == "approved"` -> `execute_mutating_tool`
     - Else -> `END` (pauses for human checkpointer)
4. `execute_mutating_tool`:
   - Checks idempotency.
   - Executes mutation on `SQLiteCompanyAPI`.
   - Records inverse action in `compensation_stack`.
   - On exception -> executes `unwind_compensation_stack`.
   - Appends `ToolMessage` with result.
   - Loops back to `agent_reasoning`.

---

## 6. CLI Runner Support (`main.py`)

Add command-line options:
- `--mode [deterministic|freeform]`: Default is `deterministic` to preserve backward compatibility.
- Interactive prompt for free-form mode prints the tool name, arguments, and the model's `reason` before prompting for `[y/N]` authorization.
