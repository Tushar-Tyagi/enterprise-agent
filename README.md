# Enterprise Autonomous AI Agent Harness

An extensible AI agent harness for realistic enterprise supply chain and operations scenarios, backed by SQLite relational tables, fine-grained RBAC permissions, transactional workflows, and LangGraph human-in-the-loop state machines.

---

## 1. Quickstart & Environment Setup

### Environment Setup

```bash
# Option A: Conda (Python 3.12 or 3.13)
conda env create -f environment.yml
conda activate enterprise-agent

# Option B: Pip
pip install -r requirements.txt
```

### Environment Configuration

Configure your environment file (`.env`). All LLM calls use an OpenAI-compliant chat completions endpoint, so switching or adding any model provider is as simple as configuring `.env`:

```bash
cp .env.example .env
```

Example provider settings in `.env`:

```bash
# Google Gemini / Google AI Studio (Direct Key)
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_API_KEY=AIzaSy...
LLM_MODEL=gemini-2.5-flash
LLM_FALLBACK_MODEL=gemini-2.5-pro

# Or OpenRouter (Default)
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-v1-...
LLM_MODEL=google/gemini-3.1-flash-lite

# Or OpenAI Direct
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-proj-...
LLM_MODEL=gpt-4o-mini

# Or Local vLLM / Ollama
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=none
LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct
```

If no API key is configured, the CLI harness automatically falls back to the deterministic mock planner with zero configuration required.

---

## 2. One-Command Execution & CLI Options

### Run All Scenarios & Failure Cases (Default)
Runs Scenario A, clock advancement to Tuesday (`2026-09-08`) with dock verification, Scenario B quality lot reallocation, and Phase 4 failure invariant proofs:
```bash
python3 main.py
```

### Command-Line Arguments & Flags

| Flag | Values | Default | Purpose |
| :--- | :--- | :--- | :--- |
| `--scenario` | `a`, `b`, `both` | `both` | Select which scenario(s) to execute. |
| `--mode` | `freeform`, `deterministic` | `freeform` | Choose agent architecture: dynamic tool-calling or declared DAG workflow. |
| `--auto-approve` | Flag | `False` | Auto-approves mutating actions without waiting for interactive input. |
| `--allow-mock-planner`| Flag | `True` | Falls back to deterministic mock planner if `OPENROUTER_API_KEY` is unset. |
| `--skip-failures` | Flag | `False` | Skips running the Phase 4 failure cases and invariant checks. |

### Targeted Usage Examples

```bash
# Run only Scenario A interactively
python3 main.py --scenario a

# Run only Scenario B in auto-approve mode
python3 main.py --scenario b --auto-approve

# Run declared deterministic DAG workflow (Scenario A)
python3 main.py --scenario a --mode deterministic

# Non-interactive automated CI run across both scenarios
python3 main.py --auto-approve
```

### Run Test Suite (50 Tests)
Executes all unit, integration, and failure-case tests:
```bash
pytest -v
```

---

## 3. Scenarios Implemented

### Scenario A: Purchasing PO Expedite & Replacement
- **Actor**: Dana Whitfield (`u-101`, Purchasing Manager).
- **Trigger**: Delayed shipment notice `M-001` for `PO-77812` (Part `P-4471`, delayed by Supplier Y to `2026-09-08`).
- **Impact**: Production Order `4812` is scheduled to start on `2026-09-07`, threatening an immediate line shutdown.
- **Workflow**:
  1. Intercepts shipping delay and correlates parts with scheduled production orders.
  2. Evaluates alternate suppliers, rejecting unapproved vendor `S-X` and slow vendor `S-W`. Selects approved Supplier Z (`S-Z`, 2-day lead time).
  3. Pauses execution and prompts the user with impact details, cost analysis, and model justification.
  4. Cancels delayed `PO-77812` and creates replacement `PO-77815`.
  5. Notifies Production Supervisor Sam Taylor (`u-301`).
  6. Schedules arrival check on dock for delivery date (`2026-09-04`).

### Tuesday Dock Follow-Up Verification
- Advances system clock from `2026-09-02` to `2026-09-08`.
- Follow-up check scheduled in `ScheduledEvents` automatically matures and delivers verification email `M-CHECK-PO-77815` directly into Dana Whitfield's mailbox.

### Scenario B: Quality Hold & Lot Reallocation
- **Actor**: Casey Chen (`u-201`, Quality Manager).
- **Trigger**: Contamination alert places Lot `L-2093` (Part `P-1180`) on hold.
- **Impact**: Production Order `4820` requires 25 units of `P-1180` starting `2026-09-05`.
- **Workflow**:
  1. Detects hold event impacting upcoming production run.
  2. Scopes tool context to Quality Manager permissions (`erp:quality:write`).
  3. Identifies released lot `L-2094` with 30 available units (rejecting contaminated hold lot `L-2095`).
  4. Intercepts mutation and prompts Casey Chen for authorization.
  5. Atomically reallocates 25 units from `L-2094` to Order `4820`.
  6. Notifies supervisor Sam Taylor. If no covering lot existed, falls back to emailing Purchasing Manager Dana Whitfield to trigger emergency PO procurement.

### Phase 4: System Invariants & Failure Traps
The runner automatically verifies 5 critical architectural invariants:
1. **RBAC Scope Enforcement**: Casey Chen blocked from `create_purchase_order` (`erp:po:create` missing); Dana Whitfield blocked from `reallocate_lot_for_order` (`erp:quality:write` missing).
2. **Domain Trap Rejection**: Unapproved supplier `S-X` excluded from vendor selection; contaminated hold lot `L-2095` rejected on allocation; $30,000 PO blocked by Dana's $25,000 spending limit.
3. **Trigger Deduplication**: Repeated out-of-band scans on unchanged state emit 0 duplicate signals.
4. **Saga Compensation Rollback**: Mid-flight step failure triggers LIFO backward compensation unwinding, restoring canceled `PO-77812` back to `'OPEN'`.
5. **Idempotency Protection**: Replaying identical idempotency keys short-circuits execution and returns cached results without duplicate mutations.

---

## 4. How to Extend the Harness

### Adding a New Tool
1. Define the tool function and input schema in [`toolcards.py`](toolcards.py):
   ```python
   class ExpediteShipmentInput(BaseModel):
       po_id: str = Field(description="Purchase order identifier")
       reason: str = Field(description="Operational justification")
   
   @tool("expedite_shipment", args_schema=ExpediteShipmentInput)
   def expedite_shipment(po_id: str, reason: str) -> str:
       # Verify caller scopes and execute logic via api
       ...
   ```
2. Register the tool card in `TOOL_CARDS` with its `required_scope` (e.g., `erp:po:update`) and `is_mutating` flag.
3. Map any Saga inverse compensation actions in [`transaction.py`](transaction.py).

### Adding or Switching an LLM Provider

LLM initialization is centralized in [`agent_freeform.py`](agent_freeform.py) inside `get_llm()`. Because the factory uses the standard OpenAI chat completions protocol, **you do not need to write code to add or switch providers**—simply configure the endpoint, model, and API key in `.env`:

```bash
# Example 1: Google Gemini Direct (Google AI Studio)
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_API_KEY=AIzaSy...
LLM_MODEL=gemini-2.5-flash
LLM_FALLBACK_MODEL=gemini-2.5-pro

# Example 2: Local vLLM or Ollama
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=none
LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct

# Example 3: OpenAI Direct
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-proj-...
LLM_MODEL=gpt-4o-mini

# Example 4: Groq
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=gsk_...
LLM_MODEL=llama-3.3-70b-versatile
```

#### Centralized LLM Factory (`get_llm`)
The factory dynamically instantiates the model client according to `.env` parameters:
```python
from agent_freeform import get_llm

# Instantiates primary model configured in .env (or custom override)
llm = get_llm()

# Instantiates fallback model configured in .env
fallback_llm = get_llm(fallback=True)
```

#### Exact Telemetry Capture
All invocations route through `extract_llm_telemetry(msg)` to capture and persist upstream metadata:
- `prompt_tokens`, `completion_tokens`, `total_tokens`
- `prompt_tokens_details` (including cached tokens)
- exact upstream `cost` and `cost_details`

### Adding a New Operational Detector
1. Create a detector function in [`detector.py`](detector.py):
   ```python
   def detect_custom_condition(api: SQLiteCompanyAPI) -> list[OperationalSignal]:
       ...
   ```
2. Register the detector in `run_periodic_detector_scan()`. All signals must use `generate_trigger_hash()` to benefit from persistent deduplication.

### Adding a New Workflow
1. For declared deterministic pipelines, create a LangGraph state machine in [`agent.py`](agent.py) with explicit node transitions, version string, and checkpoints.
2. For autonomous tool-calling agents, register user role and scopes in [`context_quality.py`](context_quality.py) or `agent_freeform.py`.

---

## 5. Architectural Cuts & Trade-offs

A summary of explicit design choices:

1. **Lightweight Relational Model over Full ERP**: Kept core purchasing, quality lots, production orders, and user scopes in SQLite. Cut deep multi-level Bills of Materials (BOM), General Ledger accounts payable, and spatial warehouse bin tracking to maintain test velocity and clear failure visibility.
2. **Deterministic DAG vs. Autonomous Free-Form**: Provided both architectures. Scenario A's declared pipeline satisfies enterprise governance where purchasing steps must never be skipped or reordered. The free-form agent demonstrates dynamic multi-step deliberation with human gating before any state mutation.
3. **Application-Level RBAC & Idempotency**: RBAC scopes and idempotency keys are enforced in the API gateway layer (`SQLiteCompanyAPI`), ensuring identical protection regardless of whether actions originate from code, deterministic graphs, or LLM tool invocations.
4. **Saga Compensations over Distributed Locks**: Employs backward LIFO compensation unwinding (`cancel_purchase_order` -> `restore_purchase_order`) instead of long-lived database locks, avoiding deadlock across human approval wait cycles.

---

## 6. Project Artifacts & Documentation

- **Data Model Rationale**: [`MODEL.md`](MODEL.md)
- **Audit Logs**: [`audit_trail.jsonl`](audit_trail.jsonl)
- **Recorded Run Transcript**: [`recorded_run.txt`](recorded_run.txt)
