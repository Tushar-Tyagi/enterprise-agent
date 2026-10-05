"""
Transaction Guard, Idempotency Registry, and Saga Compensation Ledger.
Guarantees idempotent execution and provides LIFO rollback for multi-step enterprise agent operations.
"""

from typing import Any, Callable, Dict, List, Optional
import logging

from environment.api import SQLiteCompanyAPI

logger = logging.getLogger("enterprise_agent.transaction")


class IdempotencyRegistry:
    """
    Tracks executed mutating operations by idempotency key to prevent duplicate mutations.
    """

    def __init__(self, records: Optional[Dict[str, Any]] = None):
        self._records: Dict[str, Dict[str, Any]] = dict(records) if records else {}

    def has_executed(self, key: str) -> bool:
        rec = self._records.get(key)
        return rec is not None and rec.get("status") == "SUCCESS"

    def record_success(self, key: str, result: Any) -> None:
        self._records[key] = {
            "status": "SUCCESS",
            "result": result,
        }

    def get_result(self, key: str) -> Any:
        rec = self._records.get(key)
        return rec.get("result") if rec else None

    def mark_compensated(self, key: str) -> None:
        if key in self._records:
            self._records[key]["status"] = "COMPENSATED"

    def dump_records(self) -> Dict[str, Any]:
        return dict(self._records)


class CompensationAction:
    """Represents a single registered compensation step."""

    def __init__(
        self,
        action_name: str,
        compensating_callable: Callable[[], Any],
        description: str = "",
        idempotency_key: Optional[str] = None,
    ):
        self.action_name = action_name
        self.compensating_callable = compensating_callable
        self.description = description
        self.idempotency_key = idempotency_key


class CompensationStack:
    """
    LIFO Stack of compensating actions to unwind on downstream workflow failure.
    """

    def __init__(self, api: Optional[SQLiteCompanyAPI] = None, idempotency_registry: Optional[IdempotencyRegistry] = None):
        self.api = api
        self.registry = idempotency_registry or IdempotencyRegistry()
        self._stack: List[CompensationAction] = []

    def push(
        self,
        action_name: str,
        compensating_callable: Callable[[], Any],
        description: str = "",
        idempotency_key: Optional[str] = None,
    ) -> None:
        action = CompensationAction(
            action_name=action_name,
            compensating_callable=compensating_callable,
            description=description,
            idempotency_key=idempotency_key,
        )
        self._stack.append(action)

    def is_empty(self) -> bool:
        return len(self._stack) == 0

    def unwind(self) -> List[Dict[str, Any]]:
        """
        Unwind all registered compensating actions in reverse order (LIFO).
        Returns a log of compensation execution results.
        """
        log: List[Dict[str, Any]] = []

        while self._stack:
            item = self._stack.pop()
            entry: Dict[str, Any] = {
                "action": item.action_name,
                "description": item.description,
                "status": "PENDING",
            }
            try:
                res = item.compensating_callable()
                entry["status"] = "COMPENSATED"
                entry["result"] = res
                if item.idempotency_key:
                    self.registry.mark_compensated(item.idempotency_key)
            except Exception as exc:
                entry["status"] = "FAILED"
                entry["error"] = str(exc)
                logger.error(f"Compensation action '{item.action_name}' failed: {exc}")

            log.append(entry)

        return log

    def clear(self) -> None:
        self._stack.clear()
