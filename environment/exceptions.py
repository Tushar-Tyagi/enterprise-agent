class CompanyAPIError(Exception):
    """Base exception for all company API errors."""
    pass


class UnauthorizedError(CompanyAPIError, PermissionError):
    """Raised when a user attempts an action without the required scope/permission."""
    def __init__(self, user_id: str, required_scope: str, message: str = None):
        self.user_id = user_id
        self.required_scope = required_scope
        msg = message or f"User '{user_id}' lacks required permission scope: '{required_scope}'"
        super().__init__(msg)


class ApprovalLimitExceededError(CompanyAPIError):
    """Raised when an operation exceeds the user's authorized spending or approval threshold."""
    def __init__(self, user_id: str, amount: float, max_limit: float, backup_approver_id: str = None):
        self.user_id = user_id
        self.amount = amount
        self.max_limit = max_limit
        self.backup_approver_id = backup_approver_id
        msg = (
            f"User '{user_id}' exceeded approval limit of ${max_limit:,.2f} with requested amount ${amount:,.2f}."
            + (f" Escalate to backup approver: '{backup_approver_id}'." if backup_approver_id else " No backup approver specified.")
        )
        super().__init__(msg)


class EntityNotFoundError(CompanyAPIError):
    """Raised when a requested database entity does not exist."""
    pass


class ValidationError(CompanyAPIError):
    """Raised when business logic validation fails."""
    pass
