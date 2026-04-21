class ExecutionPlanError(Exception):
    """Base error type for execution plan generation."""


class JsonFormatError(ExecutionPlanError):
    """Raised when input JSON does not match expected schema."""


class InvalidRegisterWritePlanError(ExecutionPlanError):
    """Raised when planned register writes target addresses not enabled in source bitstream."""
