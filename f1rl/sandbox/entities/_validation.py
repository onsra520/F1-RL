from math import isfinite


def validate_id(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("id must be a non-empty string")


def validate_positive_number(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
