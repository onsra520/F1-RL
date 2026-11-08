from dataclasses import dataclass


def _validate_ids(values: frozenset[str], name: str) -> None:
    if not isinstance(values, frozenset):
        raise ValueError(f"{name} must be a frozenset")
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must contain non-empty string ids")


@dataclass(frozen=True, slots=True)
class InteractionState:
    active_checkpoint_ids: frozenset[str]
    active_finish_line_ids: frozenset[str]

    def __post_init__(self) -> None:
        _validate_ids(self.active_checkpoint_ids, "active_checkpoint_ids")
        _validate_ids(self.active_finish_line_ids, "active_finish_line_ids")
