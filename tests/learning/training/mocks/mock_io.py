from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


class FailingStateWriter:
    def __init__(
        self,
        delegate: Callable[[Path, dict[str, object]], None],
        *,
        fail_on_call: int,
        error: Exception,
    ) -> None:
        self._delegate = delegate
        self._fail_on_call = fail_on_call
        self._error = error
        self.calls = 0

    def __call__(self, path: Path, state: dict[str, object]) -> None:
        self.calls += 1
        if self.calls == self._fail_on_call:
            raise self._error
        self._delegate(path, state)
