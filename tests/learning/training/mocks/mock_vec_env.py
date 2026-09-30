from __future__ import annotations


class MockVecEnv:
    def __init__(self, *, close_error: Exception | None = None) -> None:
        self.closed = False
        self.close_calls = 0
        self.close_error = close_error

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True
        if self.close_error is not None:
            raise self.close_error
