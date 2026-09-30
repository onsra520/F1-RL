from __future__ import annotations

from pathlib import Path
from typing import Callable, ClassVar


class MockPPOModel:
    progress_points: ClassVar[tuple[int, ...]] = (25_008, 51_200)
    learn_error: ClassVar[Exception | None] = None
    save_error_at_call: ClassVar[int | None] = None
    instances: ClassVar[list["MockPPOModel"]] = []

    def __init__(self, env: object, config: object) -> None:
        self.env = env
        self.config = config
        self._num_timesteps = 0
        self.learn_calls: list[int] = []
        self.save_calls: list[Path] = []

    @classmethod
    def reset(cls) -> None:
        cls.progress_points = (25_008, 51_200)
        cls.learn_error = None
        cls.save_error_at_call = None
        cls.instances = []

    @classmethod
    def create(cls, env: object, config: object) -> "MockPPOModel":
        model = cls(env, config)
        cls.instances.append(model)
        return model

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    def learn(
        self,
        total_timesteps: int,
        *,
        progress_hook: Callable[[int], None] | None = None,
    ) -> "MockPPOModel":
        self.learn_calls.append(total_timesteps)
        if self.learn_error is not None:
            raise self.learn_error
        for step in self.progress_points:
            self._num_timesteps = step
            if progress_hook is not None:
                progress_hook(step)
        return self

    def save(self, path: str | Path) -> Path:
        archive = Path(path)
        self.save_calls.append(archive)
        if self.save_error_at_call == len(self.save_calls):
            raise RuntimeError("mock save failure")
        archive.write_text("mock model", encoding="utf-8")
        return archive
