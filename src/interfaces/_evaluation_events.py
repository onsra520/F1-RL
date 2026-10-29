"""WP3 evaluation lifecycle JSONL records."""
from __future__ import annotations

import json
import os
from pathlib import Path

_EVENT_SCHEMA = "f1rl_wp3_train_eval_event_v1"


def _append_event(path: Path, event: dict) -> None:
    record = {"schema_version": _EVENT_SCHEMA, **event}
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _read_events(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError("evaluation lifecycle JSONL has an incomplete record")
            row = json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
            if row.get("schema_version") != _EVENT_SCHEMA:
                raise ValueError("evaluation event schema mismatch")
            records.append(row)
    return records
