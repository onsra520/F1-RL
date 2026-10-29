"""Versionless, explicit CLI router for F1-RL standalone scripts."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from .catalog import BY_COMMAND, WORKFLOWS

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


def manifest() -> list[dict[str, str]]:
    return [
        {"group": item.group, "name": item.name, "script": item.script, "scope": item.scope}
        for item in sorted(WORKFLOWS, key=lambda entry: (entry.group, entry.name))
    ]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help", "list"):
        if args and args[0] == "list" and args[1:] == ["--json"]:
            print(json.dumps(manifest(), indent=2))
            return 0
        if args and args[0] == "list" and args[1:] and args[1:] != ["--json"]:
            print("usage: python -m scripts.workflows list [--json]", file=sys.stderr)
            return 2
        print("usage: python -m scripts.workflows <group> <action> [script-arguments...]")
        print("       python -m scripts.workflows list [--json]")
        print("Workflow groups: " + ", ".join(sorted({item.group for item in WORKFLOWS if item.group != "internal"})))
        for item in manifest():
            if item["scope"] != "internal-only":
                print(f'  {item["group"]} {item["name"]} [{item["scope"]}]')
        print("Historical script paths are retained; commands preserve their original scientific gates.")
        return 0
    if len(args) < 2:
        print("expected <group> <action>; use 'list' for supported commands", file=sys.stderr)
        return 2
    item = BY_COMMAND.get((args[0], args[1]))
    if item is None or item.scope == "internal-only":
        print("unknown or internal-only workflow; use 'list'", file=sys.stderr)
        return 2
    script = (SCRIPTS / item.script).resolve()
    if script.parent != SCRIPTS.resolve() or not script.is_file():
        print("workflow script unavailable; refusing to execute", file=sys.stderr)
        return 2
    # Do not interpret, coerce, or grant permissions for downstream arguments.
    # The existing entrypoint owns admission, output paths, and fail-closed behavior.
    return subprocess.call([sys.executable, str(script), *args[2:]])


if __name__ == "__main__":
    raise SystemExit(main())
