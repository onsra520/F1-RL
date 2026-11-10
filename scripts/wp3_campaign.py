from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0].startswith("parallel-"):
        from f1rl.local_demo.parallel_campaign import main as parallel_main
        return parallel_main(values)
    from f1rl.local_demo.campaign import main as sequential_main
    return sequential_main(values)


if __name__ == "__main__":
    raise SystemExit(main())
