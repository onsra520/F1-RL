from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
from src.validation.historical_source_records import verify_historical_records


HISTORY = Path(os.environ.get('F1RL_WP3_EVIDENCE', r'D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1'))


def main():
    print(json.dumps(verify_historical_records(ROOT, HISTORY), sort_keys=True))


if __name__ == '__main__':
    main()
