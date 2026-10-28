from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from f1rl.research.historical_source_records import verify_historical_records


def main():
    history = Path(os.environ.get('F1RL_WP3_EVIDENCE', r'D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1'))
    print(json.dumps(verify_historical_records(ROOT, history), sort_keys=True))


if __name__ == '__main__':
    main()
