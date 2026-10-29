from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
from src.validation.historical_source_records import verify_historical_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_historical_records(ROOT, args.history), sort_keys=True))


if __name__ == '__main__':
    main()
