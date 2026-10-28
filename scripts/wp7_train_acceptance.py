"""Fixed scripted TRAIN acceptance CLI; never scientific admission."""
import argparse
import json
from pathlib import Path

from src.research.wp7_train_acceptance import (
    PROFILE, _write_new, run_acceptance, verify_gym_finish_prefix,
)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='new external evidence directory; never overwritten')
    args=parser.parse_args()
    report=run_acceptance(Path(__file__).resolve().parents[1],args.output)
    print(json.dumps(report,indent=2,allow_nan=False))
    return 0 if report['all_pass'] else 2


if __name__=='__main__':
    raise SystemExit(main())
