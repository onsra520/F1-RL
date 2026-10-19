"""CPU-only executable bootstrap; CLI preflight never starts a pilot."""
import os
from pathlib import Path
import sys

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[variable] = '4'
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from f1rl.exploratory.cli import main
    raise SystemExit(main())
