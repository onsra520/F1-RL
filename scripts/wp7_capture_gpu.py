"""Legacy command for optional local CUDA hardware inspection."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interfaces.hardware_probe import capture_hardware, main


if __name__ == "__main__":
    main()
