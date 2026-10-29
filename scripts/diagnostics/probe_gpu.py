"""Legacy command for optional local CUDA hardware inspection."""
from pathlib import Path
from importlib import import_module
import sys

ROOT = Path(__file__).resolve().parents[2]

from src.interfaces.hardware_probe import capture_hardware as _capture_hardware, main as _main


def capture_hardware():
    return _capture_hardware(import_module('torch'))


def main():
    return _main(import_module('torch'))


if __name__ == "__main__":
    main()
