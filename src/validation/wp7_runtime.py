"""Current host and model artifact capture for WP7 protocol manifests.

This records the executing machine; it is not historical run provenance and
does not grant training or scientific admission.
"""
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def cpu_model_name() -> str:
    model = platform.processor().strip() or os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
    if model:
        return model
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
                key, separator, value = line.partition(":")
                if separator and key.strip() == "model name" and value.strip():
                    return value.strip()
        except OSError:
            pass
    return "unavailable"


def capture_runtime_fingerprint() -> dict:
    """Capture package, host and GPU-probe details for a current invocation."""
    root = Path(__file__).resolve().parents[2]
    packages = {name: metadata.version(name) for name in
                ("numpy", "gymnasium", "stable-baselines3", "torch", "PyYAML")}
    result = subprocess.run([sys.executable, "-B", "-m", "scripts.diagnostics.probe_gpu"],
                            cwd=root, capture_output=True, text=True, check=True, timeout=30)
    gpu = json.loads(result.stdout)
    return {"python_version": platform.python_version(), "implementation": platform.python_implementation(),
            "os": platform.system(), "os_release": platform.release(), "packages": packages,
            "cpu": {"architecture": platform.machine(), "model": cpu_model_name(),
                    "logical_processors": os.cpu_count()},
            **gpu}


def hash_model_artifacts(artifacts) -> dict:
    """Hash explicitly supplied existing model/checkpoint ZIP files."""
    if not isinstance(artifacts, dict) or any(not isinstance(k, str) or not k.strip() for k in artifacts):
        raise ValueError("manifest artifact identities required")
    result = {}
    for identity, source in artifacts.items():
        path = Path(source)
        if path.suffix.lower() != ".zip" or path.is_symlink() or not path.is_file():
            raise ValueError("manifest requires explicit existing model/checkpoint ZIP files")
        result[identity] = {"sha256": sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}
    return result
