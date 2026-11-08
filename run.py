from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
BACKEND_PORT = 8000
FRONTEND_PORT = 5173
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"
FRONTEND_URL = f"http://127.0.0.1:{FRONTEND_PORT}"
STARTUP_TIMEOUT = 30.0
POLL_INTERVAL = 0.25
HTTP_TIMEOUT = 1.0
IS_WINDOWS = os.name == "nt"


class LaunchError(RuntimeError):
    def __init__(self, code: str, operation: str, message: str):
        super().__init__(message)
        self.code = code
        self.operation = operation


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen[str]
    reader: threading.Thread


def _read_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as response:
        return response.read().decode("utf-8")


def _read_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as response:
        return response.read()


def _local_runtime_source_sha256() -> str:
    from f1rl.local_demo.runtime.catalog import runtime_source_sha256
    return runtime_source_sha256(ROOT)


def _local_runtime_inputs_sha256() -> str:
    from f1rl.local_demo.runtime.catalog import runtime_inputs_sha256
    return runtime_inputs_sha256(ROOT)


def backend_compatibility() -> tuple[bool, str]:
    try:
        health = json.loads(_read_text(f"{BACKEND_URL}/api/v1/health"))
        if (health.get("status") != "ok" or health.get("runtime_mode") != "development" or
                health.get("scientific_admission") != "disabled" or
                health.get("protocol_version") != "f1rl_runtime_v1" or
                not isinstance(health.get("source_sha256"), str) or
                len(health["source_sha256"]) != 64 or
                any(char not in "0123456789abcdef" for char in health["source_sha256"])):
            return False, "health response does not match the development runtime protocol"
        if health["source_sha256"] != _local_runtime_source_sha256():
            return False, "runtime source does not match the local checkout"
        if health.get("runtime_inputs_sha256") != _local_runtime_inputs_sha256():
            return False, "runtime input files do not match the local checkout"
        spec = json.loads(_read_text(f"{BACKEND_URL}/openapi.json"))
        info = spec.get("info", {})
        if info.get("title") != "F1-RL Development Runtime" or info.get("version") != "1.0.0":
            return False, "OpenAPI title or version does not match F1-RL Runtime 1.0.0"
        return True, "compatible F1-RL Runtime 1.0.0"
    except (OSError, ImportError, ValueError, urllib.error.URLError, json.JSONDecodeError, UnicodeDecodeError) as error:
        return False, f"runtime health/version check failed: {error}"


def frontend_compatibility() -> tuple[bool, str]:
    try:
        page = _read_text(FRONTEND_URL)
        production_index = WEB_ROOT / "dist" / "index.html"
        if production_index.is_file():
            if page != production_index.read_bytes().decode("utf-8") or "<title>Racing Sandbox · F1-RL</title>" not in page:
                return False, "page does not match the local production build"
            runtime_url = re.search(r'<meta name="f1rl-runtime-url" content="([^"]+)">', page)
            if not runtime_url or runtime_url.group(1) not in ("http://127.0.0.1:8000", "http://localhost:8000"):
                return False, "production frontend Runtime URL does not use the default backend on port 8000"
            assets = re.findall(r'''(?:src|href)=["'](/assets/[A-Za-z0-9._-]+\.(?:js|css))["']''', page)
            if not any(asset.endswith(".js") for asset in assets):
                return False, "production build has no entry script"
            manifest = json.loads((production_index.parent / ".vite" / "manifest.json").read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or not manifest:
                return False, "production build manifest is invalid"
            files = set()
            for entry in manifest.values():
                if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
                    return False, "production build manifest is invalid"
                files.add(entry["file"])
                for key in ("css", "assets"):
                    values = entry.get(key, [])
                    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                        return False, "production build manifest is invalid"
                    files.update(values)
                for key in ("imports", "dynamicImports"):
                    values = entry.get(key, [])
                    if not isinstance(values, list) or not all(isinstance(value, str) and value in manifest for value in values):
                        return False, "production build manifest is missing an imported chunk"
            if not {asset.lstrip("/") for asset in assets}.issubset(files):
                return False, "production build manifest does not identify the entry assets"
            for asset in sorted(files):
                path = (production_index.parent / asset).resolve()
                if not path.is_relative_to(production_index.parent.resolve()) or not path.is_file():
                    return False, "production build is missing a manifest asset"
                if _read_bytes(f"{FRONTEND_URL}/{urllib.parse.quote(asset, safe='/')}") != path.read_bytes():
                    return False, "served frontend asset does not match the local production build"
            return True, "compatible F1-RL production frontend"
        if "<title>Racing Sandbox · F1-RL</title>" not in page or "/src/main.tsx" not in page:
            return False, "page is not the F1-RL Vite development frontend"
        _read_text(f"{FRONTEND_URL}/@vite/client")
        runtime_module = _read_text(f"{FRONTEND_URL}/src/lib/runtime.ts")
        match = re.search(r"import\.meta\.env\s*=\s*(\{.*?\});", runtime_module, re.DOTALL)
        if not match:
            return False, "could not verify the frontend Runtime URL"
        runtime_env = json.loads(match.group(1))
        configured_url = runtime_env.get("VITE_RUNTIME_URL", "")
        if configured_url not in ("", "http://127.0.0.1:8000", "http://localhost:8000"):
            return False, "configured Runtime URL does not use the default backend on port 8000"
        return True, "compatible F1-RL Vite frontend"
    except (OSError, urllib.error.URLError, json.JSONDecodeError, UnicodeDecodeError) as error:
        return False, f"frontend readiness check failed: {error}"


def port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.35):
            return True
    except OSError:
        return False


def _stream_output(name: str, stream) -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    for line in iter(stream.readline, ""):
        safe_line = line.rstrip().encode(encoding, errors="replace").decode(encoding)
        print(f"[{name}] {safe_line}", flush=True)


def _process_command(name: str) -> list[str]:
    if name == "BACKEND":
        return [sys.executable, "-m", "f1rl.local_demo.runtime"]
    dist = WEB_ROOT / "dist"
    if (dist / "index.html").is_file():
        return [sys.executable, "-m", "http.server", str(FRONTEND_PORT), "--bind", "127.0.0.1", "--directory", str(dist)]
    if IS_WINDOWS:
        command = f"npm run dev -- --port {FRONTEND_PORT} --strictPort"
        return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/s", "/c", command]
    return ["npm", "run", "dev", "--", "--port", str(FRONTEND_PORT), "--strictPort"]


def terminate_process_tree(process: subprocess.Popen[str], *, graceful: bool = True) -> None:
    if IS_WINDOWS:
        if graceful and process.poll() is None:
            try:
                process.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM))
                process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                pass
        result = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=5,
        )
        if result.returncode and process.poll() is None:
            raise RuntimeError(f"taskkill failed with exit code {result.returncode}")
        return
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=3)


class Launcher:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.trace_id = str(uuid.uuid4())
        self.owned: list[ManagedProcess] = []
        self.backend_requested = not args.frontend_only
        self.frontend_requested = not args.backend_only

    def log(self, name: str, message: str) -> None:
        print(f"[{name}] {message}", flush=True)

    def _start(self, name: str) -> ManagedProcess:
        if name == "FRONTEND" and (WEB_ROOT / "dist" / "index.html").is_file() and not (WEB_ROOT / "dist" / ".vite" / "manifest.json").is_file():
            raise LaunchError("build_incomplete", "start_frontend", "Production build manifest is missing; rebuild the frontend")
        if name == "FRONTEND" and not (WEB_ROOT / "dist" / "index.html").is_file() and shutil.which("npm") is None:
            raise LaunchError("npm_missing", "start_frontend", "npm was not found; install Node.js/npm and ensure npm is on PATH")
        options = {
            "cwd": ROOT if name == "BACKEND" else WEB_ROOT,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
        }
        if IS_WINDOWS:
            options["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            options["start_new_session"] = True
        try:
            process = subprocess.Popen(_process_command(name), **options)
        except OSError as error:
            raise LaunchError("process_start_failed", f"start_{name.lower()}", f"Could not start {name.lower()}: {error}") from error
        reader = threading.Thread(target=_stream_output, args=(name, process.stdout), daemon=True)
        managed = ManagedProcess(name, process, reader)
        self.owned.append(managed)
        reader.start()
        self.log(name, f"Started PID {process.pid}")
        return managed

    def _wait_for(self, service: ManagedProcess, check: Callable[[], tuple[bool, str]], url: str) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT
        last_reason = "waiting for the service to answer"
        while time.monotonic() < deadline:
            returncode = service.process.poll()
            if returncode is not None:
                raise LaunchError("service_exited", f"start_{service.name.lower()}",
                                  f"{service.name.title()} exited before becoming ready (exit code {returncode})")
            ready, last_reason = check()
            if ready:
                self.log(service.name, f"READY {url}")
                return
            time.sleep(POLL_INTERVAL)
        raise LaunchError("startup_timeout", f"start_{service.name.lower()}",
                          f"{service.name.title()} did not become ready within {STARTUP_TIMEOUT:g}s: {last_reason}")

    def _ensure_backend(self) -> None:
        if port_is_open(BACKEND_PORT):
            compatible, reason = backend_compatibility()
            if not compatible:
                raise LaunchError("port_in_use", "check_backend_port",
                                  f"Port {BACKEND_PORT} is occupied by an incompatible service ({reason}). "
                                  "No existing process was stopped; stop that service manually and retry.")
            self.log("BACKEND", f"Reusing {reason}")
            return
        process = self._start("BACKEND")
        self._wait_for(process, backend_compatibility, BACKEND_URL)

    def _ensure_frontend(self) -> None:
        if port_is_open(FRONTEND_PORT):
            compatible, reason = frontend_compatibility()
            if not compatible:
                raise LaunchError("port_in_use", "check_frontend_port",
                                  f"Port {FRONTEND_PORT} is occupied by an incompatible service ({reason}). "
                                  "No existing process was stopped; stop that service manually and retry.")
            self.log("FRONTEND", f"Reusing {reason}")
            return
        process = self._start("FRONTEND")
        self._wait_for(process, frontend_compatibility, FRONTEND_URL)

    def start(self) -> None:
        self.log("LAUNCHER", f"trace_id={self.trace_id} operation=launch outcome=starting")
        if self.backend_requested:
            self._ensure_backend()
        if self.frontend_requested:
            self._ensure_frontend()
        print("\nF1-RL Racing Sandbox\n")
        if self.backend_requested:
            print(f"Backend   READY  {BACKEND_URL}")
        if self.frontend_requested:
            print(f"Frontend  READY  {FRONTEND_URL}")
        if self.backend_requested and self.frontend_requested:
            if self.args.no_browser:
                print("Browser not opened (--no-browser).")
            else:
                try:
                    opened = webbrowser.open(FRONTEND_URL, new=1)
                except OSError as error:
                    opened = False
                    self.log("LAUNCHER", f"trace_id={self.trace_id} operation=open_browser outcome=error cause={error}")
                if opened:
                    print("Browser opened.")
                else:
                    print(f"Browser could not be opened automatically; open {FRONTEND_URL}.")
        print("Press Ctrl+C to stop.", flush=True)

    def monitor(self) -> None:
        while True:
            for service in self.owned:
                returncode = service.process.poll()
                if returncode is not None:
                    raise LaunchError("service_exited", f"monitor_{service.name.lower()}",
                                      f"{service.name.title()} exited unexpectedly (exit code {returncode})")
            time.sleep(POLL_INTERVAL)

    def cleanup(self) -> None:
        for service in reversed(self.owned):
            try:
                terminate_process_tree(service.process, graceful=service.name == "BACKEND")
                service.reader.join(timeout=1)
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                self.log("LAUNCHER", f"trace_id={self.trace_id} operation=cleanup error_code=cleanup_failed "
                         f"service={service.name.lower()} outcome=error cause={error}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the F1-RL Racing Sandbox development services.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--backend-only", action="store_true", help="Start or reuse only the Python Runtime backend.")
    modes.add_argument("--frontend-only", action="store_true", help="Start or reuse only the Vite frontend.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically.")
    return parser


def _handle_console_break(_signum, _frame) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    if IS_WINDOWS and hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handle_console_break)
    args = _parser().parse_args(argv)
    launcher = Launcher(args)
    try:
        launcher.start()
        launcher.monitor()
    except KeyboardInterrupt:
        launcher.log("LAUNCHER", f"trace_id={launcher.trace_id} operation=shutdown outcome=stop_requested")
        return 0
    except LaunchError as error:
        launcher.log("LAUNCHER", f"trace_id={launcher.trace_id} module=launcher operation={error.operation} "
                     f"outcome=error error_code={error.code} cause={error}")
        return 1
    finally:
        launcher.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
