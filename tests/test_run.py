import io
import json
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import Mock, patch

import pytest

import run


HEALTH = {
    "status": "ok",
    "runtime_mode": "development",
    "scientific_admission": "disabled",
    "protocol_version": "f1rl_runtime_v1",
    "source_sha256": "a" * 64,
}
OPENAPI = {"info": {"title": "F1-RL Development Runtime", "version": "1.0.0"}}
FRONTEND = '<title>Racing Sandbox · F1-RL</title><script type="module" src="/src/main.tsx"></script>'
RUNTIME_MODULE = 'import.meta.env = {"BASE_URL": "/", "DEV": true, "PROD": false}; export const runtime = "ok";'


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def open_response(url, **_kwargs):
    if url.endswith("/api/v1/health"):
        content = json.dumps(HEALTH).encode()
    elif url.endswith("/openapi.json"):
        content = json.dumps(OPENAPI).encode()
    elif url.endswith("/@vite/client"):
        content = b"vite client"
    elif url.endswith("/src/lib/runtime.ts"):
        content = RUNTIME_MODULE.encode()
    else:
        content = FRONTEND.encode()
    return Response(content)


def connected(_address, **_kwargs):
    return Mock(__enter__=Mock(return_value=Mock()), __exit__=Mock(return_value=False))


def refused(_address, **_kwargs):
    raise ConnectionRefusedError


class Process:
    def __init__(self, poll_values=()):
        self.pid = 7312
        self.stdout = io.StringIO("")
        self.poll_values = list(poll_values)
        self.returncode = None

    def poll(self):
        if self.poll_values:
            self.returncode = self.poll_values.pop(0)
        return self.returncode


class NarrowConsole:
    encoding = "cp1252"

    def __init__(self):
        self.value = ""

    def write(self, text):
        text.encode(self.encoding)
        self.value += text
        return len(text)

    def flush(self):
        pass


def test_default_reuses_compatible_services_and_opens_browser_once():
    output = io.StringIO()
    with (
        patch.object(run.socket, "create_connection", side_effect=connected),
        patch.object(run.urllib.request, "urlopen", side_effect=open_response),
        patch.object(run.subprocess, "Popen") as popen,
        patch.object(run.webbrowser, "open", return_value=True) as browser,
        patch.object(run.time, "sleep", side_effect=KeyboardInterrupt),
        redirect_stdout(output),
        redirect_stderr(output),
    ):
        assert run.main([]) == 0

    popen.assert_not_called()
    browser.assert_called_once_with("http://127.0.0.1:5173", new=1)
    assert "Backend   READY  http://127.0.0.1:8000" in output.getvalue()
    assert "Frontend  READY  http://127.0.0.1:5173" in output.getvalue()


def test_incompatible_backend_port_is_reported_without_killing_or_spawning():
    output = io.StringIO()
    response = lambda _url, **_kwargs: Response(json.dumps({**HEALTH, "protocol_version": "unknown"}).encode())
    with (
        patch.object(run.socket, "create_connection", side_effect=connected),
        patch.object(run.urllib.request, "urlopen", side_effect=response),
        patch.object(run.subprocess, "Popen") as popen,
        patch.object(run.webbrowser, "open") as browser,
        redirect_stdout(output),
        redirect_stderr(output),
    ):
        assert run.main(["--backend-only"]) == 1

    popen.assert_not_called()
    browser.assert_not_called()
    assert "Port 8000 is occupied" in output.getvalue()
    assert "No existing process was stopped" in output.getvalue()


def test_backend_openapi_version_must_match_before_reuse():
    output = io.StringIO()

    def wrong_version(url, **_kwargs):
        if url.endswith("/openapi.json"):
            return Response(json.dumps({"info": {"title": "F1-RL Development Runtime", "version": "2.0.0"}}).encode())
        return open_response(url)

    with (
        patch.object(run.socket, "create_connection", side_effect=connected),
        patch.object(run.urllib.request, "urlopen", side_effect=wrong_version),
        patch.object(run.subprocess, "Popen") as popen,
        redirect_stdout(output),
        redirect_stderr(output),
    ):
        assert run.main(["--backend-only"]) == 1

    popen.assert_not_called()
    assert "OpenAPI title or version" in output.getvalue()


def test_foreign_frontend_port_is_reported_without_starting_or_stopping_processes():
    output = io.StringIO()
    with (
        patch.object(run.socket, "create_connection", side_effect=connected),
        patch.object(run.urllib.request, "urlopen", return_value=Response(b"<title>Other app</title>")),
        patch.object(run.subprocess, "Popen") as popen,
        patch.object(run.webbrowser, "open") as browser,
        redirect_stdout(output),
        redirect_stderr(output),
    ):
        assert run.main(["--frontend-only"]) == 1

    popen.assert_not_called()
    browser.assert_not_called()
    assert "Port 5173 is occupied" in output.getvalue()
    assert "No existing process was stopped" in output.getvalue()


def test_frontend_reuse_rejects_a_different_runtime_url():
    output = io.StringIO()

    def wrong_runtime(url, **_kwargs):
        if url.endswith("/src/lib/runtime.ts"):
            return Response(b'import.meta.env = {"VITE_RUNTIME_URL":"http://127.0.0.1:8001"};')
        return open_response(url)

    with (
        patch.object(run.socket, "create_connection", side_effect=connected),
        patch.object(run.urllib.request, "urlopen", side_effect=wrong_runtime),
        patch.object(run.subprocess, "Popen") as popen,
        patch.object(run.time, "sleep", side_effect=KeyboardInterrupt),
        redirect_stdout(output),
        redirect_stderr(output),
    ):
        assert run.main(["--frontend-only"]) == 1

    popen.assert_not_called()
    assert "configured Runtime URL" in output.getvalue()


def test_backend_only_uses_current_python_and_cleans_owned_process_on_ctrl_c():
    process = Process([None])
    output = io.StringIO()
    with (
        patch.object(run.socket, "create_connection", side_effect=refused),
        patch.object(run.urllib.request, "urlopen", side_effect=open_response),
        patch.object(run.subprocess, "Popen", return_value=process) as popen,
        patch.object(run, "terminate_process_tree") as terminate,
        patch.object(run.webbrowser, "open") as browser,
        patch.object(run.time, "sleep", side_effect=KeyboardInterrupt),
        redirect_stdout(output),
        redirect_stderr(output),
    ):
        assert run.main(["--backend-only"]) == 0

    command = popen.call_args.args[0]
    assert command == [sys.executable, "-m", "src.runtime"]
    assert popen.call_args.kwargs["cwd"] == run.ROOT
    terminate.assert_called_once_with(process, graceful=True)
    browser.assert_not_called()
    assert "Backend   READY  http://127.0.0.1:8000" in output.getvalue()


def test_frontend_only_starts_vite_on_default_port_without_backend_or_browser():
    process = Process([None])
    with (
        patch.object(run.socket, "create_connection", side_effect=refused),
        patch.object(run.urllib.request, "urlopen", side_effect=open_response),
        patch.object(run.subprocess, "Popen", return_value=process) as popen,
        patch.object(run, "terminate_process_tree"),
        patch.object(run.webbrowser, "open") as browser,
        patch.object(run.time, "sleep", side_effect=KeyboardInterrupt),
        redirect_stdout(io.StringIO()),
        redirect_stderr(io.StringIO()),
    ):
        assert run.main(["--frontend-only"]) == 0

    command = popen.call_args.args[0]
    assert "npm run dev" in " ".join(command)
    assert "--port 5173" in " ".join(command)
    assert "--strictPort" in " ".join(command)
    assert "--host" not in " ".join(command[4:])
    assert popen.call_args.kwargs["cwd"] == run.WEB_ROOT
    browser.assert_not_called()


def test_frontend_only_reports_missing_npm_without_starting_a_process():
    output = io.StringIO()
    with (
        patch.object(run.socket, "create_connection", side_effect=refused),
        patch.object(run.shutil, "which", return_value=None),
        patch.object(run.subprocess, "Popen") as popen,
        redirect_stdout(output),
        redirect_stderr(output),
    ):
        assert run.main(["--frontend-only"]) == 1

    popen.assert_not_called()
    assert "npm was not found" in output.getvalue()


def test_no_browser_keeps_compatible_services_reused():
    output = io.StringIO()
    with (
        patch.object(run.socket, "create_connection", side_effect=connected),
        patch.object(run.urllib.request, "urlopen", side_effect=open_response),
        patch.object(run.subprocess, "Popen") as popen,
        patch.object(run.webbrowser, "open") as browser,
        patch.object(run.time, "sleep", side_effect=KeyboardInterrupt),
        redirect_stdout(output),
        redirect_stderr(output),
    ):
        assert run.main(["--no-browser"]) == 0

    popen.assert_not_called()
    browser.assert_not_called()


def test_crashed_owned_backend_cleans_up_the_other_owned_service():
    backend = Process([None, 17])
    frontend = Process([None, None])
    processes = iter((backend, frontend))
    stopped = []
    output = io.StringIO()
    with (
        patch.object(run.socket, "create_connection", side_effect=refused),
        patch.object(run.urllib.request, "urlopen", side_effect=open_response),
        patch.object(run.subprocess, "Popen", side_effect=lambda *_args, **_kwargs: next(processes)),
        patch.object(run, "terminate_process_tree", side_effect=lambda process, **_kwargs: stopped.append(process)),
        patch.object(run.time, "sleep", side_effect=lambda _seconds: None),
        redirect_stdout(output),
        redirect_stderr(output),
    ):
        assert run.main(["--no-browser"]) == 1

    assert frontend in stopped
    assert "Backend exited unexpectedly" in output.getvalue()


def test_windows_cleanup_falls_back_to_taskkill_process_tree():
    process = Process([None])
    process.send_signal = Mock(side_effect=OSError("control event unavailable"))
    process.wait = Mock(side_effect=subprocess.TimeoutExpired("launcher", 2))
    with (
        patch.object(run, "IS_WINDOWS", True),
        patch.object(run.subprocess, "run") as run_command,
        patch.object(run.time, "sleep"),
    ):
        run_command.return_value.returncode = 0
        run.terminate_process_tree(process)

    args = run_command.call_args.args[0]
    assert args[:2] == ["taskkill", "/PID"]
    assert args[-2:] == ["/T", "/F"]


def test_windows_frontend_cleanup_kills_tree_without_batch_interrupt_prompt():
    process = Process([None])
    process.send_signal = Mock()
    with (
        patch.object(run, "IS_WINDOWS", True),
        patch.object(run.subprocess, "run") as run_command,
    ):
        run_command.return_value.returncode = 0
        run.terminate_process_tree(process, graceful=False)

    process.send_signal.assert_not_called()
    args = run_command.call_args.args[0]
    assert args == ["taskkill", "/PID", str(process.pid), "/T", "/F"]


def test_windows_cleanup_kills_descendants_after_parent_exits_gracefully():
    process = Process([None])
    process.send_signal = Mock()
    process.wait = Mock(side_effect=lambda **_kwargs: setattr(process, "returncode", 0))
    with (
        patch.object(run, "IS_WINDOWS", True),
        patch.object(run.subprocess, "run") as run_command,
    ):
        run_command.return_value.returncode = 0
        run.terminate_process_tree(process)

    process.send_signal.assert_called_once()
    process.wait.assert_called_once_with(timeout=3)
    args = run_command.call_args.args[0]
    assert args == ["taskkill", "/PID", str(process.pid), "/T", "/F"]


def test_windows_console_break_enters_launcher_cleanup():
    with (
        patch.object(run, "IS_WINDOWS", True),
        patch.object(run.signal, "signal") as register,
        patch.object(run.Launcher, "start", side_effect=KeyboardInterrupt),
        redirect_stdout(io.StringIO()),
    ):
        assert run.main(["--backend-only"]) == 0

    register.assert_called_once_with(run.signal.SIGBREAK, run._handle_console_break)


def test_child_output_does_not_crash_on_windows_console_encoding():
    console = NarrowConsole()
    with redirect_stdout(console):
        run._stream_output("FRONTEND", io.StringIO("ready ➜\n"))
    assert console.value == "[FRONTEND] ready ?\n"


def test_help_and_exclusive_modes_are_declared():
    with pytest.raises(SystemExit) as help_exit:
        run.main(["--help"])
    assert help_exit.value.code == 0

    with pytest.raises(SystemExit) as mode_exit:
        run.main(["--backend-only", "--frontend-only"])
    assert mode_exit.value.code == 2
