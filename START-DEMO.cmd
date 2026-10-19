@echo off
setlocal
cd /d "%~dp0"
if defined F1RL_PYTHON (
  "%F1RL_PYTHON%" -B scripts\run_inference.py %*
) else if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -B scripts\run_inference.py %*
) else (
  python -B scripts\run_inference.py %*
)
exit /b %ERRORLEVEL%
