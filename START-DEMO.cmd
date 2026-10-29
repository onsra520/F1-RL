@echo off
setlocal
cd /d "%~dp0"
if defined F1RL_PYTHON (
  "%F1RL_PYTHON%" -B -m scripts.inference.run_policy %*
) else if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -B -m scripts.inference.run_policy %*
) else (
  python -B -m scripts.inference.run_policy %*
)
exit /b %ERRORLEVEL%
