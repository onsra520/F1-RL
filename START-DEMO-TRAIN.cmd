@echo off
setlocal
cd /d "%~dp0"
title F1-RL - DEMO PPO TRAIN
echo.
echo ========== F1-RL DEMO TRAIN ==========
echo Config: config.yml ^| CPU only ^| local TRAIN-only
echo.

if defined F1RL_PYTHON goto custompython
if exist ".venv\Scripts\python.exe" goto localpython
where py >nul 2>&1
if not errorlevel 1 goto pylauncher
goto pathpython

:custompython
"%F1RL_PYTHON%" -B -m scripts.training.launch_demo
goto result

:localpython
".venv\Scripts\python.exe" -B -m scripts.training.launch_demo
goto result

:pylauncher
py -3.12 --version >nul 2>&1
if errorlevel 1 goto pathpython
py -3.12 -B -m scripts.training.launch_demo
goto result

:pathpython
python -B -m scripts.training.launch_demo
goto result

:result
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo Run command exited successfully. Check evidence for result.
) else (
  echo Run exited with code %RC%. Check console and evidence folder.
)
pause
exit /b %RC%
