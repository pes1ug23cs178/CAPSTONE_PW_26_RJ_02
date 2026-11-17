@echo off
rem Starts queue_runner.py with caches on C: (never D:). Launch detached via scripts/start_queue.ps1.
set HF_HOME=
set TORCH_HOME=
set PIP_CACHE_DIR=
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
cd /d "%~dp0.."
if not exist runs\_queue mkdir runs\_queue
"C:\Users\Student3\AppData\Local\Programs\Python\Python312\python.exe" queue_runner.py >> runs\_queue\runner.log 2>&1
