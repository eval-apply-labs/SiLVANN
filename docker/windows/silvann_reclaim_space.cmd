@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0silvann.ps1" reclaim_space %*
pause
