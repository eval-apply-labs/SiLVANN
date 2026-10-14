@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0silvann.ps1" import %*
pause
