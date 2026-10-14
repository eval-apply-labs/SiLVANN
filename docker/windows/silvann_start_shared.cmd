@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0silvann.ps1" start_shared %*
if errorlevel 1 pause
