@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0silvann.ps1" open_data %*
if errorlevel 1 pause
