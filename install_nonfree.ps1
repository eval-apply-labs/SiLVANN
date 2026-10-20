<#
What SiLVANN reads more with and runs without, into its own Python environment on Windows: requirements-nonfree.txt
(PyAV, for video files other than animated pictures), whose wheels carry binaries under copyleft licences (the LGPL,
the GPL: ATTRIBUTION.md). install.ps1 leaves it out; this adds it, once, from PyPI.

  powershell -ExecutionPolicy Bypass -File install_nonfree.ps1

  SILVANN_DIR   the installed SiLVANN    default the folder this file is in

Run it again after an update that changes the environment's Python. ASSUMED until a Windows PC has run it (NN-94).
This file is ASCII on purpose: Windows PowerShell 5.1 reads a script without a byte-order mark in the PC's code page.
#>
$ErrorActionPreference = "Continue"
$Dir = if ($env:SILVANN_DIR) { $env:SILVANN_DIR } else { $PSScriptRoot }
$py = Join-Path $Dir "venv\Scripts\python.exe"
$req = Join-Path $Dir "requirements-nonfree.txt"
if (-not (Test-Path $py)) {
    Write-Host "STOPPED: no SiLVANN environment in $Dir - install.ps1 first, or SILVANN_DIR set to where it installed" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $req)) { Write-Host "STOPPED: no requirements-nonfree.txt in $Dir" -ForegroundColor Red; exit 1 }
Write-Host "requirements-nonfree.txt into $Dir\venv" -ForegroundColor Cyan
& $py -m pip install --quiet -r $req
if ($LASTEXITCODE -ne 0) {
    Write-Host "STOPPED: pip could not install it (the lines above); SiLVANN runs without it - a video file other than an animated picture is refused" -ForegroundColor Red
    exit 1
}
Write-Host "done: a video file is read through PyAV from the next start of the server"
