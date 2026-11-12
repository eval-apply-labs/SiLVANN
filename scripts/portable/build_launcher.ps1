# SiLVANN.exe, the portable app's launcher (SiLVANN.c), by Visual Studio's C compiler, its environment from vcvars64.
# build_portable.ps1 runs it when it is handed no launcher; the release's job runs it beside the families, so the launcher
# is signed with the libraries (.github/workflows/windows-libraries.yml, `families` and `sign`).
#
#   pwsh scripts\portable\build_launcher.ps1 -Exe lib\SiLVANN.exe
param(
    [Parameter(Mandatory = $true)][string]$Exe
)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "no Visual Studio C compiler (vswhere found none)" }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
$out = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Exe)
New-Item -ItemType Directory -Force (Split-Path -Parent $out) | Out-Null
$src = Join-Path $Here "SiLVANN.c"
cmd /c "`"$vcvars`" >nul && cl /nologo /O2 /W3 `"$src`" /Fe:`"$out`" /Fo:`"$env:TEMP\\`" /link /SUBSYSTEM:WINDOWS user32.lib"
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $out)) { throw "SiLVANN.exe was not built" }
"built $out"
