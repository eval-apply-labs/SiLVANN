# The portable app's folder (NN-162): SiLVANN.exe, Python's embeddable build with the runtime's packages and the C++
# runtime, the evaluator and the silicon families, and the release's tree — zipped as silvann_portable_windows_x64.zip. Run by the release's
# Windows job (.github/workflows/windows-libraries.yml, `portable`) from the release's checkout, after the libraries are
# built; on a PC, from a checkout with lib\ built (scripts\build.ps1) and Visual Studio's C compiler.
#
#   pwsh scripts\portable\build_portable.ps1 -Lib lib -Out portable
#
# The release's job hands it a SiLVANN.exe built and signed with the libraries (`-Launcher`); without one it builds its own.
# Python's embeddable build has no pip: the packages are installed for it with this machine's Python of the same minor
# version (`pip install --target`), so their compiled parts are its ABI's. Its `python312._pth` names where it looks — the
# folder's root (the entry points), the runtime and the packages — and `import site`.
param(
    [string]$Lib = "lib",
    [string]$Out = "portable",
    [string]$PythonVersion = "3.12.10",
    [string]$Launcher = ""
)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Tree = Resolve-Path (Join-Path $Here "..\..")         # the release's tree: scripts\portable\ is two below it
$Minor = ($PythonVersion.Split(".")[0..1] -join "")    # 312
$Root = Join-Path $Out "SiLVANN"
if (Test-Path $Root) { Remove-Item -Recurse -Force $Root }
New-Item -ItemType Directory -Force $Root | Out-Null

# ① the release's tree, without what only a repository has
$skip = @(".git", ".github", "portable", "built", "lib", "venv", "models", "user_sessions", "engine", "docker")   # the engine's source and the image's: neither runs here
Get-ChildItem $Tree -Force | Where-Object { $skip -notcontains $_.Name } | ForEach-Object {
    Copy-Item $_.FullName -Destination $Root -Recurse -Force
}

# ② the libraries: the evaluator for this Python, the families, their notices
New-Item -ItemType Directory -Force (Join-Path $Root "lib\silicon_families") | Out-Null
$pyd = Get-ChildItem $Lib -Recurse -Filter "silvann_engine_cpu.cp$Minor-*.pyd" | Select-Object -First 1
if (-not $pyd) { throw "no evaluator for Python $PythonVersion (silvann_engine_cpu.cp$Minor-*.pyd) under $Lib" }
Copy-Item $pyd.FullName (Join-Path $Root "lib")
Get-ChildItem $Lib -Recurse -Filter "libsilvann_*.dll" | ForEach-Object { Copy-Item $_.FullName (Join-Path $Root "lib\silicon_families") }
$notices = Get-ChildItem $Lib -Recurse -Directory -Filter "notices" | Select-Object -First 1
if ($notices) { Copy-Item $notices.FullName (Join-Path $Root "lib") -Recurse -Force }

# ③ Python's embeddable build, and where it looks
$zip = Join-Path $env:TEMP "python-$PythonVersion-embed-amd64.zip"
curl.exe -fsSL -o $zip "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
if ($LASTEXITCODE -ne 0) { throw "Python's embeddable build $PythonVersion could not be fetched" }
$Py = Join-Path $Root "python"
Expand-Archive -Path $zip -DestinationPath $Py -Force
@("python$Minor.zip", ".", "Lib\site-packages", "..\runtime", "..", "import site") |
    Out-File -Encoding ascii (Join-Path $Py "python$Minor._pth")

# ④ the runtime's packages, for that Python, with this machine's Python of its minor version
$have = (& python -c "import sys; print('%d%d' % sys.version_info[:2])")
if ($have -ne $Minor) { throw "this machine's python is $have, the embeddable build's ${Minor}: their packages' ABIs differ" }
& python -m pip install --quiet --only-binary=:all: --target (Join-Path $Py "Lib\site-packages") -r (Join-Path $Tree "scripts\requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "pip could not install requirements.txt for the embeddable build" }

# ⑤ from Visual Studio: the C++ runtime, and SiLVANN.exe by its C compiler (build_launcher.ps1) unless one is handed in
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "no Visual Studio C compiler (vswhere found none)" }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"

# The evaluator and the families link the C++ runtime as a library (/MD): msvcp140.dll and vcruntime140*.dll. A PC
# without Microsoft's Visual C++ Redistributable has none of them, and the embeddable build carries vcruntime140 alone,
# so the folder carries the set from the toolset that built them, one version together. Microsoft lets an app carry
# them in its own folder; they go beside pythonw.exe, the first folder Windows looks in for a process's libraries.
# (`set` after vcvars64 has run: an `echo %VCToolsRedistDir%` on the same line is expanded when cmd reads the line, before)
$redist = ((cmd /c "`"$vcvars`" >nul && set VCToolsRedistDir" | Where-Object { $_ -like "VCToolsRedistDir=*" } |
    Select-Object -Last 1) -replace "^VCToolsRedistDir=", "").Trim()
$crt = Get-ChildItem (Join-Path $redist "x64") -Directory -Filter "Microsoft.VC*.CRT" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $crt) { throw "no C++ runtime among Visual Studio's redistributables ($redist x64\Microsoft.VC*.CRT)" }
$runtime = Get-ChildItem $crt.FullName -Filter "*.dll" | Where-Object { $_.Name -match '^(msvcp140|vcruntime140)' }
if (-not ($runtime | Where-Object { $_.Name -eq "msvcp140.dll" })) { throw "no msvcp140.dll in $($crt.FullName)" }
$runtime | ForEach-Object { Copy-Item $_.FullName $Py -Force }
"the C++ runtime from $($crt.FullName): $(($runtime | ForEach-Object { $_.Name }) -join ', ')"

$exe = Join-Path (Resolve-Path $Root) "SiLVANN.exe"
if ($Launcher) {
    Copy-Item $Launcher $exe
} else {
    & (Join-Path $Here "build_launcher.ps1") -Exe $exe
}

# ⑥ the zip
$archive = Join-Path $Out "silvann_portable_windows_x64.zip"
if (Test-Path $archive) { Remove-Item -Force $archive }
Compress-Archive -Path $Root -DestinationPath $archive
"built $archive ($([math]::Round((Get-Item $archive).Length / 1MB)) MB)"
