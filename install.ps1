<#
SiLVANN's installer on Windows: the release, its Python environment and its libraries, in a folder of its own under the
current one, and a `silvann` command - install.sh's counterpart. The libraries are the release's own, prebuilt on
GitHub for every card the families list (silvann_windows_x64.zip, attached to the release): taken when they were built
from this folder's engine and scripts\build.py, else built here as scripts\build.ps1 builds them.

  powershell -ExecutionPolicy Bypass -File install.ps1

It installs nothing system-wide and asks for no administrator: a tool it needs and does not find is named, with the line
that installs it, and the installer stops. Run it again to update; models, user_sessions, the venv and the folder
`silvann data` named stay. What it needs:
  Python 3.10 to 3.14       python.org, or: winget install -e --id Python.Python.3.12
  Git                       winget install -e --id Git.Git
  a card's own driver       NVIDIA's for an NVIDIA card; AMD's or Intel's, whose OpenCL.dll the OpenCL families use
and, only to build the libraries here (scripts\build.ps1, or a release whose libraries do not match):
  Visual Studio's C++ tools and its "C++ Clang tools for Windows" (the Build Tools are enough)
  CUDA's toolkit (12 or 13), for an NVIDIA card: developer.nvidia.com/cuda-downloads - without it the CPU runs the models

  SILVANN_DIR      where it installs                    default the release it is in (a clone), else .\silvann
  SILVANN_SOURCE   the repository it fetches            default https://github.com/eval-apply-labs/SiLVANN
  SILVANN_REF      a branch of it                       default the repository's own
  SILVANN_NVCC     CUDA's compiler, when not found      default %CUDA_PATH%\bin\nvcc.exe
  SILVANN_SILICON_FAMILIES  the families to build, e.g. "x86_avx2 nvidia_cuda12"   default the CPU's and OpenCL's, and
                            CUDA's for an NVIDIA card of compute 7.5 or newer

  install.ps1               fetch (or update), the environment, the libraries, the command
  install.ps1 -FromSource   the same, the libraries built here (scripts\build.ps1 runs this)
  install.ps1 -Build        the build alone, the cards and compilers looked for again (`silvann build`, scripts\build.ps1 -Build)

scripts\requirements-nonfree.txt (PyAV, for video files) is scripts\install_nonfree.ps1's.
MEASURED 2026-10-19 on one PC (an RTX 5070 Ti, CUDA 13.4): the build from source; the prebuilt libraries ASSUMED until
a release's are taken (NN-94). This file is ASCII on purpose: Windows PowerShell 5.1 reads a script without a
byte-order mark in the PC's code page.
#>
param([switch]$FromSource, [switch]$Build, [switch]$Fetched)
# a native command's stderr is not an error here: every call checks its exit code
$ErrorActionPreference = "Continue"

function Say($m)  { Write-Host $m -ForegroundColor Cyan }
function Note($m) { Write-Host "   $m" }
function Fail($m) { Write-Host "STOPPED: $m" -ForegroundColor Red; exit 1 }

$Source = if ($env:SILVANN_SOURCE) { $env:SILVANN_SOURCE } else { "https://github.com/eval-apply-labs/SiLVANN" }
# where a release's prebuilt libraries are attached
$Releases = "https://github.com/eval-apply-labs/SiLVANN/releases"
# Python writes UTF-8 whatever the console's code page: build.py's marks and the runtime's messages are not ASCII
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# -- what the PC has ------------------------------------------------------------------------------------------------
function Find-Python {
    # the launcher first: `python` alone may be the Store's placeholder, which opens the Store instead of running
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $exe = & py -3 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $exe) { return "$exe".Trim() }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $exe = & python -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $exe) { return "$exe".Trim() }
    }
    return $null
}

function Find-VisualStudio {
    $where = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $where)) { return $null }
    $vs = & $where -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0 -or -not $vs) { return $null }
    return ($vs | Select-Object -First 1).Trim()
}

function Check-Machine {
    if (-not [Environment]::Is64BitOperatingSystem) { Fail "SiLVANN is built for 64-bit x86 Windows." }
    $script:Python = Find-Python
    $missing = @()
    if (-not $script:Python) {
        $missing += "Python 3.10 or newer: winget install -e --id Python.Python.3.12  (or python.org)"
    } else {
        & $script:Python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
        if ($LASTEXITCODE -ne 0) { Fail "Python 3.10 or newer is needed; $script:Python is older." }
    }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { $missing += "Git: winget install -e --id Git.Git" }
    Stop-If-Missing $missing
}

# what building the libraries needs beyond that
function Check-Toolchain {
    $missing = @()
    $script:VS = Find-VisualStudio
    $clang = if ($script:VS) { Join-Path $script:VS "VC\Tools\Llvm\x64\bin\clang-cl.exe" } else { "" }
    if (-not $script:VS -or -not (Test-Path $clang)) {
        $missing += ("Visual Studio's C++ tools with its Clang: winget install -e --id Microsoft.VisualStudio.2022.BuildTools " +
                     "--override `"--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended " +
                     "--add Microsoft.VisualStudio.Component.VC.Llvm.Clang --add Microsoft.VisualStudio.Component.VC.Llvm.ClangToolset`"" +
                     "  (or, in Visual Studio's installer: Desktop development with C++, and in it C++ Clang tools for Windows)")
    }
    Stop-If-Missing $missing
}

function Stop-If-Missing($missing) {
    if ($missing.Count -gt 0) {
        Write-Host "STOPPED: missing -" -ForegroundColor Red
        foreach ($m in $missing) { Write-Host "   $m" }
        Write-Host "   then open a new PowerShell and run this installer again."
        exit 1
    }
}

# the cards, and the compiler each one's family is built with; what is chosen goes to build.py as its variables
function Find-Cards {
    # the OpenCL families always, as install.sh builds them: they need nothing installed to be built (the headers carried,
    # OpenCL.dll's import library made from them), and at run time they take only the cards no other family does - an
    # NVIDIA card older than CUDA's (OpenCL 1.2, nvidia_opencl1), an Intel GPU (intel_opencl2 from Gen9, intel_opencl1_igpu
    # before), an AMD GPU (khronos_opencl2) - so a PC whose card CUDA runs is unchanged
    $script:Families = "x86_avx2 khronos_opencl2 intel_opencl2 nvidia_opencl1 intel_opencl1_igpu"
    $openclCard = $false
    $cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
    Note "the CPU: $cpu"
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        $cards = (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null) -join "; "
        # a Turing card without tensor cores, by its name - the list src/silicon_families/nvidia_tensor_cores.cuh keeps
        $bare = "GTX 1630|GTX 1650|GTX 1660|MX450|MX550|Quadro T1000|Quadro T2000| T400| T500| T550| T600| T1000| T1200"
        $rows = @(& nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>$null | ForEach-Object {
                      $f = "$_".Split(","); if ($f.Count -ge 2) { [pscustomobject]@{ Name = $f[0].Trim(); Cap = $f[1].Trim().Replace(".", "") } } } |
                  Where-Object { $_.Cap -match "^[0-9]+$" })
        if ($cards) {
            Note "NVIDIA: $cards"
            # CUDA's family takes compute 7.5 and newer (its targets, sm_75 ...) but a Turing card without tensor cores: an older
            # card - Pascal's GTX 10x0, compute 6.1 - is not built for (CUDA 13 builds no sm_61, and 12.9 stops at its tensor
            # cores), and a GTX 16xx - compute 7.5, no tensor cores - runs CUDA's tensor-core kernels slower than OpenCL 1.2's:
            # both run through OpenCL 1.2
            $isOld = { param($r) [int]$r.Cap -lt 75 -or ($r.Cap -eq "75" -and $r.Name -match $bare) }
            $new = @($rows | Where-Object { -not (& $isOld $_) } | ForEach-Object { $_.Cap } | Sort-Object -Unique)
            $old = @($rows | Where-Object { & $isOld $_ })
            if ($old.Count -gt 0) {
                $said = ($old | ForEach-Object { "$($_.Name) ($($_.Cap.Substring(0, $_.Cap.Length - 1)).$($_.Cap.Substring($_.Cap.Length - 1)))" }) -join ", "
                Note "   $said - not CUDA's family (compute 7.5 and newer, with tensor cores): runs through OpenCL 1.2"
                $openclCard = $true
            }
            if ($new.Count -gt 0 -or $old.Count -eq 0) {     # a card CUDA's family takes, or none whose compute was read
                $nvcc = $env:SILVANN_NVCC
                if (-not $nvcc -and $env:CUDA_PATH) { $nvcc = Join-Path $env:CUDA_PATH "bin\nvcc.exe" }
                if (-not $nvcc) { $c = Get-Command nvcc -ErrorAction SilentlyContinue; if ($c) { $nvcc = $c.Source } }
                if ($nvcc -and (Test-Path $nvcc)) {
                    $release = (& $nvcc --version | Select-String -Pattern "release [0-9.]+").Matches.Value
                    Note "   built with $nvcc ($release)$(if ($new.Count -gt 0) { ", for compute capability $($new -join ',')" })"
                    $env:SILVANN_NVCC = $nvcc
                    if ($new.Count -gt 0) { $env:SILVANN_CUDA_ARCHES = $new -join "," }
                    $script:Families = "$script:Families nvidia_cuda12"
                } else {
                    Note "   no nvcc: the card's family needs CUDA's compiler (12 or 13) - the CUDA toolkit from"
                    Note "   developer.nvidia.com/cuda-downloads (the driver is installed already), then 'silvann build'."
                    Note "   Until then the CPU runs the models."
                }
            }
        }
    }
    # the other GPUs - AMD's, Intel's - through OpenCL
    $others = @(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -and $_.AdapterCompatibility -notmatch "NVIDIA|Microsoft" })
    if ($others.Count -gt 0) {
        Note "other GPUs: $(($others | ForEach-Object { $_.Name }) -join '; ') - through their driver's OpenCL"
        $openclCard = $true
    }
    # an OpenCL family runs on OpenCL.dll, which a GPU's driver installs
    if ($openclCard -and -not (Test-Path (Join-Path $env:SystemRoot "System32\OpenCL.dll"))) {
        Note "   no OpenCL.dll: the GPU's own driver installs it - its vendor's current driver; until then the CPU runs the models"
    }
    $ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
    Note "memory: $ram GB of RAM"
    if ($env:SILVANN_SILICON_FAMILIES) {
        $script:Families = $env:SILVANN_SILICON_FAMILIES
        Note "the families SILVANN_SILICON_FAMILIES names: $script:Families"
    }
}

# -- the release ----------------------------------------------------------------------------------------------------
function Fetch {
    if (Test-Path (Join-Path $Dir ".git")) {
        Say "updating $Dir"
        & git -C $Dir pull --ff-only --quiet
        if ($LASTEXITCODE -ne 0) { Fail "git could not update $Dir (a change made in it by hand?)" }
        return
    }
    if (Test-Path (Join-Path $Dir "scripts\build.py")) {
        # a release unpacked here without git (a zip): built as it is, never mixed with one fetched over it
        Say "the release already in $Dir (no git: an update is a new copy, or a git clone)"
        return
    }
    # a folder of its own, or one this installer filled before: never files it did not write
    $others = Get-ChildItem -Force $Dir | Where-Object { @("models", "venv", "user_sessions") -notcontains $_.Name }
    if (-not (Test-Path (Join-Path $Dir "scripts\build.py")) -and $others) {
        Fail "$Dir already holds other files: install into a folder of its own (the default, .\silvann)"
    }
    Say "fetching SiLVANN into $Dir"
    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("silvann-" + [Guid]::NewGuid().ToString("N"))
    $clone = @("clone", "--quiet", "--depth", "1")
    if ($env:SILVANN_REF) { $clone += @("--branch", $env:SILVANN_REF) }
    & git @clone $Source $tmp
    if ($LASTEXITCODE -ne 0) { Fail "git could not fetch $Source" }
    # robocopy, not Move-Item: it merges into folders already there (models\) and crosses drives
    & robocopy $tmp $Dir /E /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { Fail "the fetched release could not be copied into $Dir" }
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

function Make-Environment {
    Say "the Python environment ($Dir\venv)"
    $py = Join-Path $Dir "venv\Scripts\python.exe"
    if (-not (Test-Path $py)) {
        & $script:Python -m venv (Join-Path $Dir "venv")
        if ($LASTEXITCODE -ne 0) { Fail "Python could not make its environment in $Dir\venv" }
    }
    & $py -m pip install --quiet --upgrade pip
    & $py -m pip install --quiet -r (Join-Path $Dir "scripts\requirements.txt")
    if ($LASTEXITCODE -ne 0) { Fail "pip could not install requirements.txt (the lines above)" }
}

# Windows 11's Smart App Control, on: it loads unsigned code only where Microsoft's reputation service knows it, and
# SiLVANN's libraries - the release's and those built here - are not signed yet
function Smart-App-Control-On {
    $policy = Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy" -ErrorAction SilentlyContinue
    return ($policy -and $policy.VerifiedAndReputablePolicyState -eq 1)
}
$SmartAppControl = ("Windows' Smart App Control refuses SiLVANN's libraries, which are not signed yet: turn it off " +
                    "(Windows Security > App & browser control > Smart App Control settings > Off; on most PCs it cannot " +
                    "be turned on again without resetting Windows), or run SiLVANN on Linux (docs\users.md)")

# the release's own libraries, when they were built from this folder's engine and scripts\build.py (git's hashes, which the
# zip's windows_libraries.txt carries): true when they are in lib\ and the evaluator loads them
function Get-Libraries {
    $m = Select-String -Path (Join-Path $Dir "README.md") -Pattern "Developer preview, (v[0-9]+\.[0-9]+\.[0-9]+)" `
                       -ErrorAction SilentlyContinue | Select-Object -First 1
    $want = @(& git -C $Dir rev-parse HEAD:engine HEAD:scripts/build.py 2>$null)
    if (-not $m -or $LASTEXITCODE -ne 0 -or $want.Count -ne 2) {
        Note "not a release's git checkout: the libraries are built here"
        return $false
    }
    $version = $m.Matches[0].Groups[1].Value
    Say "the libraries, prebuilt for $version"
    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("silvann-lib-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    $zip = Join-Path $tmp "silvann_windows_x64.zip"
    $ProgressPreference = "SilentlyContinue"   # Windows PowerShell's progress bar slows a download many times over
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing -Uri "$Releases/download/$version/silvann_windows_x64.zip" -OutFile $zip
        Expand-Archive -Path $zip -DestinationPath $tmp -Force
    } catch {
        Note "   none to fetch ($($_.Exception.Message)): the libraries are built here"
        Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
        return $false
    }
    $stamp = @{}
    Get-Content (Join-Path $tmp "lib\windows_libraries.txt") -ErrorAction SilentlyContinue | ForEach-Object {
        $k, $v = $_ -split " ", 2
        $stamp[$k] = "$v".Trim()
    }
    if ($stamp["engine"] -ne $want[0].Trim() -or $stamp["build.py"] -ne $want[1].Trim()) {
        Note "   $version's were built from another engine than this folder's: the libraries are built here"
        Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
        return $false
    }
    $lib = Join-Path $Dir "lib"
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $lib "silvann_engine_cpu*.pyd"), (Join-Path $lib "silicon_families\*.skipped")
    & robocopy (Join-Path $tmp "lib") $lib /E /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { Fail "the libraries could not be copied into $lib" }
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    # loaded once here, so a missing piece is said now and not at the first start
    $py = Join-Path $Dir "venv\Scripts\python.exe"
    $check = "import sys; sys.path.insert(0, r'$lib'); import silvann_engine_cpu as e; " +
             "print(', '.join('%s (%s)' % (f.get('silicon'), f.get('outcome')) if isinstance(f, dict) else str(f) for f in e.silicon_families()))"
    $out = (& $py -c $check 2>&1) -join " "
    if ($LASTEXITCODE -ne 0) {
        Note "   the evaluator would not load: $out"
        if (-not (Test-Path (Join-Path $env:SystemRoot "System32\msvcp140.dll"))) {
            Fail "Microsoft's Visual C++ runtime is missing: winget install -e --id Microsoft.VCRedist.2015+.x64 - then run this again"
        }
        if (Smart-App-Control-On) { Fail $SmartAppControl }  # a build here would be refused the same way
        Note "   so the libraries are built here"
        return $false
    }
    Note "   the families: $out"
    return $true
}

function Build-Engine {
    Say "the cards and the compilers"
    Find-Cards
    Say "building the engine and the families for this PC: $script:Families (a few minutes; everything in $Dir\build.log)"
    $env:SILVANN_SILICON_FAMILIES = $script:Families
    $py = Join-Path $Dir "venv\Scripts\python.exe"
    $log = Join-Path $Dir "build.log"
    $err = Join-Path $Dir "build.err"
    $p = Start-Process -FilePath $py -ArgumentList "`"$(Join-Path $Dir 'scripts\build.py')`"" -WorkingDirectory $Dir -NoNewWindow -Wait `
                       -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
    $shown = Get-Content -Encoding UTF8 $log | Where-Object { $_ -match '^==|->|skipped|failed|would not|does not|^ {6}\S' }
    foreach ($l in $shown) { Write-Host "   $l" }
    if ($p.ExitCode -ne 0) {
        Get-Content -Encoding UTF8 $err -Tail 30 | ForEach-Object { Write-Host "   $_" }
        Fail "the build failed - the lines above, and all of it in $log, $err and the .log files under $Dir\lib"
    }
}

# -- the command ----------------------------------------------------------------------------------------------------
function Command-Files {
    # `silvann`: a .cmd in the folder, the folder on the user's PATH
    $cmd = @'
@echo off
rem `silvann`, written by install.ps1 for the release in this folder.
setlocal
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"
set "PY=%HERE%\venv\Scripts\python.exe"
set "PYTHONUTF8=1"
rem the data - the models, the conversations and the rest: the folder `silvann data` named, kept in data_folder, else this one
if not defined SILVANN_DATA (
    if exist "%HERE%\data_folder" (set /p SILVANN_DATA=<"%HERE%\data_folder") else (set "SILVANN_DATA=%HERE%")
)
set "WHAT=%~1"
if "%WHAT%"=="" set "WHAT=help"
if /i "%WHAT%"=="list"     goto list
if /i "%WHAT%"=="download" goto download
if /i "%WHAT%"=="check"    goto check
if /i "%WHAT%"=="token"    goto token
if /i "%WHAT%"=="users"    goto users
if /i "%WHAT%"=="serve"    goto serve
if /i "%WHAT%"=="ui"       goto ui
if /i "%WHAT%"=="registry" goto registry
if /i "%WHAT%"=="data"     goto data
if /i "%WHAT%"=="build"    goto build
if /i "%WHAT%"=="update"   goto update
if /i "%WHAT%"=="where"    goto where
if /i "%WHAT%"=="help"     goto help
echo silvann: no command "%WHAT%" - these are:
goto help
:list
"%PY%" "%HERE%\silvann_core.py" list-models %2 %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%
:download
"%PY%" "%HERE%\silvann_core.py" download-model %2 %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%
:check
if "%~2"=="" ("%PY%" "%HERE%\silvann_core.py" check-versions & exit /b %errorlevel%)
"%PY%" "%HERE%\silvann_core.py" check-model %2 %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%
:token
"%PY%" "%HERE%\silvann_core.py" hf-token %2 %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%
:users
"%PY%" "%HERE%\silvann_core.py" users %2 %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%
:serve
if "%~2"=="" (echo silvann serve ^<model^> [config] [--port 8765]   - a model in %SILVANN_DATA%\models & exit /b 1)
set "MODEL=%~2"
if not exist "%MODEL%\" set "MODEL=%SILVANN_DATA%\models\%~2"
cd /d "%HERE%" && "%PY%" start_silvann_server.py --model "%MODEL%" %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%
:ui
cd /d "%HERE%" && "%PY%" silvann_ui.py --port 8090 %2 %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%
:registry
cd /d "%HERE%" && "%PY%" silvann_registry.py %2 %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%
:data
if "%~2"=="" (echo the models, the conversations and the rest are kept in %SILVANN_DATA% & exit /b 0)
if not exist "%~2\models" mkdir "%~2\models"
if not exist "%~2\user_sessions" mkdir "%~2\user_sessions"
for %%I in ("%~2") do set "TO=%%~fI"
if /i "%TO%"=="%HERE%" (del "%HERE%\data_folder" 2>nul) else (>"%HERE%\data_folder" echo %TO%)
echo the models, the conversations and the rest are kept in %TO% from now on - what was in %SILVANN_DATA% stays there
exit /b 0
:build
set "SILVANN_DIR=%HERE%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%\scripts\build.ps1" -Build
exit /b %errorlevel%
:update
set "SILVANN_DIR=%HERE%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%\install.ps1"
exit /b %errorlevel%
:where
echo %HERE%
exit /b 0
:help
echo silvann list                        the models this release can fetch, and which are here
echo silvann download ^<model^>            a model into %%SILVANN_DATA%%\models - when it is there, what changed in its pack
echo silvann check [model]               update or done, each model here - with one, the files its update fetches
echo silvann token [--set ^| --forget]    the Hugging Face token the downloads use (faster than none)
echo silvann users [add^|passwd^|remove^|adopt] ^<name^>   the users: each a sign-in, a folder and their own conversations
echo silvann serve ^<model^> [config]      the server, on port 8765 (--port to change it)
echo silvann ui                          the chat in a browser, on port 8090, talking to the server
echo silvann_cli                         the chat in this terminal (--notools, --nothink), talking to the server
echo silvann registry                    the registry of a network's servers, on port 8770 (--port), for the subagents
echo silvann data [folder]               where the models, the conversations and the rest are kept - another folder
echo silvann build                       build the libraries here, from source - after installing CUDA, say
echo silvann update                      the newest release and its libraries - stop the server first
echo silvann where                       the folder it is installed in
echo silvann help                        this list
exit /b 0
'@
    # CRLF: cmd looks a `goto`'s label up wrongly in a file whose lines end in LF alone
    Set-Content -Path (Join-Path $Dir "silvann.cmd") -Value ($cmd -replace "`r?`n", "`r`n") -Encoding ASCII -NoNewline
    $cli = "@echo off`r`nrem ``silvann_cli``, written by install.ps1: the chat in this terminal (silvann_cli.py).`r`n" +
           "set `"PYTHONUTF8=1`"`r`n`"%~dp0venv\Scripts\python.exe`" `"%~dp0silvann_cli.py`" %*`r`n"
    Set-Content -Path (Join-Path $Dir "silvann_cli.cmd") -Value $cli -Encoding ASCII -NoNewline
    $path = [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not $path) { $path = "" }
    if (($path -split ";") -notcontains $Dir) {
        [Environment]::SetEnvironmentVariable("Path", ($path.TrimEnd(";") + ";" + $Dir).TrimStart(";"), "User")
        Note "$Dir is on your PATH from the next PowerShell you open"
    }
}

# -- the order -------------------------------------------------------------------------------------------------------
Check-Machine
# run from a release's folder (a clone), that folder; downloaded alone, a folder of its own under the current one
$Dir = if ($env:SILVANN_DIR) { $env:SILVANN_DIR } `
       elseif ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot "scripts\build.py"))) { $PSScriptRoot } `
       else { Join-Path (Get-Location) "silvann" }
New-Item -ItemType Directory -Force -Path $Dir | Out-Null
$Dir = (Resolve-Path $Dir).Path
$env:SILVANN_DIR = $Dir
if (-not $Build -and -not $Fetched) {
    Fetch
    # the rest by the installer just fetched, which an update makes newer than this one
    $rest = @("-Fetched")
    if ($FromSource) { $rest += "-FromSource" }
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Dir "install.ps1") @rest
    exit $LASTEXITCODE
}
if (-not $Build) { Make-Environment }
$prebuilt = $false
if (-not $Build -and -not $FromSource) { $prebuilt = @(Get-Libraries)[-1] -eq $true }
if (-not $prebuilt) {
    Check-Toolchain
    Build-Engine
}
Command-Files
if (-not $Build) {
    # the models here: each one's program for this release, from its pack (a boot fetches it otherwise)
    $data = if ($env:SILVANN_DATA) { $env:SILVANN_DATA } `
            elseif (Test-Path (Join-Path $Dir "data_folder")) { (Get-Content (Join-Path $Dir "data_folder") -Raw).Trim() } `
            else { $Dir }
    if (Test-Path (Join-Path $data "models")) {
        Say "The models here: each one's program for this release"
        $env:SILVANN_DATA = $data
        & (Join-Path $Dir "venv\Scripts\python.exe") (Join-Path $Dir "silvann_core.py") update-programs
        if ($LASTEXITCODE -ne 0) { Note "not checked now - each model's boot checks its program again" }
    }
}
Say "done: SiLVANN is in $Dir, and 'silvann' with it"
if (-not $Build) {
    Write-Host @"

   silvann list                                       the models, their sizes, and which are here
   silvann download silvann_tq_D8E4_qwen3.6_35b_a3b   one of them
   silvann serve silvann_tq_D8E4_qwen3.6_35b_a3b      the server
   silvann ui                                         the chat, at http://localhost:8090
   silvann_cli                                        or the chat in this terminal
   silvann help                                       every command
   scripts\install_nonfree.ps1                        once, to read video files other than animated pictures (PyAV)
"@
}
