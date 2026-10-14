<#
SiLVANN's Docker image as an app on Windows. Each action has a .cmd beside this file to double-click.

  silvann.ps1 install         checks WSL2, Docker Desktop (or Podman) and the NVIDIA driver; offers WSL2 more memory;
                              pulls the image, makes the volume, checks a container sees the card, adds shortcuts.
                              Run it again to update: the new image over the same volume.
  silvann.ps1 update          the newest image of the same line and its scripts, the volume (models, conversations)
                              kept; a running app restarted on it
  silvann.ps1 start           the app on http://localhost:8080, this PC only
  silvann.ps1 start_shared    the same on every network interface: the LAN reaches the UI
  silvann.ps1 stop            stopped, the conversations saved first
  silvann.ps1 delete          the container and the image; the volume (models, conversations) after a second yes
  silvann.ps1 export <dir>    the image and the volume into <dir>, the models included: a portable copy
  silvann.ps1 import <dir>    both back, on this or another PC - refused where SiLVANN is installed already:
                              export it or delete it first
  silvann.ps1 install_nonfree requirements-nonfree.txt (PyAV, for video files) from PyPI into the volume: a
                              container started for it, then stopped; a running app restarted. Again after an
                              update that changes the image's Python

  SILVANN_IMAGE   the image        default ghcr.io/eval-apply-labs/silvann:main, the nightly built from main.
                                   From v1.0 a major version has its tag (:1), which its minor and patch releases
                                   update in place
  SILVANN_PORT    the UI's port    default 8080
  SILVANN_ENGINE  docker | podman  default whichever is installed, docker first
  HF_TOKEN        a Hugging Face token, handed to the container at start: its client downloads faster signed in

This file is ASCII on purpose: Windows PowerShell 5.1 reads a script without a byte-order mark in the PC's code page.
#>
param(
    [Parameter(Position = 0)][string]$Action = "help",
    [Parameter(Position = 1)][string]$Dir = ""
)
# a native command's stderr is not an error here: every call checks its exit code
$ErrorActionPreference = "Continue"

$DefaultImage = "ghcr.io/eval-apply-labs/silvann:main"
$Image = if ($env:SILVANN_IMAGE) { $env:SILVANN_IMAGE } else { $DefaultImage }
$Port = if ($env:SILVANN_PORT) { [int]$env:SILVANN_PORT } else { 8080 }
$Name = "silvann"
$Volume = "silvann-data"
$StopSeconds = 300          # the UI saves every conversation before it stops; a long one takes a while to write
$AppDir = Join-Path $env:LOCALAPPDATA "SiLVANN"
$Actions = @("install", "update", "start", "start_shared", "stop", "delete", "export", "import", "install_nonfree")

function Say($m)  { Write-Host $m -ForegroundColor Cyan }
function Note($m) { Write-Host "   $m" }
function Fail($m) { Write-Host "STOPPED: $m" -ForegroundColor Red; exit 1 }
function YesTo($q) { (Read-Host "$q [y/N]") -match '^[Yy]' }

$Engine = $env:SILVANN_ENGINE
if (-not $Engine) {
    if (Get-Command docker -ErrorAction SilentlyContinue) { $Engine = "docker" }
    elseif (Get-Command podman -ErrorAction SilentlyContinue) { $Engine = "podman" }
}

# the engine, its output shown on the console (not returned with the answer); $true when it succeeded
function E { & $Engine @args | Out-Host; return ($LASTEXITCODE -eq 0) }
# the engine, its output hidden
function Q { & $Engine @args 2>$null | Out-Null; return ($LASTEXITCODE -eq 0) }

function Need-Engine {
    if (-not $Engine) {
        Fail ("neither Docker Desktop nor Podman is installed. Docker Desktop, with its WSL 2 backend:`n" +
              "   https://docs.docker.com/desktop/setup/install/windows-install/  (or: winget install -e --id Docker.DockerDesktop)`n" +
              "   then run this again.")
    }
    if (-not (Q info)) { Fail "$Engine is installed and not running: start Docker Desktop (or podman machine start), wait for it, then run this again" }
}

function Has-Nvidia {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return $false }
    & nvidia-smi -L 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Gpu-Flags {
    if (-not (Has-Nvidia)) { return @() }
    if ($Engine -eq "podman") { return @("--device", "nvidia.com/gpu=all") }
    return @("--gpus", "all")
}

function Is-Running {
    $s = & $Engine inspect -f "{{.State.Running}}" $Name 2>$null
    return ($LASTEXITCODE -eq 0 -and "$s".Trim() -eq "true")
}

function Wait-UI {
    $url = "http://127.0.0.1:$Port/"
    for ($i = 0; $i -lt 120; $i++) {
        try {
            Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 | Out-Null
            return
        } catch { }
        if (-not (Is-Running)) {
            & $Engine logs --tail 30 $Name
            Fail "the container stopped: its last lines are above"
        }
        Start-Sleep -Seconds 1
    }
    Fail "the UI did not answer on $url in 2 minutes: '$Engine logs $Name' says why"
}

function Open-UI {
    Say "SiLVANN: http://localhost:$Port"
    Start-Process "http://localhost:$Port"
}

# WSL2 gives its Linux half the PC's memory unless .wslconfig says otherwise; a model with its experts in RAM needs more
function Offer-Memory {
    $total = [math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
    $cfg = Join-Path $env:USERPROFILE ".wslconfig"
    $lines = @()
    if (Test-Path $cfg) { $lines = @(Get-Content $cfg) }
    $current = $lines | Where-Object { $_ -match '^\s*memory\s*=' } | Select-Object -First 1
    $want = [math]::Max([math]::Floor($total / 2), $total - 6)
    if ($current) { Note "WSL2's memory: $($current.Trim()) of this PC's $total GB ($cfg)" }
    else { Note "WSL2's memory: the default, half of this PC's $total GB" }
    if ($current -or $want -le [math]::Floor($total / 2)) { return }
    if (-not (YesTo "Give WSL2 $want GB of the $total GB (a model with its experts in RAM needs it)?")) { return }
    $out = @(); $inWsl2 = $false; $done = $false
    foreach ($l in $lines) {
        if ($l -match '^\s*\[(.+)\]\s*$') {
            if ($inWsl2 -and -not $done) { $out += "memory=${want}GB"; $done = $true }
            $inWsl2 = ($Matches[1] -eq "wsl2")
        }
        $out += $l
    }
    if ($inWsl2 -and -not $done) { $out += "memory=${want}GB"; $done = $true }
    if (-not $done) { $out += "[wsl2]"; $out += "memory=${want}GB" }
    Set-Content -Path $cfg -Value $out -Encoding ASCII
    Note "written to $cfg. It applies once WSL restarts: quit Docker Desktop, run 'wsl --shutdown', start Docker Desktop."
}

function Add-Shortcuts {
    New-Item -ItemType Directory -Force -Path $AppDir | Out-Null
    $here = $PSCommandPath
    if ((Split-Path $here -Parent) -ne $AppDir) { Copy-Item $here (Join-Path $AppDir "silvann.ps1") -Force }
    foreach ($a in $Actions) {
        $pause = if ($a -in @("start", "start_shared", "stop")) { "if errorlevel 1 pause" } else { "pause" }
        Set-Content -Path (Join-Path $AppDir "silvann_$a.cmd") -Encoding ASCII -Value @(
            "@echo off",
            "powershell -NoProfile -ExecutionPolicy Bypass -File `"%~dp0silvann.ps1`" $a %*",
            $pause)
    }
    $shell = New-Object -ComObject WScript.Shell
    $desktop = [Environment]::GetFolderPath("Desktop")
    foreach ($s in @(@("SiLVANN", "start"), @("SiLVANN - stop", "stop"))) {
        $lnk = $shell.CreateShortcut((Join-Path $desktop "$($s[0]).lnk"))
        $lnk.TargetPath = Join-Path $AppDir "silvann_$($s[1]).cmd"
        $lnk.WorkingDirectory = $AppDir
        $lnk.Save()
    }
    Note "the scripts are in $AppDir; on the desktop: 'SiLVANN' starts it, 'SiLVANN - stop' stops it"
}

function Do-Install {
    Say "WSL 2"
    & wsl.exe --status 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { Fail "WSL 2 is not installed: 'wsl --install' in a PowerShell run as administrator, restart the PC, then run this again" }
    Note "installed"
    Say "the container engine: $(if ($Engine) { $Engine } else { 'none' })"
    Need-Engine
    if ($Engine -eq "docker") {
        Note "Docker Desktop keeps its images and the volume in one disk file. To keep the models on an NVMe drive:"
        Note "Docker Desktop -> Settings -> Resources -> Advanced -> Disk image location, before the first download."
    }
    Say "the card"
    if (Has-Nvidia) {
        & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | ForEach-Object { Note "NVIDIA: $_" }
    } else {
        Note "no NVIDIA card or driver: the CPU runs the models. NVIDIA's own Windows driver is the one a container uses;"
        Note "nothing is installed inside Linux for it."
    }
    Say "memory"
    Offer-Memory
    Say "the image: $Image"
    Pull-Image
    if (Is-Running) { Note "the app running now keeps the image it started with: silvann_update restarts it on this one" }
    if (-not (Q volume inspect $Volume)) { if (-not (Q volume create $Volume)) { Fail "the volume $Volume could not be made" } }
    Note "the volume $Volume holds the models and the conversations"
    if (Has-Nvidia) {
        Say "a container sees the card:"
        $gpu = Gpu-Flags
        if (-not (E run --rm @gpu --entrypoint nvidia-smi $Image -L)) {
            Fail "a container does not see the card: update NVIDIA's driver and Docker Desktop, then run this again"
        }
    }
    Say "shortcuts"
    Add-Shortcuts
    Say "done: start SiLVANN from the desktop, then choose and download a model in the UI's Options"
}

function Pull-Image {
    if (-not (E pull $Image)) {
        if (-not (Q image inspect $Image)) { Fail "the image $Image could not be pulled" }
        Note "not pulled: the image built on this PC is used"
    }
}

# the scripts the image carries, copied over the ones in $AppDir: an update brings its own installer
function Copy-Scripts {
    New-Item -ItemType Directory -Force -Path $AppDir | Out-Null
    Q rm -f silvann-scripts | Out-Null
    if ((Q create --name silvann-scripts $Image) -and (Q cp "silvann-scripts:/opt/silvann/docker/windows/." $AppDir)) {
        Note "the scripts in $AppDir are this image's"
    } else { Note "the scripts in $AppDir were left as they were" }
    Q rm -f silvann-scripts | Out-Null
}

# a running app stopped and started again, on the interfaces it had
function Restart-App {
    $bind = & $Engine inspect -f "{{range `$p, `$b := .HostConfig.PortBindings}}{{range `$b}}{{.HostIp}}{{end}}{{end}}" $Name 2>$null
    Do-Stop
    if ("$bind".Trim().StartsWith("127.0.0.1")) { Do-Start "127.0.0.1:" } else { Do-Start "" }
}

function Do-Update {
    Need-Engine
    Say "the newest $Image - the volume $Volume, the models and the conversations, stays as it is"
    Pull-Image
    Copy-Scripts
    # the app restarted when it runs on another image than the one the tag names now
    if (Is-Running) {
        $running = & $Engine inspect -f "{{.Image}}" $Name 2>$null
        $tagged = & $Engine image inspect -f "{{.Id}}" $Image 2>$null
        if ("$running".Trim() -ne "$tagged".Trim()) { Restart-App } else { Note "the app runs on this image already" }
    }
    Say "updated"
}

# what the image leaves out (requirements-nonfree.txt: its wheels carry LGPL and GPL binaries), fetched from PyPI by a
# container started for it on the volume, into /data/python/<the image's Python version> - which the image's
# environment reads when it starts (silvann-data.pth)
function Do-InstallNonfree {
    Need-Engine
    if (-not (Q image inspect $Image)) { Fail "there is no image $Image here: run install first" }
    Say "requirements-nonfree.txt (PyAV, for video files) from PyPI into the volume $Volume"
    $py = "import subprocess, sys; sys.exit(subprocess.call([sys.executable, '-m', 'pip', 'install', '--quiet', '--upgrade', '--target', '/data/python/%d.%d' % sys.version_info[:2], '-r', '/opt/silvann/requirements-nonfree.txt']))"
    if (-not (E run --rm -v "${Volume}:/data" --entrypoint /opt/silvann/venv/bin/python $Image -c $py)) {
        Fail "not installed: the lines above say why"
    }
    if (Is-Running) { Say "the app restarted, to read it"; Restart-App }
    Say "installed: a video file (MP4 and the like) is read now"
}

function Do-Start([string]$Bind) {
    Need-Engine
    if (Is-Running) { Open-UI; return }
    Q rm -f $Name | Out-Null
    $gpu = Gpu-Flags
    $token = @(); if ($env:HF_TOKEN) { $token = @("-e", "HF_TOKEN") }
    if (-not (E run -d --name $Name @gpu @token -p "$Bind${Port}:8080" -v "${Volume}:/data" --stop-timeout $StopSeconds $Image)) {
        Fail "the container did not start: the lines above say why"
    }
    Wait-UI
    if (-not $Bind) {
        Note "shared: every machine that reaches this PC on port $Port reaches the UI, and the UI has no login -"
        Note "a conversation's password is its only lock. Windows' firewall may ask whether to allow Docker."
        Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' -and $_.InterfaceAlias -notmatch 'vEthernet|WSL|Loopback' } |
            ForEach-Object { Note "on the LAN: http://$($_.IPAddress):$Port" }
    }
    Open-UI
}

function Do-Stop {
    Need-Engine
    if (Is-Running) {
        Say "stopping SiLVANN: the conversations are saved first"
        Q stop -t $StopSeconds $Name | Out-Null
    }
    Q rm -f $Name | Out-Null
    Say "stopped"
}

function Do-Delete {
    Do-Stop
    if (Q rmi $Image) { Say "the image removed" }
    if (Q volume inspect $Volume) {
        if (YesTo "Delete the volume $Volume too, with every model and conversation in it?") {
            if (Q volume rm $Volume) { Say "the volume removed" }
        } else { Note "the volume $Volume stays: install uses it again" }
    }
    foreach ($s in @("SiLVANN", "SiLVANN - stop")) {
        Remove-Item (Join-Path ([Environment]::GetFolderPath("Desktop")) "$s.lnk") -ErrorAction SilentlyContinue
    }
}

function Ask-Dir([string]$What) {
    if ($Dir) { return $Dir }
    return (Read-Host "the folder $What (another drive, say E:\SiLVANN)")
}

function Do-Export {
    $d = Ask-Dir "to export into"
    if (-not $d) { Fail "no folder given" }
    Need-Engine
    if (-not (Q volume inspect $Volume)) { Fail "there is no volume $Volume to export" }
    New-Item -ItemType Directory -Force -Path $d | Out-Null
    $d = (Resolve-Path $d).Path
    Do-Stop
    Say "the image -> $d\silvann-image.tar"
    if (-not (E save -o (Join-Path $d "silvann-image.tar") $Image)) { Fail "the image was not saved" }
    Say "the volume, the models included -> $d\silvann-data.tar (a large model takes a while)"
    if (-not (E run --rm --user 0:0 -v "${Volume}:/data:ro" -v "${d}:/out" --entrypoint tar $Image -cf /out/silvann-data.tar -C /data .)) {
        Fail "the volume was not exported"
    }
    Set-Content -Path (Join-Path $d "silvann-image.name") -Value $Image -Encoding ASCII
    Say "exported to ${d}: on the other PC, install Docker Desktop, then 'silvann.ps1 import $d'"
}

function Do-Import {
    $d = Ask-Dir "export wrote"
    if (-not $d) { Fail "no folder given" }
    $d = (Resolve-Path $d).Path
    if (-not ((Test-Path (Join-Path $d "silvann-image.tar")) -and (Test-Path (Join-Path $d "silvann-data.tar")))) {
        Fail "$d has no silvann-image.tar and silvann-data.tar"
    }
    $nameFile = Join-Path $d "silvann-image.name"
    if (Test-Path $nameFile) { $script:Image = (Get-Content $nameFile -TotalCount 1).Trim() }
    Need-Engine
    # an installation already here is never overwritten: its image, its container or its volume (its models and
    # conversations) - kept with export, or removed with delete, first; nothing is stopped or loaded before this
    $here = @()
    if (Q image inspect $Image) { $here += "the image $Image" }
    if (Q container inspect $Name) { $here += "the container $Name" }
    if (Q volume inspect $Volume) { $here += "the volume $Volume, with its models and conversations" }
    if ($here.Count -gt 0) {
        Fail ("SiLVANN is installed here already: " + ($here -join "; ") + ".`n" +
              "   Keep it with export first (silvann_export.cmd, to another folder), or remove it with delete " +
              "(silvann_delete.cmd, the volume after a second yes), then import")
    }
    Say "the image <- $d\silvann-image.tar"
    if (-not (E load -i (Join-Path $d "silvann-image.tar"))) { Fail "the image was not loaded" }
    if (-not (Q volume create $Volume)) { Fail "the volume $Volume could not be made" }
    Say "the volume <- $d\silvann-data.tar"
    if (-not (E run --rm --user 0:0 -v "${Volume}:/data" -v "${d}:/in:ro" --entrypoint tar $Image -xpf /in/silvann-data.tar -C /data)) {
        Fail "the volume was not imported"
    }
    if ($Image -ne $DefaultImage) { Note "the image is ${Image}: set SILVANN_IMAGE=$Image before starting it" }
    Add-Shortcuts
    Say "imported: start SiLVANN from the desktop"
}

switch ($Action) {
    "install"      { Do-Install }
    "update"       { Do-Update }
    "start"        { Do-Start "127.0.0.1:" }
    "start_shared" { Do-Start "" }
    "stop"         { Do-Stop }
    "delete"       { Do-Delete }
    "export"       { Do-Export }
    "import"       { Do-Import }
    "install_nonfree" { Do-InstallNonfree }
    default        { foreach ($l in (Get-Content $PSCommandPath | Select-Object -Skip 1)) { if ($l -eq "#>") { break }; $l }; exit 2 }
}
