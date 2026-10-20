<#
SiLVANN built from source on Windows: install.ps1's steps, with the libraries compiled on this PC instead of taken
from the release - for a commit no release has built (a candidate, a change of your own), or to build them with your
own toolchain. Run it from the release's folder (a git clone).

  powershell -ExecutionPolicy Bypass -File build.ps1           update, the environment, the build, the command
  powershell -ExecutionPolicy Bypass -File build.ps1 -Build    the build alone (`silvann build`)

What it needs beyond install.ps1's Python and Git:
  Visual Studio's C++ tools and its "C++ Clang tools for Windows" (the Build Tools are enough)
  CUDA's toolkit (12 or 13), for an NVIDIA card: developer.nvidia.com/cuda-downloads - without it the CPU runs the models
Its variables are install.ps1's (SILVANN_DIR, SILVANN_NVCC, SILVANN_SILICON_FAMILIES, ...). ASCII on purpose, as
install.ps1 is.
#>
param([switch]$Build)
$rest = @("-FromSource")
if ($Build) { $rest += "-Build" }
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "install.ps1") @rest
exit $LASTEXITCODE
