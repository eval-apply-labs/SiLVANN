#!/usr/bin/env python3
"""Build the engine into lib/ on Windows — build.sh's counterpart there: the evaluator, a Python module compiled with
clang-cl, and one library a silicon family — x86_avx2 with clang-cl, nvidia_cuda12 with nvcc over MSVC — each built
where its toolchain is installed and skipped where it is not.

  SILVANN_SILICON_FAMILIES  which families to build, e.g. "x86_avx2 nvidia_cuda12"   default those two; the OpenCL
                            ones, khronos_opencl2 and khronos_opencl1_igpu, when named; "none" for the evaluator alone
  SILVANN_CUDA_ARCHES       the NVIDIA cards to build for, e.g. 120                   default every one the family lists
  SILVANN_NVCC              CUDA's compiler            default %CUDA_PATH%\\bin\\nvcc.exe, else nvcc on the PATH
  SILVANN_CLANG_CL          clang-cl                   default Visual Studio's "C++ Clang tools", else on the PATH

  python build.py             build (install.ps1 runs it with the release's own Python and keeps what it says)
  python build.py --dry-run   the commands it would run, not run — on any system

The toolchains are Visual Studio's (2022 or newer, the "Desktop development with C++" workload and its "C++ Clang tools
for Windows") and CUDA's (12.x). Visual Studio's environment is taken from its `vcvars64.bat`, so a plain PowerShell
can run this. ASSUMED until a Windows PC has built with it (NN-94): every command below is written from the tools'
documentation, and the first build is the check.
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import sysconfig

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "engine", "src")
WORK = os.path.join(HERE, "engine", "build", "silicon_families")    # each family's generated translation unit
LIB = os.path.join(HERE, "lib")
FAMILIES_OUT = os.path.join(LIB, "silicon_families")
CL_HEADERS = os.path.join(HERE, "engine", "scripts", "third_party", "OpenCL-Headers")   # Khronos' own, carried
DRY = "--dry-run" in sys.argv
WINDOWS = os.name == "nt"


def say(m):
    print(m, flush=True)


def visual_studio():
    """Visual Studio's folder, from its own locator, or None."""
    where = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                         "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.isfile(where):
        return None
    r = subprocess.run([where, "-latest", "-products", "*", "-requires",
                        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                       capture_output=True, text=True)
    path = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else ""
    return path or None


def toolchain_environment(vs):
    """The environment MSVC's tools need: this one when `cl.exe` is on its PATH already (a Developer PowerShell),
    else what Visual Studio's `vcvars64.bat` sets."""
    if shutil.which("cl") or vs is None or DRY:
        return dict(os.environ)
    bat = os.path.join(vs, "VC", "Auxiliary", "Build", "vcvars64.bat")
    # ⛳ A COMMAND LINE AS ONE STRING, which Windows hands cmd as it is; `/u` makes `set` write UTF-16, so a folder named
    #   outside the PC's code page (a user's name) comes back whole
    r = subprocess.run('cmd /u /c ""%s" >nul 2>&1 && set"' % bat, capture_output=True)
    env = {}
    for line in r.stdout.decode("utf-16-le", errors="replace").splitlines():
        k, sep, v = line.partition("=")
        if sep and k:
            env[k] = v
    return env if "INCLUDE" in env else dict(os.environ)


def find(env, name, override, *candidates):
    """A tool: the variable that names it, the first candidate that exists, else the PATH."""
    if os.environ.get(override):
        return os.environ[override]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return shutil.which(name, path=env.get("PATH", env.get("Path"))) or ("<%s>" % name if DRY else None)


def clang_builtins(clang_cl):
    """compiler-rt's builtins beside clang-cl, which `__builtin_cpu_supports` reads the processor's features from: MSVC's
    linker does not add them by itself."""
    if not clang_cl or not os.path.isfile(clang_cl):
        return []
    root = os.path.dirname(os.path.dirname(clang_cl))
    found = sorted(glob.glob(os.path.join(root, "lib", "clang", "*", "lib", "windows", "clang_rt.builtins-x86_64.lib")))
    return [found[-1]] if found else []


def targets(family, row):
    """A family's cards, from the targets row of its own `includes.cuh` — the list build_silicon_families.sh reads."""
    with open(os.path.join(SRC, "silicon_families", family, "includes.cuh"), encoding="utf-8") as f:
        for line in f:
            if re.match(r"^#define %s\(X\)" % row, line):
                return re.findall(r"X\(([a-z0-9_]+)\)", line)
    return []


def unanswered(family):
    """The family's reason not to be built — a package call it does not answer — or None (build_silicon_families.sh's
    check, by the same script)."""
    script = os.path.join(HERE, "engine", "scripts", "silicon_family_unanswered.py")
    if not os.path.isfile(script) or DRY:
        return None
    r = subprocess.run([sys.executable, script, family], cwd=os.path.join(HERE, "engine"), capture_output=True, text=True)
    if r.returncode == 1:
        return "not built: it does not answer " + ", ".join(r.stdout.split())
    return None


def opencl_import_library(env, lib_tool):
    """OpenCL.dll's import library, made from the carried `cl.h`'s own declarations: every GPU driver installs the DLL,
    and nothing on Windows installs a library to link it with but a vendor's SDK. None when it could not be made."""
    with open(os.path.join(CL_HEADERS, "CL", "cl.h"), encoding="utf-8") as f:
        names = sorted(set(re.findall(r"CL_API_CALL\s+(cl\w+)\s*\(", f.read())))
    out = os.path.join(WORK, "OpenCL")
    if not DRY:
        with open(out + ".def", "w", encoding="utf-8") as f:
            f.write("LIBRARY OpenCL.dll\nEXPORTS\n" + "".join("    %s\n" % n for n in names))
    say("   OpenCL.dll's %d calls, from cl.h" % len(names))
    made = run([lib_tool, "/nologo", "/def:" + out + ".def", "/out:" + out + ".lib", "/machine:x64"], env, out + ".log")
    return out + ".lib" if made else None


def run(cmd, env, log):
    say("   " + " ".join('"%s"' % c if " " in c else c for c in cmd))
    if DRY:
        return True
    with open(log, "w", encoding="utf-8", errors="replace") as out:
        r = subprocess.run(cmd, env=env, stdout=out, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        with open(log, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        errors = [l for l in lines if "error" in l.lower()][:15]
        say("   ⛔ it failed (%s):\n      %s" % (log, "\n      ".join(errors or lines[-15:])))
    return r.returncode == 0


def smart_app_control():
    """Windows 11's Smart App Control as the registry reports it — "on", "evaluation" or "off" — or None where there is
    none. On, it loads unsigned code only where Microsoft's reputation service knows it: a library built here, or the
    release's unsigned ones, it refuses ("An Application Control policy has blocked this file")."""
    if not WINDOWS:
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\CI\Policy") as key:
            state, _ = winreg.QueryValueEx(key, "VerifiedAndReputablePolicyState")
    except OSError:
        return None
    return {0: "off", 1: "on", 2: "evaluation"}.get(state)


SMART_APP_CONTROL = ("Windows' Smart App Control refuses libraries that are not signed, as SiLVANN's are for now: turn it off (Windows Security > App & browser control > Smart App Control settings > Off; on most PCs it cannot be turned on again without resetting Windows), or run SiLVANN on Linux (docs/users.md)")


def exports_entry(path, family):
    """Whether the family's library offers `silvann_silicon_family_get` by name. An OpenCL family is linked to
    OpenCL.dll, which only a GPU's driver installs: on a PC without it (a build machine) it is said, not checked."""
    if DRY:
        return True
    if family.startswith("khronos_") and not os.path.isfile(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                                                                         "System32", "OpenCL.dll")):
        say("   ⚠ %s is not loaded here to check it: this PC has no OpenCL.dll" % family)
        return True
    import ctypes
    try:
        return hasattr(ctypes.CDLL(path), "silvann_silicon_family_get")
    except OSError as e:
        say("   ⛔ %s would not load: %s" % (path, e))
        if smart_app_control() == "on":
            say("      " + SMART_APP_CONTROL)
        return False


def main():
    if not WINDOWS and not DRY:
        sys.exit("build.py builds on Windows; on Linux, build.sh")
    os.makedirs(FAMILIES_OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    vs = visual_studio()
    env = toolchain_environment(vs)
    env["VSLANG"] = "1033"   # MSVC's messages in English whatever the PC's language, as the logs are read
    llvm = os.path.join(vs, "VC", "Tools", "Llvm", "x64", "bin", "clang-cl.exe") if vs else None
    clang_cl = find(env, "clang-cl", "SILVANN_CLANG_CL", llvm)
    nvcc = find(env, "nvcc", "SILVANN_NVCC",
                os.path.join(os.environ["CUDA_PATH"], "bin", "nvcc.exe") if os.environ.get("CUDA_PATH") else None)
    say("== the toolchains: Visual Studio %s · clang-cl %s · nvcc %s" % (vs or "not found", clang_cl or "not found",
                                                                         nvcc or "not found"))
    if not clang_cl:
        sys.exit("⛔ no clang-cl: Visual Studio's installer, \"Desktop development with C++\", and in it \"C++ Clang tools "
                 "for Windows\" — then run this again")
    builtins = clang_builtins(clang_cl)
    llvm_bin = os.path.dirname(clang_cl) if os.path.isfile(clang_cl) else ""
    # the C preprocessor the OpenCL generators flatten nn's card side with, and the librarian an import library is made
    # with: Visual Studio's own, beside clang-cl
    clang = os.path.join(llvm_bin, "clang.exe") if llvm_bin else "<clang>"
    lib_tool = shutil.which("lib", path=env.get("PATH", env.get("Path"))) or (os.path.join(llvm_bin, "llvm-lib.exe")
                                                                             if llvm_bin else "<lib>")
    opencl_lib = None
    # ⛳ THE SAME FLAGS FOR EVERY clang-cl BUILD: C++17, exceptions (pybind11 throws), the DLL C runtime Python itself
    #   uses, warnings off (the tree is checked by g++ and clang on Linux), the generated translation unit taken as C++
    #   whatever its ending
    common = ["/nologo", "/std:c++17", "/EHsc", "/MD", "/W0", "-D_CRT_SECURE_NO_WARNINGS", "/I", SRC]
    ok = True

    say("== the evaluator")
    import pybind11
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".pyd"
    module = os.path.join(LIB, "silvann_engine_cpu" + suffix)
    obj = os.path.join(HERE, "engine", "build", "evaluator") + os.sep
    os.makedirs(obj, exist_ok=True)
    ok &= run([clang_cl] + common + ["/O2", "-DSILVANN_MODULE_NAME=silvann_engine_cpu", "/I", pybind11.get_include(),
                                     "/I", sysconfig.get_path("include"), "/Tp" + os.path.join(SRC, "manifest.cu"),
                                     "/LD", "/Fo" + obj, "/Fe" + module, "/link",
                                     "/LIBPATH:" + os.path.join(sys.base_prefix, "libs")] + builtins,
              env, os.path.join(LIB, "evaluator.log"))

    say("== the silicon families")
    wanted = [f for f in os.environ.get("SILVANN_SILICON_FAMILIES", "x86_avx2 nvidia_cuda12").replace(",", " ").split()
              if f != "none"]
    built, failed = [], []
    for family in wanted:
        out = os.path.join(FAMILIES_OUT, "libsilvann_%s.dll" % family)
        skipped = os.path.join(FAMILIES_OUT, "libsilvann_%s.skipped" % family)
        for f in (out, skipped):
            if os.path.exists(f) and not DRY:
                os.remove(f)
        why = unanswered(family)
        if why:
            with open(skipped, "w", encoding="utf-8") as f:
                f.write(why + "\n")
            say("   ⚠ %s skipped — %s" % (family, why))
            continue
        tu_line = '/* GENERATED BY build.py — do not edit, and do not check in. */\n#include "silicon_families/%s/unit.cuh"\n'
        log = os.path.join(FAMILIES_OUT, family + ".log")
        if family == "x86_avx2":
            # ⛳ THE INSTRUCTION SET THE FAMILY'S NAME PROMISES, as build_silicon_families.sh's g++ line: AVX2, FMA, F16C
            tu = os.path.join(WORK, "x86_avx2.cpp")
            if not DRY:
                with open(tu, "w", encoding="utf-8") as f:
                    f.write(tu_line % family)
            done = run([clang_cl] + common + ["/O2", "/clang:-O3", "-mavx2", "-mfma", "-mf16c", "/Tp" + tu, "/LD",
                                              "/Fo" + WORK + os.sep, "/Fe" + out, "/link"] + builtins, env, log)
        elif family == "nvidia_cuda12":
            if not nvcc:
                say("   ⚠ nvidia_cuda12 skipped — no nvcc: CUDA 12's toolkit from developer.nvidia.com/cuda-downloads, "
                    "then `silvann build`")
                continue
            tu = os.path.join(WORK, "nvidia_cuda12.cu")
            if not DRY:
                with open(tu, "w", encoding="utf-8") as f:
                    f.write(tu_line % family)
            want = [t[3:] for t in targets("nvidia_cuda12", "NVIDIA_CUDA12__TARGETS")]
            arches = os.environ.get("SILVANN_CUDA_ARCHES", ",".join(want)).replace(",", " ").split()
            gencode = []
            for a in arches:
                gencode += ["-gencode", "arch=compute_%s,code=sm_%s" % (a, a)]
            gencode += ["-gencode", "arch=compute_%s,code=compute_%s" % (arches[-1], arches[-1])]
            # ⛳ MSVC IS nvcc's HOST COMPILER HERE, and three of its switches are the tree's: the conforming
            #   preprocessor (the doors' rows pass argument lists through macros, which the traditional one splits
            #   wrongly), more sections than an object holds by default, and the DLL C runtime. `-cudart static`, as on
            #   Linux: the box needs only the driver. A newer Visual Studio than this CUDA names is let through.
            done = run([nvcc] + gencode + ["-O2", "-std=c++17", "-cudart", "static", "-allow-unsupported-compiler",
                                           "-Xcompiler", "/MD,/EHsc,/Zc:preprocessor,/bigobj,/W0",
                                           "-D_CRT_SECURE_NO_WARNINGS", "-shared", "-I", SRC, tu, "-o", out], env, log)
        elif family in ("khronos_opencl2", "khronos_opencl1_igpu"):
            # ⛳ NO CARD COMPILER, as on Linux: the kernels are OpenCL C, generated from nn's rows and compiled by the
            #   driver for the card it finds. The host side is built against the carried headers and OpenCL.dll.
            generator = "opencl_kernel_source.py" if family == "khronos_opencl2" else "opencl1_kernel_source.py"
            kernels = os.path.join(WORK, family + "_kernels.cpp")
            tu = os.path.join(WORK, family + ".cpp")
            if not DRY:
                with open(tu, "w", encoding="utf-8") as f:
                    f.write(tu_line % family)
            made = run([sys.executable, os.path.join(HERE, "engine", "scripts", generator), kernels],
                       dict(env, SILVANN_CPP=clang), os.path.join(FAMILIES_OUT, family + "_kernels.log"))
            if made and opencl_lib is None:
                opencl_lib = opencl_import_library(env, lib_tool)
            done = made and opencl_lib is not None and run(
                [clang_cl] + common + ["/O2", "/I", CL_HEADERS, "/Tp" + tu, "/Tp" + kernels, "/LD", "/Fo" + WORK + os.sep,
                                       "/Fe" + out, "/link", opencl_lib] + builtins, env, log)
        else:
            say("   ⛔ %s: build.py has no recipe for it on Windows" % family)
            failed.append(family)
            continue
        if done and exports_entry(out, family):
            built.append(family)
            say("   ✅ %s -> %s" % (family, out))
        else:
            failed.append(family)

    # ⛳ A FAMILY THAT DID NOT BUILD IS SAID, NOT FATAL, as build.sh has it: the evaluator runs on what did
    if failed:
        say("== not built: %s — each one's .log in %s" % (", ".join(failed), FAMILIES_OUT))
    if DRY:
        return 0
    sys.path.insert(0, LIB)
    try:
        import silvann_engine_cpu as e
        names = ["%s (%s)" % (f.get("silicon"), f.get("outcome")) if isinstance(f, dict) else str(f)
                 for f in e.silicon_families()]
        say("== built: the evaluator, and the families " + (", ".join(names) or "(none)"))
    except ImportError as err:
        say("⛔ the evaluator does not import: %s" % err)
        if smart_app_control() == "on":
            say("      " + SMART_APP_CONTROL)
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
