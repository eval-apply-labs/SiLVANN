#!/usr/bin/env python3
"""Build the engine into lib/ on Windows — build.sh's counterpart there: the evaluator, a Python module compiled with
clang-cl, and one library a silicon family — x86_avx2 with clang-cl, nvidia_cuda12 with nvcc over MSVC, amd_rocm10 with
AMD's clang++ — each built where its toolchain is installed and skipped where it is not.

  SILVANN_SILICON_FAMILIES  which families to build, e.g. "x86_avx2 nvidia_cuda12"   default those two, and amd_rocm10
                            where AMD's compiler is found; the OpenCL ones, khronos_opencl2, intel_opencl2,
                            nvidia_opencl1 and intel_opencl1_igpu, when named; "none" for the evaluator alone
  SILVANN_CUDA_ARCHES       the NVIDIA cards to build for, e.g. 120                   default every one the family lists
  SILVANN_NVCC              CUDA's compiler            default %CUDA_PATH%\\bin\\nvcc.exe, else nvcc on the PATH
  SILVANN_HIP_ROOT          AMD's ROCm for Windows     default %HIP_PATH%, which the HIP SDK's installer sets: the folder
                            that holds include\\hip and lib\\amdhip64.lib, its clang++ in bin or in lib\\llvm\\bin
  SILVANN_HIP_ARCHES        the AMD cards to build for, e.g. gfx1200                  default every one the family lists
  SILVANN_CLANG_CL          clang-cl                   default Visual Studio's "C++ Clang tools", else on the PATH

  python scripts/build.py     build (install.ps1 runs it with the release's own Python and keeps what it says)
  python scripts/build.py --dry-run   the commands it would run, not run — on any system

The toolchains are Visual Studio's (2022 or newer, the "Desktop development with C++" workload and its "C++ Clang tools
for Windows") and CUDA's (12 or 13). Visual Studio's environment is taken from its `vcvars64.bat`, so a plain PowerShell
can run this. MEASURED on one PC (an RTX 5070 Ti beside a Ryzen 7 5800X), through install.ps1: 2026-10-19 the evaluator
and x86_avx2 built with clang-cl and nvidia_cuda12 with nvcc 13.4; 2026-10-28 v0.21.2 served Qwen 3.6 35B-A3B at 48
tokens/s (measurements/2026-10-28_nn94_windows_5070ti.md). A release's libraries are this file's on GitHub's Windows
runners, every family (.github/workflows/windows-libraries.yml).

⚠ amd_rocm10 ON WINDOWS HAS RUN ON NO CARD (⚖ 2026-11-23 *"the next release will need to work on adding that card on
windows through hip, even if it is untested"*): it is built here as scripts/build_silicon_families.sh builds it on Linux —
the same source, the same switches — by a runner with no card, and what it does on a Radeon is a card's to say.
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import sysconfig

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # the release's folder: this file is in its scripts/
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


def amd_toolchain():
    """AMD's ROCm for Windows — (its folder, its clang++) — or (None, None): the folder SILVANN_HIP_ROOT names, else
    HIP_PATH (the HIP SDK's installer sets it), where it holds HIP's header and its import library; the compiler in its
    `bin` (the HIP SDK's layout) or its `lib\\llvm\\bin` (ROCm's own archive's)."""
    for root in (os.environ.get("SILVANN_HIP_ROOT"), os.environ.get("HIP_PATH")):
        root = (root or "").rstrip("\\/")
        if not root or not os.path.isfile(os.path.join(root, "include", "hip", "hip_runtime.h")) \
                or not os.path.isfile(os.path.join(root, "lib", "amdhip64.lib")):
            continue
        for sub in (("bin",), ("lib", "llvm", "bin")):
            compiler = os.path.join(root, *sub, "clang++.exe")
            if os.path.isfile(compiler):
                return root, compiler
    return ("<hip>", "<amd-clang++>") if DRY else (None, None)


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


def _line(cmd):
    """A compile line as the unit key reads it: the compiler's name and every switch, a path in this tree relative to it,
    one outside it by what it is (the toolset's, Python's, pybind11's) — so the key follows the flags, not this script's
    text or the runner's folders."""
    def word(a):
        for flag in ("/Tp", "/Fo", "/Fe", "-I"):
            if a.startswith(flag) and len(a) > len(flag):
                return flag + word(a[len(flag):])
        if os.path.isabs(a):
            rel = os.path.relpath(a, HERE) if os.path.splitdrive(a)[0].lower() == os.path.splitdrive(HERE)[0].lower() else ".."
            return rel.replace(os.sep, "/") if not rel.startswith("..") else "<outside:%s>" % os.path.basename(a.rstrip("\\/"))
        return a
    return [os.path.basename(cmd[0]).lower()] + [word(a) for a in cmd[1:]]


def unit(compile_line, env, out, log, keep_dir=None, nvcc_line=None, sides=None):
    """`out`.unit beside the binary `out`: the key scripts/sign_plan.py reads it by (▶ unit_key.py, NN-195) — the lines
    this tree wrote of what the build's own compile line preprocesses: clang-cl's and cl's line, its output and link left
    out, run again with `/E`; or nvcc's own `.ii` files, which `--keep` left in `keep_dir`; or, for AMD's clang++, each
    of `sides` run — the host's side and every card's, `-E` a line — with `nvcc_line` the line the key names. False,
    said, when it could not be made: the plan then keys the file by its includes alone, as before."""
    import unit_key
    if DRY:
        say("   the unit: %d sides, each its line with -E" % len(sides) if sides else
            "   " + " ".join('"%s"' % c if " " in c else c for c in compile_line + ["/E"]) if compile_line else
            "   the unit: nvcc's .ii in %s" % keep_dir)
        return True
    if sides is not None:
        texts = []
        for line in sides:
            r = subprocess.run(line, env=env, capture_output=True, text=True, errors="replace")
            if r.returncode != 0:
                with open(log, "w", encoding="utf-8", errors="replace") as f:
                    f.write(" ".join(line) + "\n" + r.stdout[-20000:] + r.stderr[-20000:])
                say("   ⚠ no unit key for %s: the preprocessor stopped (%s)" % (os.path.basename(out), log))
                return False
            texts.append(r.stdout)
        how = "%s -E, the host's side and %d cards'" % (os.path.basename(sides[0][0]), len(sides) - 1)
    elif keep_dir is not None:
        files = sorted(os.path.join(keep_dir, f) for f in os.listdir(keep_dir) if f.endswith(".ii"))
        texts = [open(f, encoding="utf-8", errors="replace").read() for f in files]
        how = "nvcc --keep: " + ", ".join(os.path.basename(f) for f in files)
    else:
        r = subprocess.run(compile_line + ["/E"], env=env, capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            with open(log, "w", encoding="utf-8", errors="replace") as f:
                f.write(r.stdout[-20000:] + r.stderr[-20000:])
            say("   ⚠ no unit key for %s: the preprocessor stopped (%s)" % (os.path.basename(out), log))
            return False
        texts, how = [r.stdout], "%s /E" % os.path.basename(compile_line[0])
    if not texts or not any(t.strip() for t in texts):
        say("   ⚠ no unit key for %s: nothing preprocessed" % os.path.basename(out))
        return False
    digest, n = unit_key.write(out + ".unit", texts, HERE, how, _line(compile_line or nvcc_line))
    say("   the unit: %s over %d of this tree's lines" % (digest[:16], n))
    return True


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


def pe_names(path):
    """(the names a Windows library exports, the libraries it asks for), read from the file itself — its export table
    and its import table — so a library this PC cannot load is still read. ([], []) where the file is not one."""
    import struct
    with open(path, "rb") as f:
        d = f.read()
    try:
        pe = struct.unpack_from("<I", d, 0x3C)[0]
        if d[:2] != b"MZ" or d[pe:pe + 4] != b"PE\0\0":
            return [], []
        sections, optional_size = struct.unpack_from("<H", d, pe + 6)[0], struct.unpack_from("<H", d, pe + 20)[0]
        optional = pe + 24
        if struct.unpack_from("<H", d, optional)[0] != 0x20B:             # a 64-bit library's header
            return [], []
        table = [struct.unpack_from("<IIII", d, optional + optional_size + 40 * k + 8) for k in range(sections)]

        def at(rva):                                                      # where in the file an address is
            for size, start, raw_size, raw in table:
                if start <= rva < start + max(size, raw_size):
                    return raw + rva - start
            raise ValueError(rva)

        def text(rva):
            o = at(rva)
            return d[o:d.index(b"\0", o)].decode("ascii", "replace")
        export_rva, _, import_rva, _ = struct.unpack_from("<IIII", d, optional + 112)
        exports, imports = [], []
        if export_rva:
            e = at(export_rva)
            count, names = struct.unpack_from("<I", d, e + 24)[0], struct.unpack_from("<I", d, e + 32)[0]
            exports = [text(struct.unpack_from("<I", d, at(names) + 4 * k)[0]) for k in range(count)]
        if import_rva:
            o = at(import_rva)
            while struct.unpack_from("<IIIII", d, o) != (0, 0, 0, 0, 0):
                imports.append(text(struct.unpack_from("<I", d, o + 12)[0]))
                o += 20
        return exports, imports
    except (struct.error, ValueError, IndexError):
        return [], []


def exports_entry(path, family):
    """Whether the family's library offers `silvann_silicon_family_get` by name. An OpenCL family is linked to
    OpenCL.dll, which only a GPU's driver installs: on a PC without it (a build machine) it is said, not checked. AMD's
    is linked to HIP's runtime, which a build machine does not hold either: its export table is read instead, and the
    runtime it asks for said."""
    if DRY:
        return True
    if family == "amd_rocm10":
        exports, imports = pe_names(path)
        say("   it asks for: %s" % (", ".join(imports) or "(nothing read)"))
        if not exports:
            say("   ⚠ %s's export table was not read: its entry point is not checked here" % family)
            return True
        if "silvann_silicon_family_get" not in exports:
            say("   ⛔ %s does not export silvann_silicon_family_get (it exports %s)" % (path, ", ".join(exports[:6])))
        return "silvann_silicon_family_get" in exports
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
        sys.exit("build.py builds on Windows; on Linux, scripts/build.sh")
    os.makedirs(FAMILIES_OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    vs = visual_studio()
    env = toolchain_environment(vs)
    env["VSLANG"] = "1033"   # MSVC's messages in English whatever the PC's language, as the logs are read
    llvm = os.path.join(vs, "VC", "Tools", "Llvm", "x64", "bin", "clang-cl.exe") if vs else None
    clang_cl = find(env, "clang-cl", "SILVANN_CLANG_CL", llvm)
    nvcc = find(env, "nvcc", "SILVANN_NVCC",
                os.path.join(os.environ["CUDA_PATH"], "bin", "nvcc.exe") if os.environ.get("CUDA_PATH") else None)
    hip_root, amd_clang = amd_toolchain()
    say("== the toolchains: Visual Studio %s · clang-cl %s · nvcc %s · AMD's clang++ %s"
        % (vs or "not found", clang_cl or "not found", nvcc or "not found", amd_clang or "not found"))
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
    evaluator = [clang_cl] + common + ["/O2", "-DSILVANN_MODULE_NAME=silvann_engine_cpu", "/I", pybind11.get_include(),
                                       "/I", sysconfig.get_path("include"), "/Tp" + os.path.join(SRC, "manifest.cu")]
    built_evaluator = run(evaluator + ["/LD", "/Fo" + obj, "/Fe" + module, "/link",
                                       "/LIBPATH:" + os.path.join(sys.base_prefix, "libs")] + builtins,
                          env, os.path.join(LIB, "evaluator.log"))
    ok &= built_evaluator
    if built_evaluator:
        unit(evaluator, env, module, os.path.join(LIB, "evaluator_unit.log"))

    say("== the silicon families")
    default = "x86_avx2 nvidia_cuda12" + (" amd_rocm10" if amd_clang and not DRY else "")
    wanted = [f for f in os.environ.get("SILVANN_SILICON_FAMILIES", default).replace(",", " ").split() if f != "none"]
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
            compile_line = [clang_cl] + common + ["/O2", "/clang:-O3", "-mavx2", "-mfma", "-mf16c", "/Tp" + tu]
            done = run(compile_line + ["/LD", "/Fo" + WORK + os.sep, "/Fe" + out, "/link"] + builtins, env, log)
            keep_dir = None
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
            # `--keep` leaves nvcc's own preprocessed files — the host's and each architecture's device side — for the unit
            #   key (▶ `unit`); it changes no step of the compile
            keep_dir = os.path.join(WORK, "nvidia_cuda12_keep")
            if not DRY:
                shutil.rmtree(keep_dir, ignore_errors=True)
                os.makedirs(keep_dir)
            compile_line = None
            nvcc_line = [nvcc] + gencode + ["-O2", "-std=c++17", "-cudart", "static", "-allow-unsupported-compiler",
                                            "-Xcompiler", "/MD,/EHsc,/Zc:preprocessor,/bigobj,/W0",
                                            "-D_CRT_SECURE_NO_WARNINGS", "-shared", "-I", SRC, tu]
            done = run(nvcc_line[:-1] + ["--keep", "--keep-dir", keep_dir, tu, "-o", out], env, log)
        elif family == "amd_rocm10":
            if not amd_clang:
                say("   ⚠ amd_rocm10 skipped — no AMD compiler: the HIP SDK for Windows from amd.com (its installer sets "
                    "HIP_PATH), then `silvann build`")
                continue
            tu = os.path.join(WORK, "amd_rocm10.cpp")
            if not DRY:
                with open(tu, "w", encoding="utf-8") as f:
                    f.write(tu_line % family)
            want = targets("amd_rocm10", "AMD_ROCM10__TARGETS_ON_WINDOWS")
            arches = [a for a in os.environ.get("SILVANN_HIP_ARCHES", ",".join(want)).replace(",", " ").split() if a in want]
            if not arches:
                say("   ⚠ amd_rocm10 skipped — none of SILVANN_HIP_ARCHES is its (%s)" % ", ".join(want))
                continue
            # ⛳ AS build_silicon_families.sh BUILDS IT ON LINUX, by AMD's clang++ itself where Linux goes through hipcc,
            #   which adds `-x hip` and the runtime's library: a code object for every card of the row a Windows build
            #   reads, 64 lanes to a wavefront (a Radeon runs 32 unless built so), every device function inlined into
            #   its kernel. MSVC's is the host's C++ library here, and the DLL C runtime, as every library of this
            #   build. The compiler finds its ROCm by ROCM_PATH and HIP_PATH, so no line names the PC's folders.
            # the unit: clang++ preprocesses a side at a time — the host's, then each card's (▶ `unit`)
            amd_env = dict(env, ROCM_PATH=hip_root, HIP_PATH=hip_root)
            flags = ["-x", "hip", "-mwavefrontsize64", "-mllvm", "-amdgpu-early-inline-all=true",
                     "-mllvm", "-amdgpu-function-calls=false", "-O2", "-std=c++17", "-fms-runtime-lib=dll", "-w",
                     "-D_CRT_SECURE_NO_WARNINGS", "-I", os.path.join(hip_root, "include"), "-I", SRC]
            offload = ["--offload-arch=" + a for a in arches]
            compile_line, keep_dir = None, None
            amd_line = [amd_clang] + offload + flags + [tu]
            # ⛳ THE DLL C RUNTIME AT THE LINK TOO: clang++, where it is not clang-cl, hands the linker the static one
            #   (`libcmt`) whatever the objects were compiled for, and the two do not mix — the objects' own, `msvcrt`,
            #   is left as the only one
            done = run(amd_line + ["-shared", "-o", out, "-L", os.path.join(hip_root, "lib"), "-lamdhip64",
                                   "-Xlinker", "/NODEFAULTLIB:libcmt"], amd_env, log)
            sides = [[amd_clang] + offload + flags + ["--cuda-host-only", "-E", tu]] + \
                    [[amd_clang, "--offload-arch=" + a] + flags + ["--cuda-device-only", "-E", tu] for a in arches]
        elif family in ("khronos_opencl2", "intel_opencl2", "nvidia_opencl1", "intel_opencl1_igpu"):
            # ⛳ NO CARD COMPILER, as on Linux: the kernels are OpenCL C, generated from nn's rows and compiled by the
            #   driver for the card it finds. The host side is built against the carried headers and OpenCL.dll.
            generator = "opencl_kernel_source.py" if family in ("khronos_opencl2", "intel_opencl2") else "opencl1_kernel_source.py"
            kernels = os.path.join(WORK, family + "_kernels.cpp")
            tu = os.path.join(WORK, family + ".cpp")
            if not DRY:
                with open(tu, "w", encoding="utf-8") as f:
                    f.write(tu_line % family)
            made = run([sys.executable, os.path.join(HERE, "engine", "scripts", generator), kernels],
                       dict(env, SILVANN_CPP=clang), os.path.join(FAMILIES_OUT, family + "_kernels.log"))
            if made and opencl_lib is None:
                opencl_lib = opencl_import_library(env, lib_tool)
            compile_line = [clang_cl] + common + ["/O2", "/I", CL_HEADERS, "/Tp" + tu, "/Tp" + kernels]
            keep_dir = None
            done = made and opencl_lib is not None and run(
                compile_line + ["/LD", "/Fo" + WORK + os.sep, "/Fe" + out, "/link", opencl_lib] + builtins, env, log)
        else:
            say("   ⛔ %s: build.py has no recipe for it on Windows" % family)
            failed.append(family)
            continue
        if done and exports_entry(out, family):
            built.append(family)
            say("   ✅ %s -> %s" % (family, out))
            if family == "amd_rocm10":
                unit(None, amd_env, out, os.path.join(FAMILIES_OUT, family + "_unit.log"), None, amd_line, sides)
            else:
                unit(compile_line, env, out, os.path.join(FAMILIES_OUT, family + "_unit.log"), keep_dir,
                     nvcc_line if family == "nvidia_cuda12" else None)
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
