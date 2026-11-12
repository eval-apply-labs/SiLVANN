#!/usr/bin/env python3
"""What a release's Windows job signs, and what it takes unchanged from the last signed release.

Each file the job signs — the evaluator for each Python, each silicon family, SiLVANN.exe — has a key made of what it is
built from: the engine's files its compile includes (its translation unit's quoted includes, followed — every family
includes each family's `includes.cuh` through the roster, and nothing else of theirs), for an OpenCL family every file of
the packages, of the folders it reaches and of the carried OpenCL headers as well (its kernels are generated from them),
the script that builds it, and
what it is built with (the Python, CUDA's version, an OpenCL family's kernel generator). A file whose key is the last signed release's is taken from that release's zips, its bytes
checked against that release's manifest; the rest are signed. The release carries its own manifest, `signed_manifest.json`,
for the next one. Run from a release's folder (`.github/workflows/windows-libraries.yml`, `sign`):

    python3 scripts/sign_plan.py plan BUILT PREVIOUS PLAN          each file's key; which are signed and which are reused
    python3 scripts/sign_plan.py reuse BUILT PLAN PREVDIR          the reused files from PREVDIR's zips into BUILT, checked
    python3 scripts/sign_plan.py manifest BUILT PLAN TAG OUT       the release's manifest, from the files it ships

BUILT is the job's artifacts as downloaded: evaluator-3.12/silvann_engine_cpu.cp312-win_amd64.pyd,
families/lib/silicon_families/libsilvann_x86_avx2.dll, families/lib/SiLVANN.exe. PREVIOUS is the last signed release's
`signed_manifest.json`, or - when there is none (every file is then signed).
"""
import hashlib
import json
import os
import re
import sys
import zipfile

SRC = os.path.join("engine", "src")
FAMILIES = os.path.join(SRC, "silicon_families")
CL_HEADERS = os.path.join("engine", "scripts", "third_party", "OpenCL-Headers")   # Khronos' own, carried
INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)
OPENCL_GENERATOR = {"khronos_opencl2": "opencl_kernel_source.py", "intel_opencl2": "opencl_kernel_source.py",
                    "nvidia_opencl1": "opencl1_kernel_source.py", "intel_opencl1_igpu": "opencl1_kernel_source.py"}
# where each file is in the release's zips, for a reuse
WINDOWS_ZIP, PORTABLE_ZIP = "silvann_windows_x64.zip", "silvann_portable_windows_x64.zip"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def engine_files():
    out = []
    for root, _, names in os.walk(SRC):
        out += [os.path.normpath(os.path.join(root, n)) for n in names]
    return sorted(out)


def family_of(path):
    """The silicon family folder a file is in, or None."""
    rel = os.path.relpath(path, FAMILIES)
    parts = rel.split(os.sep)
    return parts[0] if not rel.startswith("..") and len(parts) > 1 else None


def closure(starts):
    """The engine's files a compile starting from `starts` includes: its quoted includes followed, each resolved from
    the including file's folder, else from engine/src (the build's -I)."""
    seen, todo = set(), [os.path.normpath(s) for s in starts]
    while todo:
        f = todo.pop()
        if f in seen or not os.path.isfile(f):
            continue
        seen.add(f)
        try:
            text = open(f, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for inc in INCLUDE.findall(text):
            for base in (os.path.dirname(f), SRC):
                p = os.path.normpath(os.path.join(base, inc))
                if os.path.isfile(p):
                    todo.append(p)
                    break
    return seen


def key(files, extra_files, extra_words):
    """The key: the files (engine files a compile reads), the extra files, the extra words."""
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(("%s\t%s\n" % (f.replace(os.sep, "/"), sha256(f))).encode())
    for f in extra_files:
        h.update(("%s\t%s\n" % (f.replace(os.sep, "/"), sha256(f))).encode())
    for w in extra_words:
        h.update(("%s\n" % w).encode())
    return h.hexdigest()


def files_of(built):
    """The job's files to ship, by their name in BUILT: (id, key, where in the zips)."""
    out = []
    cuda = os.environ.get("CUDA_VERSION", "")
    for d in sorted(os.listdir(built)):
        if not d.startswith("evaluator-"):
            continue
        for n in sorted(os.listdir(os.path.join(built, d))):
            if n.endswith(".pyd"):
                k = key(closure([os.path.join(SRC, "manifest.cu")]), [os.path.join("scripts", "build.py")], ["evaluator", n])
                out.append((d + "/" + n, k, (WINDOWS_ZIP, "lib/" + n)))
    fam_dir = os.path.join(built, "families", "lib", "silicon_families")
    for n in sorted(os.listdir(fam_dir)) if os.path.isdir(fam_dir) else []:
        m = re.match(r"libsilvann_(.+)\.dll$", n)
        if not m:
            continue
        fam = m.group(1)
        files = closure([os.path.join(FAMILIES, fam, "unit.cuh")])
        extra = [os.path.join("scripts", "build.py")]
        if fam in OPENCL_GENERATOR:
            extra.append(os.path.join("engine", "scripts", OPENCL_GENERATOR[fam]))
            extra += sorted(os.path.join(r, x) for r, _, xs in os.walk(CL_HEADERS) for x in xs)   # build.py's /I for them
            folders = {family_of(f) for f in files if not f.endswith("includes.cuh")} - {None}   # its own and its base
            files |= {f for f in engine_files() if f.startswith(os.path.join(SRC, "packages") + os.sep)
                      or (family_of(f) in folders and not f.endswith("includes.cuh"))}
        k = key(files, extra, ["family", fam] + (["cuda " + cuda] if fam == "nvidia_cuda12" else []))
        out.append(("families/lib/silicon_families/" + n, k, (WINDOWS_ZIP, "lib/silicon_families/" + n)))
    if os.path.isfile(os.path.join(built, "families", "lib", "SiLVANN.exe")):
        launcher = [os.path.join("scripts", "portable", "SiLVANN.c"), os.path.join("scripts", "portable", "build_launcher.ps1")]
        out.append(("families/lib/SiLVANN.exe", key(set(), launcher, ["launcher"]), (PORTABLE_ZIP, "SiLVANN/SiLVANN.exe")))
    return out


def plan(built, previous, out):
    prev = json.load(open(previous)) if previous != "-" else {"release": None, "files": {}}
    files, signs = {}, 0
    for fid, k, (zipname, member) in files_of(built):
        before = prev["files"].get(fid)
        if before and before["key"] == k:
            files[fid] = {"key": k, "action": "reuse", "zip": zipname, "member": member, "sha256": before["sha256"],
                          "signed_in": before["signed_in"]}
        else:
            files[fid] = {"key": k, "action": "sign"}
            signs += 1
    json.dump({"previous": prev["release"], "files": files}, open(out, "w"), indent=1, sort_keys=True)
    for fid, f in sorted(files.items()):
        print("%-6s %s%s" % (f["action"], fid, "" if f["action"] == "sign" else "  (signed in %s)" % f["signed_in"]))
    print("%d to sign, %d reused from %s" % (signs, len(files) - signs, prev["release"] or "no earlier release"))


def reuse(built, plan_path, prevdir):
    p = json.load(open(plan_path))
    zips = {}
    for fid, f in sorted(p["files"].items()):
        if f["action"] != "reuse":
            continue
        z = zips.get(f["zip"]) or zips.setdefault(f["zip"], zipfile.ZipFile(os.path.join(prevdir, f["zip"])))
        data = z.read(f["member"])
        if hashlib.sha256(data).hexdigest() != f["sha256"]:
            sys.exit("%s in %s's %s is not the bytes its manifest names" % (f["member"], p["previous"], f["zip"]))
        with open(os.path.join(built, fid), "wb") as out:
            out.write(data)
        print("reused %s (signed in %s)" % (fid, f["signed_in"]))


def manifest(built, plan_path, tag, out):
    p = json.load(open(plan_path))
    files = {fid: {"key": f["key"], "sha256": sha256(os.path.join(built, fid)),
                   "signed_in": f["signed_in"] if f["action"] == "reuse" else tag}
             for fid, f in p["files"].items()}
    json.dump({"release": tag, "files": files}, open(out, "w"), indent=1, sort_keys=True)
    print("%s: %d files, %d signed in it" % (out, len(files), sum(f["signed_in"] == tag for f in files.values())))


if __name__ == "__main__":
    cmd, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if cmd == "plan" and len(args) == 3:
        plan(*args)
    elif cmd == "reuse" and len(args) == 3:
        reuse(*args)
    elif cmd == "manifest" and len(args) == 4:
        manifest(*args)
    else:
        sys.exit(__doc__)
