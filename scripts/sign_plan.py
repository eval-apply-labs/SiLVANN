#!/usr/bin/env python3
"""What a release's Windows job signs, and what it takes unchanged from the last signed release.

Each file the job signs — the evaluator for each Python, each silicon family, SiLVANN.exe — has a key made of what it is
built from: the engine's files its compile includes (its translation unit's quoted includes, followed — every family
includes each family's `includes.cuh` through the roster, and nothing else of theirs), for an OpenCL family every file of
the packages, of the folders it reaches and of the carried OpenCL headers as well (its kernels are generated from them),
the script that builds it, and
what it is built with (the Python, CUDA's version, AMD's HIP's version, an OpenCL family's kernel generator). A file whose key is the last signed release's is taken from that release's zips, its bytes
checked against that release's manifest; the rest are signed. The release carries its own manifest, `signed_manifest.json`,
for the next one.

⭐ AND BY ITS UNIT (NN-195, ⚖ 2026-11-16 *"yes, build it on a review branch"*): the build writes beside each file a `.unit`
(scripts/unit_key.py) — the lines of this tree its compiler preprocessed with the build's own switches, and that compile
line — and a file whose unit key is the last signed release's is taken from it too, whatever its includes did. A
`#define` no family expands (sys's verb table, which re-signed all eleven of rel1.1.0's files) leaves the unit as it was;
a macro, a branch or a literal that changes the code changes it. A file is reused when either key is the last release's:
the unit only adds reuses. A file with no `.unit` (a build that could not make one), or a last release whose manifest
has no unit keys, is planned by the includes alone, as before.

⭐ AND A BUILD SCRIPT'S EDIT THAT CHANGES NO COMPILE LINE CHANGES NO KEY: every key holds scripts/build.py's bytes, so an
edit to it — the unit keys' own, a comment — once changed every key. Given the last signed release's build.py
(`--previous-build`), the plan runs both with `--dry-run` over this tree and compares the lines they would run (the
compilers', the librarian's, the kernels' generators), leaving out what reads and changes no output: the unit's `/E`
pass, nvcc's `--keep --keep-dir`. Where they are the same, a file's key is also tried with the last release's build.py
in this one's place, and a match is a reuse "by its includes, the build script's recipe the same". Run from a release's folder (`.github/workflows/windows-libraries.yml`, `sign`):

    python3 scripts/sign_plan.py plan BUILT PREVIOUS PLAN [--previous-build BUILD_PY]   which are signed, which reused
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
WINDOWS_ZIP, PORTABLE_ZIP = "silvann_windows_x64.zip", "silvann_portable_x64.zip"


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


def key(files, extra_files, extra_words, swap=None):
    """The key: the files (engine files a compile reads), the extra files, the extra words. `swap`: {extra file: the
    file whose bytes stand for it} — the last release's build script in this one's place (▶ `recipe_same`)."""
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(("%s\t%s\n" % (f.replace(os.sep, "/"), sha256(f))).encode())
    for f in extra_files:
        h.update(("%s\t%s\n" % (f.replace(os.sep, "/"), sha256((swap or {}).get(f, f)))).encode())
    for w in extra_words:
        h.update(("%s\n" % w).encode())
    return h.hexdigest()


def unit_of(built, fid, words):
    """The unit key of the file `fid` in BUILT — its `.unit`'s hash of what its compiler preprocessed of this tree, its
    compile line and `words` (what it is: an evaluator and its Python, a family and CUDA's version) — or None where the
    build wrote no `.unit` beside it."""
    try:
        with open(os.path.join(built, fid + ".unit"), encoding="utf-8") as f:
            u = json.load(f)
        digest, line = u["unit"], u.get("line") or []
    except (OSError, ValueError, KeyError):
        return None
    h = hashlib.sha256(("unit\t%s\n" % digest).encode())
    h.update(("line\t%s\n" % json.dumps(line)).encode())
    for w in words:
        h.update(("%s\n" % w).encode())
    return h.hexdigest()


BUILD_PY = os.path.join("scripts", "build.py")
# what a dry run says that reads and changes no output: the unit's preprocessor pass, the unit's own line, nvcc's kept files
DRY_LEAVES_OUT = (re.compile(r" /E$"), re.compile(r"^\s*the unit: "), re.compile(r" --keep --keep-dir \S+"))


def _dry_lines(script):
    """What `script` (a build.py, set in this tree's scripts/ so it reads this tree) would run, by its `--dry-run`: the
    lines naming a compiler, the librarian or a kernel generator, less what DRY_LEAVES_OUT names."""
    import subprocess
    here = os.path.join("scripts", "_build_under_test.py") if os.path.abspath(script) != os.path.abspath(BUILD_PY) else BUILD_PY
    if here != BUILD_PY:
        import shutil
        shutil.copy(script, here)
    try:
        env = dict(os.environ, SILVANN_SILICON_FAMILIES=os.environ.get(
            "SILVANN_SILICON_FAMILIES", "x86_avx2 nvidia_cuda12 khronos_opencl2 intel_opencl2 nvidia_opencl1 intel_opencl1_igpu"))
        r = subprocess.run([sys.executable, here, "--dry-run"], capture_output=True, text=True, env=env)
    finally:
        if here != BUILD_PY and os.path.exists(here):
            os.remove(here)
    if r.returncode != 0:
        return None
    out = []
    for line in r.stdout.splitlines():
        if not any(w in line for w in ("<clang-cl>", "<nvcc>", "<amd-clang++>", "<lib>", "kernel_source.py")) or DRY_LEAVES_OUT[1].search(line):
            continue
        if DRY_LEAVES_OUT[0].search(line):
            continue
        out.append(DRY_LEAVES_OUT[2].sub("", line).replace(os.path.basename(here), "build.py"))
    return out


def recipe_same(previous_build):
    """Whether this tree's build.py and the last release's would run the same lines (▶ `_dry_lines`); False where either
    dry run fails or says nothing."""
    if not previous_build or not os.path.isfile(previous_build):
        return False
    now, before = _dry_lines(BUILD_PY), _dry_lines(previous_build)
    return bool(now) and now == before


def files_of(built, previous_build=None):
    """The job's files to ship, by their name in BUILT: (id, key, the key with the last release's build.py where its
    recipe is this one's or None, unit key or None, where in the zips)."""
    swap = {BUILD_PY: previous_build} if previous_build else None
    out = []
    cuda = os.environ.get("CUDA_VERSION", "")
    # AMD's compiler is whichever of AMD's two distributions the families' job could install: the job says its HIP's version
    hip_said = os.path.join(built, "families", "notices", "HIP_VERSION.txt")
    hip = open(hip_said, encoding="utf-8-sig", errors="replace").read().split("\n")[0].strip() if os.path.isfile(hip_said) else ""
    for d in sorted(os.listdir(built)):
        if not d.startswith("evaluator-"):
            continue
        for n in sorted(os.listdir(os.path.join(built, d))):
            if n.endswith(".pyd"):
                files, words = closure([os.path.join(SRC, "manifest.cu")]), ["evaluator", n]
                out.append((d + "/" + n, key(files, [BUILD_PY], words), swap and key(files, [BUILD_PY], words, swap),
                            unit_of(built, d + "/" + n, words), (WINDOWS_ZIP, "lib/" + n)))
    fam_dir = os.path.join(built, "families", "lib", "silicon_families")
    for n in sorted(os.listdir(fam_dir)) if os.path.isdir(fam_dir) else []:
        m = re.match(r"libsilvann_(.+)\.dll$", n)
        if not m:
            continue
        fam = m.group(1)
        files = closure([os.path.join(FAMILIES, fam, "unit.cuh")])
        extra = [BUILD_PY]
        if fam in OPENCL_GENERATOR:
            extra.append(os.path.join("engine", "scripts", OPENCL_GENERATOR[fam]))
            extra += sorted(os.path.join(r, x) for r, _, xs in os.walk(CL_HEADERS) for x in xs)   # build.py's /I for them
            folders = {family_of(f) for f in files if not f.endswith("includes.cuh")} - {None}   # its own and its base
            files |= {f for f in engine_files() if f.startswith(os.path.join(SRC, "packages") + os.sep)
                      or (family_of(f) in folders and not f.endswith("includes.cuh"))}
        words = ["family", fam] + (["cuda " + cuda] if fam == "nvidia_cuda12" else []) + (["hip " + hip] if fam == "amd_rocm10" else [])
        fid = "families/lib/silicon_families/" + n
        out.append((fid, key(files, extra, words), swap and key(files, extra, words, swap), unit_of(built, fid, words),
                    (WINDOWS_ZIP, "lib/silicon_families/" + n)))
    if os.path.isfile(os.path.join(built, "families", "lib", "SiLVANN.exe")):
        launcher = [os.path.join("scripts", "portable", "SiLVANN.c"), os.path.join("scripts", "portable", "build_launcher.ps1")]
        out.append(("families/lib/SiLVANN.exe", key(set(), launcher, ["launcher"]), None,
                    unit_of(built, "families/lib/SiLVANN.exe", ["launcher"]), (PORTABLE_ZIP, "SiLVANN/windows/SiLVANN.exe")))
    return out


def plan(built, previous, out, previous_build=None):
    prev = json.load(open(previous)) if previous != "-" else {"release": None, "files": {}}
    same = recipe_same(previous_build)
    if previous_build:
        print("the last release's build.py %s" % ("runs the same lines as this one's: a key is tried with it too" if same
                                                  else "runs other lines than this one's (or did not dry-run): not tried"))
    files, signs = {}, 0
    for fid, k, k_before, u, (zipname, member) in files_of(built, previous_build if same else None):
        before = prev["files"].get(fid)
        by = (None if not before else "key" if before["key"] == k else "recipe" if k_before and before["key"] == k_before
              else "unit" if u and before.get("unit_key") == u else None)
        if by:
            files[fid] = {"key": k, "unit_key": u, "action": "reuse", "by": by, "zip": zipname, "member": member,
                          "sha256": before["sha256"], "signed_in": before["signed_in"]}
        else:
            files[fid] = {"key": k, "unit_key": u, "action": "sign"}
            signs += 1
    json.dump({"previous": prev["release"], "files": files}, open(out, "w"), indent=1, sort_keys=True)
    for fid, f in sorted(files.items()):
        print("%-6s %s%s%s" % (f["action"], fid, "" if f["action"] == "sign" else "  (signed in %s, by its %s)"
                               % (f["signed_in"], {"key": "includes", "recipe": "includes, the build script's recipe the same",
                                                   "unit": "unit"}[f["by"]]),
                               "" if f["unit_key"] else "  [no .unit]"))
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
    files = {fid: {"key": f["key"], "unit_key": f.get("unit_key"), "sha256": sha256(os.path.join(built, fid)),
                   "signed_in": f["signed_in"] if f["action"] == "reuse" else tag}
             for fid, f in p["files"].items()}
    json.dump({"release": tag, "files": files}, open(out, "w"), indent=1, sort_keys=True)
    print("%s: %d files, %d signed in it" % (out, len(files), sum(f["signed_in"] == tag for f in files.values())))


if __name__ == "__main__":
    cmd, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if cmd == "plan" and len(args) in (3, 5) and (len(args) == 3 or args[3] == "--previous-build"):
        plan(args[0], args[1], args[2], args[4] if len(args) == 5 else None)
    elif cmd == "reuse" and len(args) == 3:
        reuse(*args)
    elif cmd == "manifest" and len(args) == 4:
        manifest(*args)
    else:
        sys.exit(__doc__)
