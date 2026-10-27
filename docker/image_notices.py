"""What SiLVANN's Docker image holds, and each part's licence — written into the image as /opt/silvann/IMAGE_NOTICES.md
when it is built, from what is installed in it: the release's commit, the CUDA version its family was compiled with,
Ubuntu's packages (dpkg) and the Python packages in its environment (their metadata). Not legal advice.

    /opt/silvann/venv/bin/python docker/image_notices.py > IMAGE_NOTICES.md
"""
import importlib.metadata as md
import os
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(name, default=""):
    try:
        return open(os.path.join(HERE, name), encoding="utf-8").read().strip()
    except OSError:
        return default


def licence(dist):
    """A Python package's licence as its metadata states it: the SPDX expression, else the licence field when it is a
    short name, else the classifiers'."""
    m = dist.metadata
    if m.get("License-Expression"):
        return m["License-Expression"]
    text = (m.get("License") or "").strip()
    if text and "\n" not in text and len(text) < 60:
        return text
    named = [c.split(" :: ")[-1] for c in m.get_all("Classifier") or [] if c.startswith("License ::")]
    return ", ".join(named) or "see its dist-info"


def main():
    commit, _, source = read("SOURCE_COMMIT", "unknown").partition("\n")
    cuda = read("notices/CUDA_VERSION.txt", "unknown").replace("\n", " ")
    families = sorted(f for f in os.listdir(os.path.join(HERE, "lib", "silicon_families")) if f.endswith(".so"))
    print("# What this image holds, and each part's licence\n")
    print("> Written when the image was built, from what is installed in it. Not legal advice.\n")
    print("## SiLVANN\n")
    print("The release at commit `%s` of %s, under the GNU Affero General Public License, version 3 (`LICENSE`), "
          "with the additional permission to convey it linked with NVIDIA's CUDA runtime (`ATTRIBUTION.md` §1); "
          "`ATTRIBUTION.md` names the third-party code in its source. Its source is in this image, in `/opt/silvann` "
          "(`engine/`, `runtime/`, `lisp/`, the entry points), and at that commit of that repository.\n"
          % (commit.strip(), source.strip() or "its repository"))
    print("Built in this image from that source: the evaluator (`lib/silvann_engine_cpu*.so`) and the silicon "
          "families (%s).\n" % ", ".join("`lib/silicon_families/%s`" % f for f in families))
    if "libsilvann_nvidia_cuda12.so" in families:
        print("## NVIDIA's CUDA runtime\n")
        print("`libsilvann_nvidia_cuda12.so` holds NVIDIA's CUDA runtime, linked statically when it was compiled (%s). "
              "The runtime is NVIDIA's, distributed under the NVIDIA CUDA Toolkit End User License Agreement, which is in "
              "`notices/CUDA_EULA.txt`. The image holds no other part of the CUDA toolkit: the driver is the host's, "
              "handed to the container when it starts.\n" % cuda)
    rows = [r.split("\t") for r in subprocess.run(["dpkg-query", "-W", "-f", "${Package}\t${Version}\t${Maintainer}\n"],
                                                  capture_output=True, text=True, check=True).stdout.split("\n") if r]
    rocm = [r for r in rows if "amd.com" in r[2].lower()]
    if rocm:
        version = read("/opt/rocm/.info/version", "unknown").strip()
        print("## AMD's ROCm, its HIP runtime\n")
        print("`libsilvann_amd_rocm6_wave64.so` is linked against ROCm's HIP runtime (%s), installed from AMD's repository "
              "(repo.radeon.com) as the packages below. Each one's licence is in `/opt/rocm/share/doc/<component>/` "
              "(and `/usr/share/doc/<package>/` where it has one); their source is published by AMD at "
              "https://github.com/ROCm. The card's kernel driver is the host's.\n" % version)
        print("| package | version |\n|---|---|")
        for r in sorted(rocm):
            print("| %s | %s |" % (r[0], r[1]))
        print()
    print("## Ubuntu's packages\n")
    print("The system is Ubuntu 22.04. Each package's copyright and licence are in `/usr/share/doc/<package>/copyright`, "
          "and its source is in Ubuntu's archive at the version below.\n")
    print("| package | version |\n|---|---|")
    for r in sorted(r for r in rows if r not in rocm):
        print("| %s | %s |" % (r[0], r[1]))
    print("\n## Python packages\n")
    print("In `/opt/silvann/venv`, installed from PyPI by `requirements.txt` and what it requires. Each one's licence "
          "files are in its `.dist-info` folder under `venv/lib/python3.10/site-packages/`.\n")
    print("| package | version | licence |\n|---|---|---|")
    seen = set()
    for dist in sorted(md.distributions(), key=lambda d: d.metadata["Name"].lower()):
        name = dist.metadata["Name"]
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        print("| %s | %s | %s |" % (name, dist.version, licence(dist).replace("|", "/")))


if __name__ == "__main__":
    main()
