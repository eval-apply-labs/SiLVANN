"""A MODEL THAT IS NOT THERE YET — asked for on the command line, and fetched from Hugging Face.

⚖ *"if the bundled model is not there it asks it to download it from the command line... so it is idiot proof and i
can avoid bundling the model in the github code release"*. The release carries the code; a model folder is fetched the
first time it is named:
```
  ensure_model("models/Qwen3.6-35B-A3B_silvann_tq_D8E4")     # there: answers the path · not there: asks, then downloads
```
The folder's name picks the repository (`KNOWN`), or `repo=` names one. A name says its widths: `D` the dense layer
matrices, `E` the routed experts — `D8E4` is 8 bits and 4. Nothing is downloaded without a yes; with no
terminal to ask on, it fails and names the command that fetches it.
"""
import os
import sys

from .model_folder import Refused

KNOWN = {
    "Qwen3.6-35B-A3B_silvann_tq_D8E4": ("eval-apply/Qwen3.6-35B-A3B_silvann_tq_D8E4", "20 GB"),
    "Qwen3.6-35B-A3B_silvann_tq_D4E4": ("eval-apply/Qwen3.6-35B-A3B_silvann_tq_D4E4", "19 GB"),
    "Qwen3.5-122B-A10B_silvann_tq_D8E4": ("eval-apply/Qwen3.5-122B-A10B_silvann_tq_D8E4", "67 GB"),
    "Qwen3.5-122B-A10B_silvann_tq_D4E4": ("eval-apply/Qwen3.5-122B-A10B_silvann_tq_D4E4", "65 GB"),
    "Qwen3.8-27B_silvann_tq_D4": ("eval-apply/Qwen3.8-27B_silvann_tq_D4", "16 GB"),
    "GLM-5.3-Flash_silvann_tq_D4E4": ("eval-apply/GLM-5.3-Flash_silvann_tq_D4E4", "151 GB"),
}


def present(path):
    return os.path.isfile(os.path.join(path, "pack.json")) and os.path.isfile(os.path.join(path, "config.json"))


def ensure_model(path, repo=None, ask=input, yes=False):
    """The model folder at `path`, downloaded first if it is not there and the answer is yes — or at once when `yes`,
    which is what `silvann_core.py download-model` is: the asking already done."""
    path = os.path.abspath(path)
    if present(path):
        return path
    name = os.path.basename(path.rstrip(os.sep))
    repo, size = (repo, "several GB") if repo else KNOWN.get(name, (None, None))
    if repo is None:
        raise Refused("%s is not a model folder, and %r is not a model this release knows how to fetch — known: %s"
                      % (path, name, ", ".join(sorted(KNOWN))))
    command = "silvann_core.py download-model %s" % name
    if yes:
        return _download(repo, path)
    if not sys.stdin.isatty():
        raise Refused("%s is not here. Fetch it (%s, from huggingface.co/%s) with:\n    %s" % (name, size, repo, command))
    print("The model %s is not in %s." % (name, os.path.dirname(path)))
    if ask("Download it now from huggingface.co/%s (%s)? [y/N] " % (repo, size)).strip().lower() not in ("y", "yes"):
        raise Refused("not downloaded — run `%s` when you want it" % command)
    return _download(repo, path)


def _download(repo, path):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, local_dir=path)
    if not present(path):
        raise Refused("the download of %s finished but %s is not a model folder" % (repo, path))
    return path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m silvann_runtime.fetch MODEL_FOLDER [REPO]")
    print(ensure_model(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
