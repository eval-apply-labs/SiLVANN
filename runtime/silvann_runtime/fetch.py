"""A MODEL THAT IS NOT THERE YET — asked for on the command line, and fetched from Hugging Face.

⚖ *"if the bundled model is not there it asks it to download it from the command line... so it is idiot proof and i
can avoid bundling the model in the github code release"*. The release carries the code; a model folder is fetched the
first time it is named:
```
  ensure_model("models/silvann_tq_D8E4_qwen3.6_35b_a3b")     # there: answers the path · not there: asks, then downloads
```
The folder's name picks the repository (`KNOWN`), or `repo=` names one. A name says its widths: `D` the dense layer
matrices, `E` the routed experts — `D8E4` is 8 bits and 4. Nothing is downloaded without a yes; with no
terminal to ask on, it fails and names the command that fetches it.
"""
import os
import sys

from .model_folder import Refused

KNOWN = {
    "silvann_tq_D8E4_qwen3.6_35b_a3b": ("eval-apply/silvann_tq_D8E4_qwen3.6_35b_a3b", "20 GB"),
    "silvann_tq_D4E4_qwen3.6_35b_a3b": ("eval-apply/silvann_tq_D4E4_qwen3.6_35b_a3b", "19 GB"),
    "silvann_tq_D8E4_qwen3.5_122b_a10b": ("eval-apply/silvann_tq_D8E4_qwen3.5_122b_a10b", "67 GB"),
    "silvann_tq_D4E4_qwen3.5_122b_a10b": ("eval-apply/silvann_tq_D4E4_qwen3.5_122b_a10b", "65 GB"),
    "silvann_tq_D4E4_qwen3.5_397b_a17b": ("eval-apply/silvann_tq_D4E4_qwen3.5_397b_a17b", "207 GB"),
    "silvann_tq_D4_qwen3.8_27b": ("eval-apply/silvann_tq_D4_qwen3.8_27b", "16 GB"),
    "silvann_tq_D4E4_glm_5.3_flash": ("eval-apply/silvann_tq_D4E4_glm_5.3_flash", "162 GB"),
    "silvann_tq_D8E4_glm_5.3_flash": ("eval-apply/silvann_tq_D8E4_glm_5.3_flash", "166 GB"),
    "silvann_tq_D4_ministral_3_14b": ("eval-apply/silvann_tq_D4_ministral_3_14b", "7.4 GB"),
    "silvann_tq_D8_ministral_3_14b": ("eval-apply/silvann_tq_D8_ministral_3_14b", "13.5 GB"),
}


def present(path):
    return os.path.isfile(os.path.join(path, "pack.json")) and os.path.isfile(os.path.join(path, "config.json"))


def ensure_model(path, repo=None, ask=input, yes=False):
    """The model folder at `path`, downloaded first if it is not there and the answer is yes — or at once when `yes`,
    which is what `silvann_core.py download-model` is: the asking already done. A folder that is there without its
    program, `lisp/` or a file of it its family reads, is asked the same about that alone."""
    path = os.path.abspath(path)
    name = os.path.basename(path.rstrip(os.sep))
    if present(path):
        if _lacks_program(path):
            _program(path, repo or KNOWN.get(name, (None, None))[0], ask, yes)
        return path
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


def _lacks_program(path):
    """Whether the folder lacks `lisp/`, or a file of it its family's runtime reads (▶ its `PROGRAM`) — a folder fetched
    for an engine before that file was one of them."""
    import json
    from . import ARCHITECTURES
    lisp = os.path.join(path, "lisp")
    if not os.path.isdir(lisp):
        return True
    try:
        whole = json.load(open(os.path.join(path, "config.json")))
    except (OSError, ValueError):
        return False
    cls = ARCHITECTURES.get(whole.get("text_config", whole).get("model_type", ""))
    return cls is not None and any(not os.path.isfile(os.path.join(lisp, n + ".lisp")) for n in cls.PROGRAM)


def _program(path, repo, ask, yes):
    """A folder with its weights and without its program, or a file of it — `lisp/` — gets `lisp/` from the folder's
    repository, a few files of text. Nothing is fetched without a yes; a folder no repository is known for is left as
    it is, and the boot names what it lacks."""
    if repo is None:
        return
    if not yes:
        if not sys.stdin.isatty():
            return
        if ask("%s lacks lisp/ or a file of it — the model's program. Fetch it from huggingface.co/%s? [Y/n] "
               % (path, repo)).strip().lower() in ("n", "no"):
            return
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, local_dir=path, allow_patterns=["lisp/*"])


def _download(repo, path):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, local_dir=path, **_reported(os.environ.get("SILVANN_DOWNLOAD_PROGRESS")))
    if not present(path):
        raise Refused("the download of %s finished but %s is not a model folder" % (repo, path))
    return path


def _reported(report):
    """`snapshot_download`'s progress, counted into the file `report` names (`SILVANN_DOWNLOAD_PROGRESS`, which the UI's
    downloader sets) as JSON at most twice a second: `{bar: [count, total]}` for each of its bars, the bytes received
    among them. Hugging Face's client writes a file in blocks of a few hundred MB, so the bytes received are what moves
    on a slow connection. No file named: nothing is passed and the client's own bars are drawn."""
    if not report:
        return {}
    import json
    import threading
    import time
    from huggingface_hub.utils import tqdm as hf_tqdm
    counts, lock, last = {}, threading.Lock(), [0.0]

    def write(force):
        now = time.monotonic()
        if not force and now - last[0] < 0.5:
            return
        last[0] = now
        tmp = report + ".tmp"
        with open(tmp, "w") as f:
            json.dump(counts, f)
        os.replace(tmp, report)

    class Reported(hf_tqdm):
        # counted here as well as by tqdm, which counts nothing when it draws nothing (no terminal)
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._bar = str(kwargs.get("name") or kwargs.get("desc") or "bar")
            self._count = int(kwargs.get("initial") or 0)

        def update(self, n=1):
            with lock:
                self._count += int(n or 0)
                counts[self._bar] = [self._count, int(self.total or 0)]
                write(False)
            return super().update(n)

        def close(self):
            with lock:
                if hasattr(self, "_bar"):
                    counts[self._bar] = [self._count, int(self.total or 0)]
                    write(True)
            return super().close()

    return {"tqdm_class": Reported}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m silvann_runtime.fetch MODEL_FOLDER [REPO]")
    print(ensure_model(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
