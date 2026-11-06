"""A MODEL THAT IS NOT THERE YET — asked for on the command line, and fetched from Hugging Face.

⚖ *"if the bundled model is not there it asks it to download it from the command line... so it is idiot proof and i
can avoid bundling the model in the github code release"*. The release carries the code; a model folder is fetched the
first time it is named:
```
  ensure_model("models/silvann_tq_D8E4_qwen3.6_35b_a3b")     # there: answers the path · not there: asks, then downloads
```
The folder's name picks the repository (`KNOWN`), or `repo=` names one. A name says its widths: `D` the dense layer
matrices, `E` the routed experts — `D8E4` is 8 bits and 4. No weights are downloaded without a yes; with no
terminal to ask on, it fails and names the command that fetches it. A folder that is there is updated by what changed
(`differences`, `update`: `silvann_core.py check-model` and `download-model`, the UI's "check for updates"), and a
boot fetches this release's program when the pack's is not the folder's — a few files of text.
"""
import glob
import json
import os
import re
import sys

from .model_folder import DEV, Refused, choose, newer, program_dir
from .version import RELEASE, parse

# a pack's name: its repository, its size, and the model it was made from — the makers' own repository, whose license
#   the pack carries and whose page the UI links to beside the pack's
KNOWN = {
    "silvann_tq_D8E4_qwen3.6_35b_a3b": ("eval-apply/silvann_tq_D8E4_qwen3.6_35b_a3b", "20 GB", "Qwen/Qwen3.6-35B-A3B"),
    "silvann_tq_D8E4_qwen3.5_122b_a10b": ("eval-apply/silvann_tq_D8E4_qwen3.5_122b_a10b", "67 GB", "Qwen/Qwen3.5-122B-A10B"),
    "silvann_tq_D4_qwen3.5_4b": ("eval-apply/silvann_tq_D4_qwen3.5_4b", "3.5 GB", "Qwen/Qwen3.5-4B"),
    "silvann_tq_D8_qwen3.5_4b": ("eval-apply/silvann_tq_D8_qwen3.5_4b", "5.3 GB", "Qwen/Qwen3.5-4B"),
    "silvann_tq_D4_qwen3.8_27b": ("eval-apply/silvann_tq_D4_qwen3.8_27b", "16 GB", "Qwen/Qwen3.8-27B"),
    "silvann_tq_D8E4_glm_5.3_flash": ("eval-apply/silvann_tq_D8E4_glm_5.3_flash", "166 GB", "zai-org/GLM-5.3-Flash"),
    "silvann_tq_D8E4_deepseek_v4.1_flash_1017": ("eval-apply/silvann_tq_D8E4_deepseek_v4.1_flash_1017", "492 GB",
                                                 "deepseek-ai/DeepSeek-V4.1-Flash-1017"),
    "silvann_tq_D4_ministral_3_14b": ("eval-apply/silvann_tq_D4_ministral_3_14b", "7.4 GB",
                                      "mistralai/Ministral-3-14B-Instruct-2512"),
    "silvann_tq_D8_ministral_3_14b": ("eval-apply/silvann_tq_D8_ministral_3_14b", "13.5 GB",
                                      "mistralai/Ministral-3-14B-Instruct-2512"),
    "silvann_tq_D8_qwen_image_2.1": ("eval-apply/silvann_tq_D8_qwen_image_2.1", "18 GB", "Qwen/Qwen-Image-2.1"),
}


def present(path):
    """A model folder here: a text model's (its `config.json` and its pack's `pack.json`), or an image pack (diffusers'
    `model_index.json`, its transformer packed — ▶ `qwen_image.image_pack`)."""
    from .qwen_image import image_pack
    return (os.path.isfile(os.path.join(path, "pack.json")) and os.path.isfile(os.path.join(path, "config.json"))
            or image_pack(path))


def programs(path):
    """The program files a model folder carries: [(its folder, the names it reads)] — a text model's `lisp/`, its
    family's runtime's `PROGRAM`; an image pack's `lisp/` (the transformer's) and its encoder's `text_encoder/lisp/`
    (the encoder's family's). [] for a folder whose family this release does not know."""
    import json
    from . import ARCHITECTURES
    from .qwen_image import QwenImage, image_pack
    if image_pack(path):
        return [(os.path.join(path, "lisp"), QwenImage.PROGRAM)] + programs(os.path.join(path, "text_encoder"))
    try:
        whole = json.load(open(os.path.join(path, "config.json")))
    except (OSError, ValueError):
        return []
    cls = ARCHITECTURES.get(whole.get("text_config", whole).get("model_type", ""))
    return [(os.path.join(path, "lisp"), cls.PROGRAM)] if cls is not None else []


def ensure_model(path, repo=None, ask=input, yes=False):
    """The model folder at `path`, downloaded first if it is not there and the answer is yes — or at once when `yes`,
    which is what `silvann_core.py download-model` is: the asking already done. A folder that is there gets, when `yes`
    asks for it, what it lacks of its pack or holds otherwise (▶ `differences`, `update`) — an update fetches the files
    that changed, not the pack again; and at every boot this release's program, when the pack's is not the folder's
    (▶ `update_program`)."""
    path = os.path.abspath(path)
    name = os.path.basename(path.rstrip(os.sep))
    if present(path):
        repo = repo or KNOWN.get(name, (None, None))[0]
        if repo is None:
            return path
        if yes:
            d = differences(path, repo)
            update(path, repo, [f for f, _ in d["files"]])
            print(said(d, fetched=True), flush=True)
        else:
            update_program(path, repo)
        return path
    repo, size = (repo, "several GB") if repo else KNOWN.get(name, (None, None))[:2]
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


# ⭐ A PACK'S VERSION, `version.json` at its root: {"version": n, "release": "X.Y.Z", "updated": "YYYY-MM-DD"} —
#   `version` the pack's own count, raised by every upload that changes its files (scripts/pack_version.py,
#   scripts/model_lisp.py --upload), `release` the newest SiLVANN release it has a program for. ⚖ 2026-11-06 *"i would
#   like to simply add a version file in the pack so we can simply do a check there and know if there is something to
#   update or not, so the ui can have "download" "update" or "done""*. A folder's own is the pack's as it stood when the
#   folder's download or its last update finished — fetched last, so one cut short leaves the older one, and says
#   update. What an update fetches is still decided file by file (▶ `differences`).
VERSION_FILE = "version.json"


def folder_version(path):
    """The folder's `version.json`, or None when it has none (a folder from before the packs had one)."""
    try:
        with open(os.path.join(path, VERSION_FILE)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def pack_version(repo):
    """The pack's `version.json` on Hugging Face, or None when it has none; an error when the Hub cannot be asked."""
    from huggingface_hub import hf_hub_download
    try:
        with open(hf_hub_download(repo, VERSION_FILE)) as f:
            return json.load(f)
    except Exception as ex:
        if type(ex).__name__ in ("EntryNotFoundError", "RemoteEntryNotFoundError"):
            return None
        raise


def state(path, repo, release=None):
    """What the UI says of a model: {"state": "download" (not here) · "update" (the pack's version is above the
    folder's, or the folder has none) · "done" · "unknown" (the pack has no version, or the Hub cannot be asked — "why"),
    "here", "pack": the two versions, "newer": the pack has a program for a later release than this one}."""
    if not present(path):
        return {"state": "download"}
    here = folder_version(path)
    try:
        pack = pack_version(repo)
    except Exception as ex:
        return {"state": "unknown", "here": here, "why": "not checked: %s" % ((str(ex).splitlines() or ["?"])[0][:200])}
    if pack is None:
        return {"state": "unknown", "here": here, "why": "the pack has no %s" % VERSION_FILE}
    later = parse(pack.get("release") or "") or (0, 0, 0)
    out = {"here": here, "pack": pack, "newer": later > parse(release or RELEASE)}
    if here is None or int(here.get("version", 0)) < int(pack.get("version", 0)):
        return dict(out, state="update")
    if pack.get("release") and later <= parse(release or RELEASE) and not holds_program(path, pack, release):
        return dict(out, state="update", why="this release's program")    # a folder updated under an older release
    return dict(out, state="done")


def holds_program(path, pack, release=None):
    """Whether the folder holds the program this release reads of the pack whose version is `pack`: its `lisp/dev/`
    (a program of its own), or the pack's newest — `pack["release"]`, at or below this release — in each of its
    `lisp/`. A pack whose newest program is a later release's, or that names none, is not known so: its programs are
    listed (▶ `update_program`). A folder updated under an older release is at the pack's version without the program
    a newer one reads — measurements/2026-11-06_program_update_node03.md."""
    top = pack.get("release")
    if not top or parse(top) is None or parse(top) > parse(release or RELEASE):
        return False
    for lisp, _ in programs(path):
        held = program_dir(lisp, release)
        if os.path.basename(held) not in (DEV, "v" + top):
            return False
    return True


# ⭐ AN UPDATE FETCHES WHAT CHANGED. ⚖ 2026-11-06 *"make it so that the ui can check if there is a diff and it allows you
#   to download just the missing files, so an update does not need to redownload everything"*. `differences` holds the
#   folder against the pack's file list (one listing of the repository): a file is the pack's when it is there at the
#   pack's size and — a small one — has its git blob hash, or — a large one (LFS) — Hugging Face's client recorded it as
#   that file when it fetched it (`.cache/huggingface/download/`, a record it drops when the file changes after) or,
#   with no record — a folder copied in, or packed here — its SHA-256 is the pack's, read once and then recorded. A file
#   that is a symbolic link, or under one, is a development box's and is left as it is; a file here the pack does not
#   have stays. Of the pack's programs only the one this release reads counts (▶ `model_folder.choose`). `update`
#   fetches the files that differ and nothing else. (The client's own dry run is not the check: it answers "nothing to
#   download" for a file it would copy from its cache over the folder's — `MEASURED` 2026-11-06 node03, the 35B's
#   folder whose lisp/ links into the tree came back "up to date": measurements/2026-11-06_lisp_versions_node03.md.)
PROGRAMS = ["lisp/*", "*/lisp/*"]           # a pack's program files: its lisp/, and an image pack's encoder's
_LISP = re.compile(r"^((?:[^/]+/)*?lisp/)(.+)$")


def programs_of(names, release=None):
    """Of a pack's file names: (the names that count for this release — every file that is not a program's, and of each
    `lisp/` the files of the program this release reads: its `v<X.Y.Z>/` at or below the release, else the files in
    `lisp/` itself — and the folders of programs for a later release, e.g. `lisp/v0.25.0`)."""
    roots = {}
    for n in names:
        m = _LISP.match(n)
        if m:
            roots.setdefault(m.group(1), []).append(m.group(2))
    keep, later = [n for n in names if not _LISP.match(n)], []
    for root, rest in sorted(roots.items()):
        subs = sorted({r.split("/", 1)[0] for r in rest if "/" in r})
        chosen = choose(subs, release)
        keep += [root + r for r in rest if (r.startswith(chosen + "/") if chosen else "/" not in r)]
        later += [root + v for v in newer(subs, release)]
    return set(keep), later


def differences(path, repo, programs_only=False, release=None):
    """What `update` fetches into the model folder at `path` from its pack `repo` (▶ the note above): {"repo", "commit",
    "files": [(name, bytes)] — those the folder lacks or holds otherwise —, "bytes", "newer": [a later release's program
    folder], "linked": [a file here that is a symbolic link or under one, left as it is], "read": the bytes hashed}.
    `programs_only`: the program's files alone, listed from the pack's lisp/ folders."""
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile
    api = HfApi()
    path = os.path.abspath(path)
    commit = api.model_info(repo).sha
    if programs_only:
        tree = []
        for lisp, _ in programs(path):
            try:
                tree += list(api.list_repo_tree(repo, path_in_repo=os.path.relpath(lisp, path).replace(os.sep, "/"),
                                                recursive=True, revision=commit))
            except Exception as ex:          # a pack without that lisp/: nothing of it to fetch
                if type(ex).__name__ != "EntryNotFoundError":
                    raise
    else:
        tree = list(api.list_repo_tree(repo, recursive=True, revision=commit))
    tree = [f for f in tree if isinstance(f, RepoFile)]
    keep, later = programs_of([f.path for f in tree], release)
    files, linked, hashed = [], [], []
    for f in tree:
        if f.path not in keep:
            continue
        local = os.path.join(path, *f.path.split("/"))
        if _linked(path, f.path):
            linked.append(f.path)
        elif not os.path.isfile(local) or os.path.getsize(local) != f.size:
            files.append((f.path, f.size))
        elif f.lfs is None:
            if _blob(local) != f.blob_id:
                files.append((f.path, f.size))
        elif _recorded(path, f.path) != f.lfs.sha256:
            hashed.append(f)
    if hashed:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max(1, min(8, os.cpu_count() or 1))) as pool:
            sums = list(pool.map(lambda f: _sha256(os.path.join(path, *f.path.split("/"))), hashed))
        for f, digest in zip(hashed, sums):
            if digest == f.lfs.sha256:
                _record(path, f.path, commit, digest)
            else:
                files.append((f.path, f.size))
    return {"repo": repo, "commit": commit, "files": sorted(files), "bytes": sum(b for _, b in files),
            "newer": later, "linked": sorted(linked), "read": sum(f.size for f in hashed)}


def update(path, repo, files, quiet=False):
    """The files named — `differences`' — fetched from the pack `repo` into the folder at `path`, as the pack holds
    them now; every other file as it is."""
    if not files:
        return path
    from huggingface_hub import snapshot_download
    rest = [f for f in files if f != VERSION_FILE]
    if rest:
        snapshot_download(repo_id=repo, local_dir=os.path.abspath(path), allow_patterns=[glob.escape(f) for f in rest],
                          force_download=True, **({"tqdm_class": _quiet()} if quiet else
                                                  _reported(os.environ.get("SILVANN_DOWNLOAD_PROGRESS"))))
    if VERSION_FILE in files:               # last: the folder says the pack's version once it holds the pack's files
        snapshot_download(repo_id=repo, local_dir=os.path.abspath(path), allow_patterns=[VERSION_FILE],
                          force_download=True, **({"tqdm_class": _quiet()} if quiet else {}))
    return path


def _linked(path, name):
    """Whether the file `name` of the folder at `path` is a symbolic link, or under one."""
    parts = name.split("/")
    return any(os.path.islink(os.path.join(path, *parts[:i + 1])) for i in range(len(parts)))


def _blob(local):
    """A file's git blob hash — what the Hub lists for a file it keeps in git."""
    import hashlib
    with open(local, "rb") as f:
        data = f.read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _sha256(local):
    import hashlib
    h = hashlib.sha256()
    with open(local, "rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _recorded(path, name):
    """The hash Hugging Face's client recorded for the file when it fetched it, while the file is unchanged since; None
    when it has no record (or one this client cannot read)."""
    try:
        from pathlib import Path
        from huggingface_hub._local_folder import read_download_metadata
        meta = read_download_metadata(Path(path), name)
        return meta.etag if meta is not None else None
    except Exception:
        return None


def _record(path, name, commit, digest):
    """The client's record that the file is the pack's at `commit`, so the next check reads it no more."""
    try:
        from pathlib import Path
        from huggingface_hub._local_folder import write_download_metadata
        write_download_metadata(Path(path), name, commit, digest)
    except Exception:
        pass


def size(b):
    """Bytes as a person reads them: GB, MB or KB."""
    return "%.2f GB" % (b / 1e9) if b >= 1e9 else "%.1f MB" % (b / 1e6) if b >= 1e6 else "%.0f KB" % (b / 1e3)


def said(d, fetched=False):
    """`differences`' answer as a line: what differs (or was fetched), the links left, a later release's program."""
    n = len(d["files"])
    line = (("fetched %d file%s, %s" if fetched else "%d file%s to fetch, %s") % (n, "" if n == 1 else "s", size(d["bytes"]))
            if n else "up to date with huggingface.co/%s" % d["repo"])
    if d.get("read"):
        line += "; %s read to compare, which the next check does not read again" % size(d["read"])
    if d["linked"]:
        line += "; %d linked here, left as they are (%s)" % (len(d["linked"]), ", ".join(d["linked"][:3])
                                                              + (" …" if len(d["linked"]) > 3 else ""))
    if d["newer"]:
        line += "; the pack has a program for a later release (%s) — a newer SiLVANN is out" % ", ".join(d["newer"])
    return line


def update_program(path, repo, say=True):
    """This release's program from the pack where the folder's is not it — at a boot, and for every model here after
    an update of SiLVANN (`silvann_core.py update-programs`): a few files of text, fetched without asking, the folder
    being the pack's and the program what its weights run. A file the folder lacks is fetched, and one the pack changed
    while the folder's is as it was fetched (Hugging Face's client's record of it); one changed here is kept, and said —
    `silvann download` fetches the pack's over it. Nothing is asked of the pack beyond its version when the folder is at
    it and holds its program for this release (▶ `holds_program`); a development box's `lisp/dev/` is left as it is; a
    pack that cannot be reached (offline, `HF_HUB_OFFLINE=1`) leaves the folder's own program, and the boot reads it.
    Answers {"fetched", "kept", "newer", "why"} — `why` when nothing was asked or fetched — and says it when `say`."""
    def told(line):
        if say:
            print(line, flush=True)
    out = {"fetched": [], "kept": [], "newer": [], "why": None}
    if any(os.path.isdir(os.path.join(lisp, DEV)) for lisp, _ in programs(path)):
        out["why"] = "lisp/dev/, a program of its own"
        return out
    try:
        here, pack = folder_version(path), pack_version(repo)
        if (here is not None and pack is not None and int(here.get("version", 0)) >= int(pack.get("version", 0))
                and holds_program(path, pack)):
            out["why"] = "at the pack's version %s, this release's program held" % pack.get("version")
            return out
        d = differences(path, repo, programs_only=True)
    except Exception as ex:                 # offline, or the pack unreachable: the folder's own program runs
        out["why"] = "the pack's program was not checked (%s): the folder's own runs" % (str(ex).splitlines() or ["?"])[0]
        told(out["why"])
        return out
    out["fetched"] = [f for f, _ in d["files"] if not os.path.exists(os.path.join(path, *f.split("/")))
                      or _recorded(path, f)]
    out["kept"] = [f for f, _ in d["files"] if f not in out["fetched"]]
    out["newer"] = d["newer"]
    if out["fetched"]:
        update(path, repo, out["fetched"], quiet=True)
        told("this release's program from huggingface.co/%s: %s" % (repo, ", ".join(out["fetched"])))
    if out["kept"]:
        told("the program's %s changed here, kept — `silvann download %s` fetches the pack's"
             % (", ".join(out["kept"]), os.path.basename(path.rstrip(os.sep))))
    if out["newer"]:
        told("the pack has a program for a later release (%s) — a newer SiLVANN is out" % ", ".join(out["newer"]))
    if not (out["fetched"] or out["kept"]):
        out["why"] = "its program is this release's" + ("" if not d["linked"] else ", %d linked here" % len(d["linked"]))
    return out


def _quiet():
    """A progress bar that draws nothing — a program's few files, fetched beside a boot's or an install's lines."""
    from huggingface_hub.utils import tqdm as hf_tqdm

    class Quiet(hf_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
    return Quiet


def _download(repo, path):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, local_dir=path, ignore_patterns=[VERSION_FILE],
                      **_reported(os.environ.get("SILVANN_DOWNLOAD_PROGRESS")))
    snapshot_download(repo_id=repo, local_dir=path, allow_patterns=[VERSION_FILE])     # last, as `update` fetches it
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
