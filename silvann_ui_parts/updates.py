"""This copy of SiLVANN and the release after it: the version it runs, what GitHub has when it is asked, and the way to
it for this kind of install. A portable app is updated by its own updater, a program beside it that this only names."""
import asyncio
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

import httpx
from nicegui import ui

from .local import REPO_ROOT
from .part import Part
from .sign_in import _runtime_module

# ⭐ THE VERSION, AND THE ONE AFTER IT (NN-203; ⚖ 2026-11-22 *"a version checker in the options: we could have an in-ui
#   version number shower that also checks on the github if there is a new release and tells you to update either by
#   clicking update if on the portable app ... or with the one-liner for containers"*).
# ⛔ GITHUB IS ASKED ONLY WHEN A PERSON PRESSES THE BUTTON (⚖ 2026-11-22 *"just make the check a button, i dont want to
#   ping stuff on the web from a local app... that would be self doxing"*): nothing here reaches the network by itself
#   — not when Options opens, not on a timer, not when SiLVANN starts — and the button asks one question and fetches
#   nothing.
# ⭐ A PORTABLE APP'S UPDATE IS ITS UPDATER'S (⚖ 2026-11-22 *"we can have a update folder alongside windows, data and
#   linux. this way we can click update/windows.exe and it does the update for you, and we can also make it update
#   itself if needed... this is only for the portable version btw, the container one still uses the container update
#   script"*): a program cannot rename the folder it runs from, so the app names the updater and does one thing for
#   it — the updater a newer version brought is put in the old one's place as the app starts (▶ `settle`).
LATEST = "https://api.github.com/repos/eval-apply-labs/SiLVANN/releases/latest"
ASSET = "silvann_portable_x64.zip"                    # both systems' parts and the updater
CONTAINER_LINE = "curl -fsSL https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/docker/setup.sh | bash -s -- update"
UPDATER = "update\\windows.exe" if os.name == "nt" else "update/linux.sh"
NOT_ASKED = ("GitHub is asked only when you press the button: until then nothing here says whether a newer release "
             "is out, and nothing leaves this machine.")


def running() -> str:
    """The version this copy runs: the runtime's `RELEASE`."""
    return _runtime_module("version").RELEASE


def numbers(version: str) -> tuple:
    """A version as numbers, to set two in order: "1.4.2" and the tag "rel1.4.2" are (1, 4, 2)."""
    return tuple(int(n) for n in re.findall(r"\d+", version))


def newer(a: str, than: str) -> bool:
    return numbers(a) > numbers(than)


def kind(root: Path = REPO_ROOT) -> str:
    """How this copy is updated: "portable" — a portable app's part, run by the Python it carries; "container";
    "install" — install.sh's or install.ps1's clone, `silvann` beside it; "source" — a tree of its own."""
    carried = root / "python"
    if carried.is_dir() and carried.resolve() in Path(sys.executable).resolve().parents:
        return "portable"
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        return "container"
    if (root / ".git").exists() and any((root.parent / "bin" / name).exists() for name in ("silvann", "silvann.cmd", "silvann.ps1")):
        return "install"
    return "source"


def release_of(said: dict) -> Optional[dict]:
    """What a release says of itself, from GitHub's answer: its version, its page, its portable zip and the zip's
    size; None where the answer names no version."""
    tag = str(said.get("tag_name") or "")
    if not numbers(tag):
        return None
    asset = next((a for a in said.get("assets") or [] if a.get("name") == ASSET), {})
    return {"version": ".".join(str(n) for n in numbers(tag)), "page": said.get("html_url") or "",
            "zip": asset.get("browser_download_url") or "", "bytes": int(asset.get("size") or 0)}


def latest() -> Optional[dict]:
    """GitHub's latest release of SiLVANN (▶ `release_of`), asked now; None where GitHub does not answer. Called by the
    button alone."""
    try:
        r = httpx.get(LATEST, timeout=8, follow_redirects=True,
                      headers={"Accept": "application/vnd.github+json", "User-Agent": "SiLVANN"})
        return release_of(r.json()) if r.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


def put_updater(app: Path) -> Optional[bool]:
    """`update_updated`, the updater an update brought, put in `update`'s place in a portable app's folder. The
    updater leaves it there: on Windows it runs from `update` and cannot rename it. None where nothing waits, True
    where it is in place, False where `update` is in use and stays — tried again at the next start."""
    new, old, was = app / "update_updated", app / "update", app / "update_before"
    if not new.is_dir():
        return None
    try:
        if was.exists():
            shutil.rmtree(was)
        if old.exists():
            old.rename(was)
    except OSError:
        return False
    try:
        new.rename(old)
    except OSError:
        if was.exists():
            was.rename(old)
        return False
    shutil.rmtree(was, ignore_errors=True)
    return True


def settle() -> None:
    """A portable app as its UI starts (▶ `put_updater`); nothing for any other kind of install, and nothing that
    reaches the network."""
    if kind() != "portable":
        return
    put = put_updater(REPO_ROOT.parent)
    if put is not None:
        print("[ui] the updater this version brought %s" % ("is in place" if put else "waits: the update folder is in use"),
              flush=True)


def how(release: Optional[dict], now: str, of: str, updater: Optional[bool] = None) -> str:
    """What Options says under the version once GitHub was asked: it did not answer, nothing newer, or the way to the
    newer release for this kind of install (`of`, ▶ `kind`). `updater` says whether a portable app's folder has its
    updater, read from the folder where it is not said."""
    if release is None:
        return "GitHub was not reached: whether a newer release is out is not known."
    if not newer(release["version"], now):
        return "This is the latest release."
    there = "SiLVANN %s is out (%s). " % (release["version"], release["page"])
    if of == "portable":
        if not ((REPO_ROOT.parent / "update").is_dir() if updater is None else updater):
            return there + ("This folder has no updater: unzip the release's %s over it once, and from then on %s "
                            "updates it." % (ASSET, UPDATER))
        return there + ("Close SiLVANN and open %s in SiLVANN's folder: it downloads %s and puts it in this "
                        "version's place. The models, the conversations and the accounts stay."
                        % (UPDATER, release["version"]))
    if of == "container":
        return there + "On the machine that runs the container: " + CONTAINER_LINE
    if of == "install":
        return there + "In a terminal on this machine: silvann update"
    return there + "This copy is a source tree: update it as it was made."


class Updates(Part):
    """Options' version block: the version, and the button that asks GitHub."""

    def build(self):
        """The block, in Options, an admin's: the version, and the button — nothing is asked of GitHub until it is
        pressed."""
        self.release = None
        with ui.column().classes("w-full gap-1 mt-4"):
            ui.separator()
            ui.label("Version").classes("font-bold")
            with ui.row().classes("items-center gap-3"):
                self.version_lbl = ui.label("SiLVANN %s" % running()).classes("font-mono")
                self.check_btn = ui.button("check for a newer release", icon="refresh", on_click=self.check).props("dense flat")
            self.how_lbl = ui.label(NOT_ASKED).classes("text-sm whitespace-pre-wrap max-w-3xl")

    async def check(self):
        """GitHub asked, by the button, and the block's words set by its answer."""
        self.check_btn.disable()
        try:
            self.release = await asyncio.to_thread(latest)
            self.how_lbl.text = how(self.release, running(), kind())
        finally:
            self.check_btn.enable()
