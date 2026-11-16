#!/usr/bin/env python3
"""The portable app, one archive for both systems (NN-194): silvann_portable_x64.zip, which unzips to

    SiLVANN/windows/   SiLVANN.exe and Windows' part (build_portable.ps1's silvann_portable_windows_part.zip)
    SiLVANN/linux/     SiLVANN.sh and Linux's part (build_portable.sh's silvann_portable_linux_part.tar.gz)
    SiLVANN/data/      the models, the conversations and the logs, both systems'

⚖ 2026-11-16 *"i'd like to have a single portable app, and when unzipped it should have windows, linux and data
folders. my point is that i would like to have one usb key and being able to move it across systems"* · *"lets keep
silvann.exe under windows and silvann.sh under linux"*. Each part keeps its own path inside SiLVANN/; Linux's files keep
their permissions in the zip (its programs executable where the unzip sets them, as Linux's own unzip does), a hard
link — the tar's form of one file the standalone Python names twice (`python3` and `python3.12`) — written as a file of
its own, as a zip and exFAT hold them; a symbolic link is refused, and two paths that differ only in case, since the
drives both systems write (exFAT, NTFS) cannot hold both. Run by
the release's job (.github/workflows/windows-libraries.yml, `portable_app`):

    python3 scripts/portable/pack_portable.py WINDOWS_PART.zip LINUX_PART.tar.gz OUT.zip
"""
import stat
import sys
import tarfile
import time
import zipfile

PARTS = {"windows": "SiLVANN/windows/", "linux": "SiLVANN/linux/"}


def pack(windows, linux, out):
    seen = {}

    def place(z, name, data, mode, when):
        folded = name.lower()
        if folded in seen:
            sys.exit("%s and %s differ only in case: one drive for both systems cannot hold both" % (seen[folded], name))
        seen[folded] = name
        info = zipfile.ZipInfo(name, date_time=when)
        info.create_system = 3                        # Unix: the permissions below are read by the unzip
        info.external_attr = (stat.S_IFREG | mode) << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, data, compresslevel=6)

    count = {"windows": 0, "linux": 0}
    with zipfile.ZipFile(out, "w", allowZip64=True) as z:
        with zipfile.ZipFile(windows) as w:
            for i in w.infolist():
                name = i.filename.replace("\\", "/")
                if name.endswith("/"):
                    continue
                if not name.startswith(PARTS["windows"]):
                    sys.exit("%s in Windows' part is not under %s" % (name, PARTS["windows"]))
                place(z, name, w.read(i), 0o644, i.date_time)
                count["windows"] += 1
        with tarfile.open(linux) as t:
            for m in t.getmembers():
                name = m.name[2:] if m.name.startswith("./") else m.name
                if m.isdir():
                    continue
                if not (m.isfile() or m.islnk()):           # a hard link is read as the file it names
                    sys.exit("%s in Linux's part is not a file (a symbolic link?): a drive for both systems holds none" % name)
                if not name.startswith(PARTS["linux"]):
                    sys.exit("%s in Linux's part is not under %s" % (name, PARTS["linux"]))
                place(z, name, t.extractfile(m).read(), m.mode & 0o777, time.localtime(m.mtime)[:6])
                count["linux"] += 1
        for part in ("windows/SiLVANN.exe", "linux/SiLVANN.sh"):
            if "silvann/" + part.lower() not in seen:
                sys.exit("no SiLVANN/%s: the part that holds it is not whole" % part)
        data = zipfile.ZipInfo("SiLVANN/data/", date_time=time.localtime()[:6])
        data.create_system = 3
        data.external_attr = ((stat.S_IFDIR | 0o755) << 16) | 0x10    # and MS-DOS's directory bit
        z.writestr(data, b"")
    print("%s: Windows' part %d files, Linux's %d, and data/" % (out, count["windows"], count["linux"]))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    pack(*sys.argv[1:])
