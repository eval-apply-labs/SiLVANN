"""A FILE AS ANY MODEL READS IT — a document's text, put in the message that names it, or kept as a file the message names.

    doc = read_document(data, "report.docx")    # .name, .text, .kind
    text = doc.framed() + user_text             # <file name="report.docx"> … </file> before the message
    path = doc.saved(workspace)                 # its text under the workspace's attached/, for the model's tool
    text = doc.named(path) + user_text          # where it is and how it is looked into — and not its text

⭐ A DOCUMENT IS NOT READ WHOLE UNLESS IT IS SMALL OR THE USER ASKS (⚖ 2026-11-20: *"maybe i want to send him a whole
encyclopedia to say look at the definition of barricade, it would be good if uploaded documents would not trigger a full
read unless explicitly asked to do so"*). `carried` says which way a message takes one: its text where it is at most
`INLINE` tokens and fits, or where the user asked for the whole, or where the conversation has no tool to read a file
with; else kept in the account's workspace and named, for `(grep …)` and `(read …)` to look into.

What it reads: a Word document (`.docx`, the zip of XML Word writes — its paragraphs, tables a row a line with tabs between
cells, read with the standard library); a PDF through pypdf, where it is installed (each page's text, a blank line between
pages); and any file whose bytes are text (UTF-8, or Latin-1 failing that): Markdown, CSV, JSON, code. A picture is not
read here (▶ `vision.py`, `vision_glm.py`), nor an old binary Word document (`.doc`).
"""
import io
import os
import re
import zipfile
from xml.etree import ElementTree

from .model_folder import Refused

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
TEXT_SUFFIXES = (".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".xml", ".html", ".htm", ".yaml", ".yml",
                 ".toml", ".ini", ".log", ".py", ".js", ".ts", ".c", ".h", ".cpp", ".cu", ".cuh", ".java", ".go", ".rs",
                 ".sh", ".sql", ".tex", ".rst", ".lisp")


INLINE = 4000                     # tokens: a document at most this long goes in its message — about what one (read …)
#                                   answers, so reading it there costs no more than a call would
ATTACHED = "attached"             # the folder of a workspace where the documents a message names are kept
HEAD = 600                        # the characters of a document's beginning its line in the message shows
CHARS = 12                        # a text longer than this many characters a token of its room is taken to be past it:
#                                   only that much of it is encoded, so a book costs what a page does to place


def carried(tokens, share, tool, whole):
    """How a message carries a document of `tokens` tokens where `share` is the room it may take: "text", all of it in
    the message; "cut", what fits of it; "named", kept as a file and named. `tool`: the conversation has a tool that
    reads a file. `whole`: the user asked for all of it in the message."""
    if whole or not tool:
        return "text" if tokens <= share else "cut"
    return "text" if tokens <= min(INLINE, share) else "named"


def fitted(doc, share, tool, whole, encode, decode):
    """How a message carries `doc`, and the text it carries: ("text", its text), ("cut", what fits `share` tokens of it)
    or ("named", None) — `carried`'s rule, with `encode` and `decode` the chat's. No more of a long text is encoded
    than could fit: the rest is past the room by its length (`CHARS`)."""
    limit = max(0, share if whole or not tool else min(INLINE, share))
    part = doc.text if len(doc.text) <= limit * CHARS else doc.text[:limit * CHARS]
    ids = encode(part)
    tokens = len(ids) if len(part) == len(doc.text) else max(len(ids), limit + 1)
    how = carried(tokens, share, tool, whole)
    return how, (doc.text if how == "text" else decode(ids[:limit]) if how == "cut" else None)


class Document:
    def __init__(self, name, text, kind):
        self.name, self.text, self.kind = name, text, kind

    def framed(self, text=None):
        """The document's text (or `text`, a part of it) as the message carries it."""
        return '<file name="%s">\n%s\n</file>\n' % (self.name.replace('"', "'"), (self.text if text is None else text).strip())

    def file_name(self):
        """The name its text is kept under: its own, without a folder and with nothing a file system refuses — and
        `.txt` after a Word document's or a PDF's, whose text it is and not the file."""
        base = os.path.basename((self.name or "").replace("\\", "/")).strip()
        base = re.sub(r"[^\w .()+,=@\[\]-]", "_", base).strip(" .") or "file"
        return base[:120] + (".txt" if self.kind in ("docx", "pdf") else "")

    def saved(self, workspace):
        """This document's text as a file in `workspace`'s `ATTACHED` folder, and its path. Its lines end as a text
        file's do. The same text sent again is the file already there; another text under the same name is numbered."""
        folder = os.path.join(workspace, ATTACHED)
        os.makedirs(folder, exist_ok=True)
        data = self.text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        stem, suffix = os.path.splitext(self.file_name())
        n = 1
        while True:
            path = os.path.join(folder, "%s%s%s" % (stem, "" if n == 1 else "-%d" % n, suffix))
            if not os.path.exists(path):
                with open(path, "wb") as f:
                    f.write(data)
                return path
            if os.path.isfile(path) and os.path.getsize(path) == len(data):
                with open(path, "rb") as f:
                    if f.read() == data:
                        return path
            n += 1

    def named(self, path, grep=True):
        """The document as a message carries it by name: where its text is kept, how long it is, how it begins, and how
        the tool looks into it. `grep`: the conversation has the command that finds a file's lines."""
        at = path.replace("\\", "/")
        text = self.text.replace("\r\n", "\n").replace("\r", "\n")
        lines = text.count("\n") + (not text.endswith("\n"))
        head = text.strip()[:HEAD]
        if len(text.strip()) > HEAD and "\n" in head[HEAD // 2:]:
            head = head.rsplit("\n", 1)[0]                     # whole lines, where the last one would be cut short
        look = ('(grep "words" "%s") answers the lines that hold them, each with its number; ' % at if grep else "")
        return ('<file name="%s" kept="%s" lines="%d" characters="%d">\n'
                "This file's text is not in the message: it is kept at the path above, for your tool. %s"
                '(read "%s" 120 80) answers 80 lines from line 120. Look for what is asked, and read all of it only where '
                "the user asks for all of it. It begins:\n%s\n</file>\n"
                % (os.path.basename(self.name.replace("\\", "/")).replace('"', "'") or "file", at, lines, len(text), look, at,
                   head))


def is_document(data, name):
    """Whether `name` and the first bytes say this is a file `read_document` takes rather than a picture or a video."""
    suffix = os.path.splitext(name or "")[1].lower()
    if suffix in (".docx", ".pdf") or suffix in TEXT_SUFFIXES:
        return True
    return data[:5] == b"%PDF-" or (data[:2] == b"PK" and b"word/document.xml" in data[:65536])


def read_document(data, name):
    """`data`, a file's bytes, and its `name` -> a `Document`, or Refused saying why it is not read."""
    suffix = os.path.splitext(name or "")[1].lower()
    if suffix == ".docx" or (data[:2] == b"PK" and suffix not in TEXT_SUFFIXES):
        return Document(name, _docx(data), "docx")
    if suffix == ".pdf" or data[:5] == b"%PDF-":
        return Document(name, _pdf(data), "pdf")
    if suffix == ".doc":
        raise Refused("an old Word document (.doc) is not read here: save it as .docx")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        if b"\x00" in data[:8192]:
            raise Refused("%s is not a document read here (.docx, .pdf, or text)" % name)
        text = data.decode("latin-1")
    return Document(name, text, "text")


def _docx(data):
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
        root = ElementTree.fromstring(z.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError):
        raise Refused("not a Word document (.docx): no word/document.xml in it")
    body = root.find(W + "body")
    lines = []

    def paragraph(p):
        out = []
        for node in p.iter():
            if node.tag == W + "t":
                out.append(node.text or "")
            elif node.tag == W + "tab":
                out.append("\t")
            elif node.tag in (W + "br", W + "cr"):
                out.append("\n")
        return "".join(out)
    for block in (body if body is not None else []):
        if block.tag == W + "p":
            lines.append(paragraph(block))
        elif block.tag == W + "tbl":
            for row in block.iter(W + "tr"):
                lines.append("\t".join(" ".join(paragraph(p) for p in cell.iter(W + "p")) for cell in row.iter(W + "tc")))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _pdf(data):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise Refused("a PDF is read through pypdf (`pip install pypdf`), which is not installed")
    try:
        pages = PdfReader(io.BytesIO(data)).pages
        return "\n\n".join((p.extract_text() or "").strip() for p in pages).strip()
    except Exception as ex:
        raise Refused("the PDF was not read: %s" % ex)
