"""A FILE AS ANY MODEL READS IT — a document's text, put in the message that names it, or kept as a file the message names.

    doc = read_document(data, "report.docx")    # .name, .text, .kind
    text = doc.framed() + user_text             # <file name="report.docx"> … </file> before the message
    text = doc.named(path) + user_text          # where its text is kept and how it is looked into — and not its text

⭐ A DOCUMENT IS NOT READ WHOLE UNLESS IT IS SMALL OR THE USER ASKS (⚖ 2026-11-20: *"maybe i want to send him a whole
encyclopedia to say look at the definition of barricade, it would be good if uploaded documents would not trigger a full
read unless explicitly asked to do so"*). `carried` says which way a message takes one: its text where it is at most
`INLINE` tokens and fits, or where the user asked for the whole, or where the conversation has no tool to read a file
with; else named, for `(grep …)` and `(read …)` to look into where the server keeps it — in the conversation's own
attachments, as every file sent to a conversation is (NN-200, ▶ space.py, the server's `upload_image`).

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


class Attached:
    """A file sent with a message that is not text — a spreadsheet, an archive, a picture for a model that reads none —
    kept as it was sent and named to the model: `named` is its line in the message."""

    def __init__(self, name, path, size):
        self.name, self.kept, self.size = name, path, size

    def named(self):
        at = self.kept.replace("\\", "/")
        return ('<file name="%s" kept="%s" bytes="%d">\n'
                "This file was sent with the message and is kept at the path above. It is not text: (read …) does not "
                "show it. Where you have a command that opens its kind, use it; else tell the user what you would need."
                "\n</file>\n" % (os.path.basename(self.name.replace("\\", "/")).replace('"', "'") or "file", at, self.size))


class Document:
    def __init__(self, name, text, kind, sent=None):
        self.name, self.text, self.kind = name, text, kind
        self.sent = sent              # its bytes as they were sent, where the reader kept them
        self.kept = None              # where its text is kept for the tool, once the server has kept it

    def framed(self, text=None):
        """The document's text (or `text`, a part of it) as the message carries it — with where the file is kept, once
        the server has kept it, so a model that reaches for the file itself finds it."""
        kept = ' kept="%s"' % self.kept.replace("\\", "/") if self.kept else ""
        return '<file name="%s"%s>\n%s\n</file>\n' % (self.name.replace('"', "'"), kept,
                                                      (self.text if text is None else text).strip())

    def text_name(self, kept_as):
        """The name its text is kept under beside `kept_as`, the file as it was sent — None where the file is its text
        (a text file in UTF-8): `<name>.txt` for a Word document or a PDF, and for a text in another encoding."""
        same = self.kind == "text" and self.as_text() == self.sent
        return None if same else os.path.basename(kept_as) + ".txt"

    def as_text(self):
        """Its text as a file holds it: UTF-8, its lines ending as a text file's."""
        return self.text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")

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


MEDIA_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".mp4", ".mov", ".m4v", ".webm", ".mkv",
                  ".avi", ".mpg", ".mpeg")


def is_media(data, name):
    """Whether `name` or the first bytes say a picture or a video — what a model that reads them is given to read. A
    file that is neither this nor a document is kept and named (▶ `Attached`)."""
    suffix = os.path.splitext(name or "")[1].lower()
    head = bytes(data[:16])
    return (suffix in MEDIA_SUFFIXES or head.startswith((b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"BM"))
            or (head[:4] == b"RIFF" and head[8:12] in (b"WEBP", b"AVI ")) or head[4:8] == b"ftyp"
            or head[:4] == b"\x1a\x45\xdf\xa3")


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
        return Document(name, _docx(data), "docx", data)
    if suffix == ".pdf" or data[:5] == b"%PDF-":
        return Document(name, _pdf(data), "pdf", data)
    if suffix == ".doc":
        raise Refused("an old Word document (.doc) is not read here: save it as .docx")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        if b"\x00" in data[:8192]:
            raise Refused("%s is not a document read here (.docx, .pdf, or text)" % name)
        text = data.decode("latin-1")
    return Document(name, text, "text", data)


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
