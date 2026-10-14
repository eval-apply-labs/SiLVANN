"""A FILE AS ANY MODEL READS IT — a document's text, put in the message that names it.

    doc = read_document(data, "report.docx")    # .name, .text, .kind
    text = doc.framed() + user_text             # <file name="report.docx"> … </file> before the message

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


class Document:
    def __init__(self, name, text, kind):
        self.name, self.text, self.kind = name, text, kind

    def framed(self, text=None):
        """The document's text (or `text`, a part of it) as the message carries it."""
        return '<file name="%s">\n%s\n</file>\n' % (self.name.replace('"', "'"), (self.text if text is None else text).strip())


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
