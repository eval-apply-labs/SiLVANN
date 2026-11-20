"""The Help view: the release's documents, their pictures and links, and the view that shows them."""
import re
from pathlib import Path
from typing import Optional

from nicegui import app, ui

from .local import CONFIG_GUIDE_DOC, DOCS_DIR, ENGINE_SRC, LICENCE_DIR, RELEASE_DIR
from .part import Part


# The Help view's documents, a group each: (group, label, path) — every document the release ships, the UI's guide first
HELP_DOCS = [
    ("Using SiLVANN", "The UI", DOCS_DIR / "ui.md"),
    ("Using SiLVANN", "Overview", RELEASE_DIR / "README.md"),
    ("Using SiLVANN", "Running SiLVANN", DOCS_DIR / "users.md"),
    ("Using SiLVANN", "A model's config", DOCS_DIR / "config.md"),
    ("Using SiLVANN", "The model's tool and your own commands", DOCS_DIR / "tools.md"),
    ("Using SiLVANN", "Skills", RELEASE_DIR / "skills" / "README.md"),
    ("Using SiLVANN", "SiLVANN in a container", RELEASE_DIR / "docker" / "README.md"),
    ("Using SiLVANN", "More container setups", RELEASE_DIR / "docker" / "advanced.md"),
    ("Using SiLVANN", "Every release's notes", DOCS_DIR / "release_logs" / "v1.md"),
    ("Using SiLVANN", "The 0.x releases' notes", DOCS_DIR / "release_logs" / "v0.md"),
    ("Using SiLVANN", "Why AI on your own machine", DOCS_DIR / "local_ai.md"),
    ("The models", "The models", DOCS_DIR / "models.md"),
    ("The models", "Their results", DOCS_DIR / "results.md"),
    ("The models", "A model's folder", RELEASE_DIR / "models" / "README.md"),
    ("The models", "How the weights are packed", DOCS_DIR / "compression.md"),
    ("Developing", "Working on SiLVANN", DOCS_DIR / "developers.md"),
    ("Developing", "The concepts it is built on", DOCS_DIR / "concepts.md"),
    ("Developing", "What inspired it", DOCS_DIR / "inspiration.md"),
    ("Developing", "Where it could go", DOCS_DIR / "directions.md"),
    ("Developing", "The engine's source", ENGINE_SRC / "README.md"),
    ("Developing", "Package sys", ENGINE_SRC / "packages" / "sys" / "README.md"),
    ("Developing", "Package nn", ENGINE_SRC / "packages" / "nn" / "README.md"),
    ("Developing", "Package ai_qwen_3", ENGINE_SRC / "packages" / "ai_qwen_3" / "README.md"),
    ("Developing", "Package ai_z__glm_5_3", ENGINE_SRC / "packages" / "ai_z__glm_5_3" / "README.md"),
    ("Developing", "Package ai_qwen_image", ENGINE_SRC / "packages" / "ai_qwen_image" / "README.md"),
    ("Developing", "Package ai_deepseek_4", ENGINE_SRC / "packages" / "ai_deepseek_4" / "README.md"),
    ("Developing", "Adding a silicon family", ENGINE_SRC / "silicon_families" / "add_a_family.md"),
    ("Developing", "OpenCL 1.2: old NVIDIA cards and integrated GPUs", ENGINE_SRC / "silicon_families" / "khronos_opencl1_base" / "README.md"),
    ("Developing", "Contributing", LICENCE_DIR / "CONTRIBUTING.md"),
    ("Licences", "The licence", LICENCE_DIR / "LICENSE"),
    ("Licences", "Attribution", LICENCE_DIR / "ATTRIBUTION.md"),
    ("Licences", "The contributor licence agreement", LICENCE_DIR / "CLA.md"),
]
HELP_FIRST = "Using SiLVANN · The UI"


# ⭐ THE DOCUMENTS' PICTURES: docs/images/, served to the Help view under a path of its own — the release's public files,
#   that folder alone. A document embeds a picture as GitHub shows it, a `<picture>` of a light and a dark file chosen by
#   the system's theme; the Help view shows the one the UI's own theme asks for, at the address it serves it from.
HELP_IMAGES_DIR = DOCS_DIR / "images"
HELP_IMAGES_URL = "/silvann-help-images"
HELP_PICTURE = re.compile(r'<picture>\s*<source[^>]*\bsrcset="([^"]+)"[^>]*>\s*<img\s+src="([^"]+)"([^>]*)>\s*</picture>')
if HELP_IMAGES_DIR.is_dir():
    app.add_static_files(HELP_IMAGES_URL, str(HELP_IMAGES_DIR))


def help_image_url(current: Path, src: str) -> Optional[str]:
    """The UI's address for a picture `current` names by a relative path — None where it is not a file of docs/images."""
    if src.startswith("/") or re.match(r"[a-z][a-z0-9+.-]*:", src, re.I):
        return None
    path = (current.parent / src).resolve()
    try:
        rel = path.relative_to(HELP_IMAGES_DIR.resolve())
    except ValueError:
        return None
    return HELP_IMAGES_URL + "/" + rel.as_posix() if path.is_file() else None


def help_pictures(current: Path, text: str, dark: bool) -> str:
    """`text` with each `<picture>` the Help view can serve made one image: the dark file or the light, its alt kept."""
    def one(m):
        url = help_image_url(current, m.group(1) if dark else m.group(2))
        if url is None:
            return m.group(0)
        alt = re.search(r'\balt="([^"]*)"', m.group(3))
        alt_attr = ' alt="%s"' % " ".join(alt.group(1).split()) if alt else ""
        return '<img src="%s"%s style="max-width: 100%%; height: auto;">' % (url, alt_attr)
    return HELP_PICTURE.sub(one, text)


def help_doc(path: Path, dark: bool = False) -> str:
    """A document as the Help view shows it: Markdown with its pictures served (▶ `help_pictures`), any other text in a
    block of its own."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return f"_`{path.name}` could not be read: {e}_"
    return help_pictures(path, text, dark) if path.suffix == ".md" else "```text\n" + text + "\n```"


HELP_HREF = re.compile(r"\]\(<?([^)\s>]+)")             # a Markdown link's target


def help_target(current: Path, href: str) -> Optional[Path]:
    """The file a relative link in `current` names, without its anchor — None where it names no file."""
    target = href.split("#", 1)[0]
    if not target or re.match(r"[a-z][a-z0-9+.-]*:", target, re.I):
        return None
    path = (current.parent / target).resolve()
    return path if path.is_file() else None


def help_shipped() -> set:
    """The files the Help view shows: the picker's documents, and the files their links name, document to document —
    the release's own text. The browser says which link was clicked, and it can say any path: the Help view shows no
    file outside this set, or anyone who reaches the UI could read any file it can (a token, a home folder's keys)."""
    seen, todo = set(), [path.resolve() for _, _, path in HELP_DOCS if path.is_file()]
    while todo:
        doc = todo.pop()
        if doc in seen:
            continue
        seen.add(doc)
        if doc.suffix == ".md":
            try:
                text = doc.read_text(encoding="utf-8")
            except OSError:
                continue
            todo += [p for p in (help_target(doc, href) for href in HELP_HREF.findall(text)) if p is not None]
    return seen


HELP_SHIPPED = help_shipped()


def help_link(current: Path, href: str) -> Optional[Path]:
    """The document a link in `current` names, without its anchor — None where it names none the release ships."""
    path = help_target(current, href)
    return path if path in HELP_SHIPPED else None


class HelpView(Part):
    """The Help view: a document picker over the release's documents."""

    def build(self):
        """The Help view's widgets, the first document shown."""
        # --- Help view: a document picker over the release's documents, the UI's guide first ---
        self.help_view = ui.column().classes("silvann-main w-full p-6 overflow-auto")
        with self.help_view:
            with ui.column().classes("max-w-4xl mx-auto gap-3 w-full"):
                self.help_paths = {"%s · %s" % (g, label): path for g, label, path in HELP_DOCS}
                self.help_shown = {"path": self.help_paths[HELP_FIRST]}
                self.help_doc_select = ui.select(list(self.help_paths), value=HELP_FIRST, label="Document").classes("w-96")
                self.help_doc_md = ui.markdown(help_doc(self.help_paths[HELP_FIRST], bool(self.state.get("dark", False))))

                self.help_doc_select.on_value_change(lambda e: self.show_help(self.help_paths[e.value]))

                # ⛳ a relative link would take the browser to a page the UI does not serve: it is followed here instead, and
                #   a web page opens in a tab of its own
                self.help_doc_md.on("click", self.on_help_link, js_handler="""(e) => {
                const a = e.target.closest('a');
                if (!a) return;
                const href = a.getAttribute('href') || '';
                if (href.startsWith('#')) return;
                e.preventDefault();
                if (/^[a-z][a-z0-9+.-]*:/i.test(href)) { window.open(href, '_blank', 'noopener'); return; }
                emit(href);
            }""")

    def show_help(self, path: Path):
        self.help_shown["path"] = path
        self.help_doc_md.set_content(help_doc(path, bool(self.state.get("dark", False))))

    def open_config_guide(self):
        """Help, at the guide to a config's keys and every model's options."""
        self.help_doc_select.set_value(CONFIG_GUIDE_DOC)
        self.show_help(self.help_paths[CONFIG_GUIDE_DOC])
        self.state["view"] = "help"
        self.shell.update_view()

    def on_help_link(self, e):
        """A link between the documents: the one it names, in the picker where it is listed."""
        path = help_link(self.help_shown["path"], str(e.args))
        if path is None:
            self.safe_notify("%s is not in the release" % e.args, type="warning")
            return
        listed = next((k for k, v in self.help_paths.items() if v.resolve() == path), None)
        if listed is not None:
            self.help_doc_select.set_value(listed)
        else:
            self.show_help(path)
