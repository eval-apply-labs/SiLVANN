"""SiLVANN chat UI — a NiceGUI client of a start_silvann_server.py HTTP endpoint.

Layout: a left drawer holds the navigation (Chat / Options / Help), the conversations' sub-tree and the endpoint's
settings; the main area shows the view the menu selects. The drawer is collapsible.

Run:
    python3 silvann_ui.py                            # serves the UI on http://localhost:8080
    python3 silvann_ui.py --port 8090 --no-browser   # another port, no browser opened

This file is the entry point; the UI is the package beside it, `silvann_ui_parts/`, one concern a module:
    run.py              the process: where it listens, the model it starts again, `main`
    page.py             a page: what it is made of and the order it is built in (`Page`, and `Part`, in part.py, under
                        every part of one)
    shell.py            the drawer, the view shown, the theme
    chat.py  turn.py  conversations.py  connection.py
                        the Chat view and a conversation drawn in it; a message sent and its answer streamed; the
                        conversations' list; the server asked
    options.py  models_panel.py  users_view.py  help.py  remote_sign_in.py
                        the other views, and another machine's sign-in
    client.py  answer.py  session.py  sign_in.py  local.py  server_process.py  downloads.py  head.py
                        the server's HTTP API, an answer's parts, a browser's state, who a page is for, this machine's
                        folders, the server and the downloads it starts, the page's styles and scripts
The local panel (Options) works where start_silvann_server.py is beside this file.
"""
import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:                                  # loaded by its path, not run: its parts are beside it
    sys.path.insert(0, _HERE)

# ⛳ THE NAMES THIS FILE HAD WHEN IT WAS THE WHOLE UI ARE STILL READ HERE — `silvann_ui.LOCAL_USERS`, `.help_link`,
#   `.DOCS_DIR`: the checks in test/ load this file and read them.
from silvann_ui_parts import page as _page                 # noqa: E402,F401 — the pages, made as it is imported
from silvann_ui_parts.answer import *                      # noqa: E402,F401,F403
from silvann_ui_parts.client import *                      # noqa: E402,F401,F403
from silvann_ui_parts.downloads import *                   # noqa: E402,F401,F403
from silvann_ui_parts.head import *                        # noqa: E402,F401,F403
from silvann_ui_parts.help import *                        # noqa: E402,F401,F403
from silvann_ui_parts.local import *                       # noqa: E402,F401,F403
from silvann_ui_parts.run import *                         # noqa: E402,F401,F403
from silvann_ui_parts.server_process import *              # noqa: E402,F401,F403
from silvann_ui_parts.session import *                     # noqa: E402,F401,F403
from silvann_ui_parts.sign_in import *                     # noqa: E402,F401,F403
from silvann_ui_parts.run import main                      # noqa: E402

if __name__ in {"__main__", "__mp_main__"}:
    main()
