"""What a browser keeps between pages, and the answers this process is streaming."""
import os
from typing import Optional


# ---------- Per-browser session state ----------

def _session_default():
    return {
        "endpoint": os.environ.get("SILVANN_DEFAULT_ENDPOINT", "http://localhost:8765"),
        "active_id": None,
        "active_name": None,
        "active_model": None,      # the model the open conversation is with; another serving empties the chat
        # Per-cart passwords this browser session has unlocked. Kept in app.storage.user so they
        # survive page reloads for the same user (one browser identity). They are NOT shared with
        # other users' sessions.
        "passwords": {},
        "protected_active": False,
        "archived_active": False,                 # the conversation shown is archived: read, and not written to
        # Whether the active cart is currently bound to VRAM. False after a lazy /activate
        # (metadata only) — flips True after the user clicks the Activate banner.
        "active_bound": False,
        "history": [],     # [{"role": "user"|"assistant", "content": str}]
        "streaming": False,
        # Which conversation's inline-actions drawer is currently open (one at a time).
        "expanded_cart_id": None,
        # Display toggle: render <think>...</think> blocks in assistant messages or strip them.
        "show_thinking": False,
        # The page light or dark — this browser's choice.
        "dark": False,
        # Engine toggle for the active conversation: whether the model is asked to think at all.
        # Synced from the server on activate (the canonical value lives in the cartridge sidecar).
        "active_thinking_mode": True,
        # Nav state: which main view is selected and whether the chat sub-tree is expanded.
        # The drawer itself defaults to mini-with-hover-expand; that's a fixed UX, not stored.
        "view": "chat",            # chat | options | help
        "convs_expanded": True,
        # Options-view prefs (was the Server tab)
        "server_mode": "local",    # local (this machine's server) | remote (another's, at `endpoint`)
        "selected_model": "",      # a folder name under main/models/ (drives the config selector)
        "selected_config": "",
        "server_port": 8765,
        # Index of the last log line shown in the subprocess log pane
        "log_cursor": 0,
        # Per-cart pending image attachments awaiting the next /messages send.
        # Shape: {cart_id: [{"image_id": str, "filename": str, "n_pads": int,
        # "merged_thw": [T,H,W]}]}. Cleared after each successful turn (server
        # also pops the cache server-side — one-shot semantics).
        "pending_images": {},
    }


# ---------- The answers being streamed, and the pages open ----------

# ⭐ THE ANSWERS THIS UI IS STREAMING, A BROWSER'S AT A TIME: browser id -> {"task": the stream's, "cart": its
#   conversation, "before": the conversation before the turn, "turn": the turn's messages — the answer last —, "ask":
#   the model's ask it ended on, "seen": bumped as it grows}. The answer is written there, in this process's memory, and
#   never into a page's saved state: an answer goes on when its page goes, and NiceGUI then closes the browser's storage
#   and reads it again from its file for the next page — an answer written into the closed copy is lost to that page,
#   and writes the file over behind it (⚖ 2026-11-13: "if the user leaves the page ... the resume gets messy"). Each page
#   of the browser copies the answer into its own state as it grows, and a page opened while it streams — the user back,
#   on the same tab or a new one — follows it. The last answer's record stays until the next, for a page that followed it.
LIVE: dict = {}
# the pages open for each web user (a browser tab's id): one gone tells the server only when it was the last — a reload
#   opens the new page before NiceGUI deletes the old one
PAGES: dict = {}


def _streaming(browser_id: str) -> Optional[dict]:
    """The answer this browser's pages are streaming, while it is; else None."""
    live = LIVE.get(browser_id)
    return live if live is not None and not live["task"].done() else None


def _with_turn(live: dict) -> list:
    """The conversation as the streaming answer's record has it: before the turn, then the turn so far — copies."""
    return [dict(m) for m in live["before"]] + [dict(m) for m in live["turn"]]
