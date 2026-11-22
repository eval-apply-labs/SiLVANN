"""The Chat view: a conversation's messages as they are drawn, the answer growing, the attachments."""
import asyncio
import base64

import httpx
from nicegui import ui

from .answer import ASK_LABELS, PICTURE_RE, answer_parts, tools_block
from .part import Part
from .session import LIVE, _streaming


class ChatView(Part):
    """The Chat view's widgets, and a conversation drawn into them."""

    def build(self):
        """The Chat view's widgets: its header and switches, the messages' column, the attachments, the input."""
        # ============================================================
        # MAIN AREA — three views (only one visible at a time)
        # ============================================================

        # --- Chat view ---
        self.chat_view = ui.column().classes("silvann-main w-full p-4 gap-2 overflow-hidden")
        with self.chat_view:
            # Header row: conversation name on the left, speed counter + two toggles + save button.
            with ui.row().classes("w-full items-center gap-3"):
                self.chat_header = ui.label("(no conversation active)").classes("text-sm opacity-60 flex-grow")
                # Live tok/s during streaming; final "<n> tok in <s>s · <r> tok/s" after each turn.
                self.speed_label = ui.label("—").classes("text-xs font-mono opacity-70")
                # the conversation's tokens of the most the model takes: how full it is
                with ui.row().classes("items-center gap-1 no-wrap") as self.context_box:
                    self.context_bar = ui.linear_progress(value=0, show_value=False, size="6px").props(
                        "rounded color=primary").classes("w-16")
                    self.context_label = ui.label("").classes("text-xs font-mono opacity-70")
                self.context_box.tooltip("The conversation's tokens, of the most the model takes (its max_context)")
                self.context_box.set_visibility(False)
                self.thinking_mode_toggle = ui.switch("thinking mode",
                                                  value=self.state.get("active_thinking_mode", True)
                                                  ).props("dense").classes("text-xs")
                self.show_think_toggle = ui.switch("show thinking",
                                               value=self.state.get("show_thinking", False)
                                               ).props("dense").classes("text-xs")
                # the model's tool (NN-88): a Lisp REPL on the server's machine, switched as thinking is — declared in a
                # conversation's first turn, so a conversation begun without it cannot turn it on — and what its commands
                # may do unasked, the conversation's (a new one's the server config's tools, whose "no" turns it off)
                self.tools_toggle = ui.switch("tools", value=False).props("dense").classes("text-xs").tooltip(
                    "The model may use a Lisp REPL on the server's machine — list folders, read and write files, run "
                    "commands, read web pages. Off: its calls are refused.")
                self.tools_mode = ui.select({"ask": "ask first", "chill": "reads unasked", "full-auto": "full auto"},
                                            value=None).props("dense options-dense borderless").classes("text-xs").tooltip(
                    "What this conversation's tool may do without asking you: 'ask first' reads in its workspace and "
                    "asks for anything else; 'reads unasked' reads anywhere and asks to write, run or fetch; 'full auto' "
                    "asks nothing.")
                # Lock icon is the password control: click to set / change / remove the password for
                # the active conversation. lock_open = unprotected, lock = protected. Since saves
                # happen automatically after each turn, there's no separate "save" button.
                self.protected_icon = ui.button(icon="lock_open"
                                                ).props("flat dense round size=sm"
                                                        ).classes("opacity-70"
                                                                  ).tooltip("Set / change / remove password")

                self.thinking_mode_toggle.on_value_change(self.on_thinking_mode)

                self.show_think_toggle.on_value_change(self.on_show_thinking)

                self.tools_toggle.on_value_change(self.on_tools)

                self.tools_mode.on_value_change(self.on_tools_mode)
            # ⭐ AN ARCHIVED CONVERSATION IS SHOWN, AND NOT WRITTEN TO (NN-200, ⚖ 2026-11-20: *"the ui can say "this
            #   conversation is archived" and show it to you anyway but with no send enabled"*): this banner in the
            #   place of the one that activates, the message box and its buttons off (▶ `set_archived`)
            with ui.row().classes("w-full items-center gap-2 p-3 rounded bg-slate-100 border border-slate-300"
                                  ).bind_visibility_from(self.state, "archived_active"):
                ui.icon("inventory_2", size="sm").classes("text-slate-600")
                ui.label("This conversation is archived: its transcript is kept and its state was given up. It is shown "
                         "here, and not written to.").classes("text-sm flex-grow")
            # Banner shown when the active cart is loaded into UI memory but NOT bound to VRAM.
            # User must click Activate to /bind. Hidden once bound; reappears after server eviction.
            with ui.column().classes("w-full gap-0").bind_visibility_from(self.state, "archived_active", lambda a: not a):
                with ui.row().classes("w-full items-center gap-2 p-3 rounded bg-amber-50 border border-amber-300"
                                      ).bind_visibility_from(self.state, "active_bound", lambda b: not b
                                                             ) as bind_banner:
                    ui.icon("info", size="sm").classes("text-amber-700")
                    self.bind_banner_label = ui.label("Conversation loaded. Type a message, or click Activate, to bring the session online."
                                                  ).classes("text-sm flex-grow")
                    self.bind_btn = ui.button("Activate", icon="play_arrow"
                                              ).props("color=primary unelevated")
            self.chat_scroll = ui.scroll_area().classes("silvann-chat w-full flex-grow min-h-0 border rounded")
            with self.chat_scroll:
                self.chat_history_col = ui.column().classes("w-full p-4 gap-3")
                self.chat_tail_col = ui.column().classes("w-full px-4 pb-4 gap-3")   # below the messages: "…", the model's ask
            # Attached-image chip row. Rendered into by render_pending_images() whenever
            # an image is uploaded; empty (and visually absent) otherwise.
            self.pending_images_row = ui.row().classes("w-full items-center gap-2 flex-wrap")
            # Shown while an attached image is being uploaded + run through the vision encoder
            # (server-side ViT can take many seconds with no other feedback). Hidden otherwise.
            self.image_busy_row = ui.row().classes("w-full items-center gap-2")
            with self.image_busy_row:
                ui.label("Encoding image…").classes("text-sm opacity-80 whitespace-nowrap")
                ui.linear_progress(show_value=False).props(
                    "indeterminate rounded color=primary").classes("flex-grow")
            self.image_busy_row.set_visibility(False)
            with ui.row().classes("w-full items-end gap-2"):
                # The paperclip uses the same `ui.upload` widget the cartridge import uses, but
                # accepting pictures, videos and documents (each read by the server at the same endpoint)
                # and bound to the active cart's /images endpoint.
                # auto_upload fires on file selection so the user doesn't have to click upload twice.
                self.attach_upload = ui.upload(auto_upload=True, max_files=1,
                                           max_file_size=100 * 1024 * 1024
                                           ).props('label="" flat hide-upload-btn color=primary')
                self.attach_upload.classes("hidden")
                self.attach_btn = ui.button(icon="attach_file"
                                            ).props("color=primary"
                                                    ).tooltip("Send a file to this conversation, for the next message: a document, a "
                                                              "picture or a video for a model that reads them, or any other "
                                                              "file. It is kept with the conversation and deleted with it.")
                self.attach_btn.set_visibility(False)       # shown when the server's model reads pictures or files (its /health)
                self.input_box = ui.textarea(placeholder="Type a message — Enter to send, Shift+Enter for newline"
                                             ).classes("flex-grow").props("autogrow outlined")
                self.send_btn = ui.button(icon="send")

            self.attach_upload.on_upload(self.on_attach_upload)
            self.attach_upload.on_rejected(self.on_attach_rejected)
            # ⭐ THE 📎 OPENS THE FILE CHOOSER IN THE CLICK ITSELF, EVERY TIME (⚖ 2026-11-20: *"i dont see a popup allowing
            #   me to upload my file"*): in the page, with no round trip to this process — a browser opens a file chooser
            #   only for the person's own click — and with the uploader emptied first: it takes one file at a time, and
            #   one left in it by an upload that did not go through would have it open no chooser again.
            self.attach_btn.on("click", js_handler="() => { runMethod(%d, 'reset', []); runMethod(%d, 'pickFiles', []); }"
                                                   % (self.attach_upload.id, self.attach_upload.id))
            # Hide chips initially (no attachments).
            self.pending_images_row.set_visibility(False)

    def set_archived(self, archived: bool):
        """The conversation shown is archived, or is not: the banner, and the message box and its buttons off or on."""
        self.state["archived_active"] = bool(archived)
        for w in (self.input_box, self.send_btn, self.attach_btn):
            w.set_enabled(not archived)
        self.input_box.props('placeholder="%s"' % ("This conversation is archived" if archived else
                                                    "Type a message — Enter to send, Shift+Enter for newline"))

    def wire(self):
        """What a drawn conversation is kept in."""
        self.pictures: dict = {}                                # (conversation, picture) -> its data URI, or None while it loads

        # ⭐ AN ANSWER IS DRAWN AS ITS PARTS (`answer_parts`): its text; each run of tool calls a section that opens and closes,
        #   closed at first; its thinking — with "show thinking" on a section of its own, open at first, and off nothing once
        #   the block is closed and "thinking…" while the model is still in it. While an answer streams only its own message
        #   is brought up to date, a part at a time and at most ten times a second: drawing the whole conversation again for
        #   every token is what made the chat flicker.
        self.opened: dict = {}                                  # (message, part) -> whether its section is open
        self.answers: dict = {}                                 # an answer's index -> its message and its drawn parts
        self.live_answer: dict = {}                             # the streaming answer's index, and whether its text has grown

        self.empty_note: dict = {}                              # the "(empty — say hi)" line while the conversation has none

    async def on_thinking_mode(self, e):
        self.state["active_thinking_mode"] = bool(e.value)
        if self.state["active_id"]:
            try:
                await self.client.set_thinking(self.state["active_id"], bool(e.value))
            except Exception as ex:
                self.safe_notify(f"couldn't update thinking mode: {ex}", type="negative")

    def on_show_thinking(self, e):
        self.state["show_thinking"] = bool(e.value)
        self.render_chat()

    async def on_tools(self, e):
        if not self.state["active_id"] or bool(e.value) == self.state.get("active_tools", False):
            return
        try:
            await self.client.set_tools(self.state["active_id"], bool(e.value))
            self.state["active_tools"] = bool(e.value)
        except httpx.HTTPStatusError as ex:
            self.tools_toggle.set_value(self.state.get("active_tools", False))
            self.safe_notify(ex.response.json().get("detail", str(ex)) if ex.response.status_code == 409
                             else f"couldn't set the tool: {ex}", type="warning")
        except Exception as ex:
            self.safe_notify(f"couldn't set the tool: {ex}", type="negative")

    async def on_tools_mode(self, e):
        if not e.value or not self.state["active_id"] or e.value == self.state.get("tools_mode"):
            return
        try:
            await self.client.set_tools_mode(self.state["active_id"], e.value)
            self.state["tools_mode"] = e.value
        except Exception as ex:
            self.tools_mode.set_value(self.state.get("tools_mode"))
            self.safe_notify(f"couldn't set what the tool may do: {ex}", type="negative")

    def _cart_pending(self) -> list:
        """Mutable list of pending image attachments for the active cart."""
        cart_id = self.state.get("active_id")
        if not cart_id:
            return []
        return self.state["pending_images"].setdefault(cart_id, [])

    def render_pending_images(self):
        self.pending_images_row.clear()
        entries = self._cart_pending()
        if not entries:
            self.pending_images_row.set_visibility(False)
            return
        self.pending_images_row.set_visibility(True)
        with self.pending_images_row:
            ui.icon("attach_file", size="sm").classes("opacity-70")
            for idx, ent in enumerate(entries):
                if ent.get("kind") in ("document", "file"):
                    # ⭐ A DOCUMENT IS NOT READ WHOLE UNLESS IT IS SMALL OR ASKED FOR (⚖ 2026-11-20): the chip says how the
                    #   next message carries it — the server's own word, `carried` — and one kept as a file is put in
                    #   the message whole by a click, and back by another
                    how = "whole in the message" if ent.get("whole") else {
                        "text": "in the message", "cut": "what fits of it",
                        "named": "kept as a file for the model to look into"}.get(ent.get("carried"), "in the message")
                    if ent.get("kind") == "file":             # not text: kept with the conversation, named to the model
                        how = "kept with the conversation, named to the model"
                    chip = ui.chip(f"{ent['filename']} · {how}", icon="description", removable=True
                                   ).props("dense color=blue-2 text-color=blue-9")
                    if ent.get("carried") == "named" and ent.get("kind") == "document":
                        chip.props("clickable")
                        chip.tooltip("Not read whole: the model searches it and reads the lines it needs. Click to put all of "
                                     "it in the message instead — as much as the conversation has room for."
                                     if not ent.get("whole") else
                                     "All of it goes in the message, as much as the conversation has room for. Click to keep "
                                     "it as a file the model looks into.")
                        def _on_whole(idx=idx):
                            cur = self._cart_pending()
                            if 0 <= idx < len(cur):
                                cur[idx]["whole"] = not cur[idx].get("whole")
                                self.render_pending_images()
                        chip.on("click", _on_whole)
                else:
                    chip = ui.chip(
                        f"{ent['filename']} · {ent['n_pads']} pads",
                        icon="image", removable=True,
                    ).props("dense color=blue-2 text-color=blue-9")
                def _on_remove(idx=idx):
                    cur = self._cart_pending()
                    if 0 <= idx < len(cur):
                        cur.pop(idx)
                        self.render_pending_images()
                chip.on("remove", _on_remove)

    def on_attach_rejected(self, *_):
        """A file the page did not take — larger than an attachment may be: said, and the 📎 ready for another."""
        self.safe_notify("That file is larger than an attachment may be: 100 MB at most", type="warning")
        self.attach_upload.reset()

    async def on_attach_upload(self, e):
        # Give the user feedback for the whole round-trip — the upload itself is quick but
        # the server then runs the vision encoder, which has no other progress signal.
        self.attach_btn.disable()
        self.image_busy_row.set_visibility(True)
        try:
            # ⭐ A FILE CHOSEN WITH NO CONVERSATION OPEN GOES TO A NEW ONE, and one not online is brought online: choosing
            #   a file is the user's say-so, as typing a message is (▶ turn.py's `send`). Whatever comes of it, the
            #   uploader is emptied after (`finally`), so the 📎 opens its chooser the next time.
            if self.state["active_id"] is None and not await self.page.conversations.start_conversation():
                return
            if not self.state.get("active_bound"):
                await self.page.conversations.bind_active()
                if not self.state.get("active_bound"):
                    return
            cart_id = self.state["active_id"]
            # NiceGUI 3.x upload event: e.file is a FileUpload with async read() +
            # name + content_type. (Older NiceGUI used e.content/e.name/e.type.)
            blob = await e.file.read()
            filename = e.file.name or "image.png"
            content_type = e.file.content_type or "application/octet-stream"
            resp = await self.client.upload_image(cart_id, blob, filename=filename,
                                              content_type=content_type)
            self._cart_pending().append({
                "image_id": resp["image_id"],
                "filename": filename,
                "n_pads": int(resp.get("n_placeholders", 0)),
                "merged_thw": resp.get("merged_thw", []),
                "kind": resp.get("kind"), "carried": resp.get("carried"), "whole": False,
            })
            self.render_pending_images()
            if resp.get("kind") == "document":
                self.safe_notify(f"attached {filename} ({int(resp.get('chars', 0)):,} characters)" + (
                    " — kept as a file for the model to look into, not read whole; click its chip to put it in the message"
                    if resp.get("carried") == "named" else ""), type="positive")
            elif resp.get("kind") == "file":
                self.safe_notify(f"attached {filename} ({int(resp.get('bytes', 0)):,} bytes) — kept with the conversation "
                                 "and named to the model", type="positive")
            else:
                self.safe_notify(f"attached {filename} ({resp.get('n_placeholders')} placeholders)",
                                 type="positive")
        except httpx.HTTPStatusError as ex:
            self.safe_notify(f"image upload refused: {ex.response.status_code} "
                             f"{ex.response.text}", type="negative")
        except Exception as ex:
            self.safe_notify(f"image upload failed: {ex}", type="negative")
        finally:
            self.image_busy_row.set_visibility(False)
            self.attach_btn.enable()
            self.attach_upload.reset()

    # ----- Chat helpers -----
    def show_context(self, used=None, context=None):
        """The top bar's meter: the open conversation's tokens of the context, from the server's numbers — a
        conversation's when it opens, each token's as the answer streams, the answer's end."""
        if context is not None:
            self.state["context"] = int(context or 0)
        if used is not None:
            self.state["context_used"] = int(used)
        total, used = self.state.get("context") or 0, self.state.get("context_used") or 0
        self.context_box.set_visibility(bool(total and self.state.get("active_id")))
        if not total:
            return
        share = min(1.0, used / total)
        self.context_bar.value = share
        self.context_bar.props("color=" + ("negative" if share >= 0.9 else "warning" if share >= 0.75 else "primary"))
        self.context_label.text = f"{used:,} / {total:,}"

    def picture_markdown(self, alt: str, name: str) -> str:
        """A picture in an answer: drawn once the server's copy is here, fetched the first time it is wanted."""
        key = (self.state["active_id"], name)
        if key not in self.pictures:
            self.pictures[key] = None
            asyncio.create_task(self.load_picture(*key))
        uri = self.pictures[key]
        return "![%s](%s)" % (alt, uri) if uri else "_(the picture is loading…)_"

    async def load_picture(self, cid: str, name: str):
        try:
            data = await self.client.picture(cid, name)
            self.pictures[(cid, name)] = "data:image/jpeg;base64," + base64.b64encode(data).decode()
        except Exception as ex:                         # noqa: BLE001 — said where the picture would be
            self.pictures[(cid, name)] = None
            self.safe_notify(f"the picture {name} could not be read: {ex}", type="warning")
            return
        if self.state["active_id"] != cid:
            return
        last = len(self.state["history"]) - 1
        for index, m in enumerate(self.state["history"]):     # only the answers that name it drawn again
            if index in self.answers and m["role"] == "assistant" and "(picture:%s)" % name in (m["content"] or ""):
                self.refresh_answer(index, bool(self.state["streaming"]) and index == last)

    def shown_parts(self, content: str, live: bool) -> list:
        """What an answer shows, part by part: ("text", markdown), ("tools", title, markdown), ("thought", title,
        markdown), ("thinking", label)."""
        show_think = bool(self.state.get("show_thinking", False))
        out = []
        for part in answer_parts(content):
            if part[0] == "text":
                out.append(("text", PICTURE_RE.sub(lambda p: self.picture_markdown(p.group(1), p.group(2)), part[1])))
            elif part[0] == "tools":
                if out and out[-1][0] == "tools":       # runs apart only by thinking not shown: one section
                    out[-1] = ("tools", out[-1][1] + part[1])
                else:
                    out.append(("tools", list(part[1])))
            elif show_think:
                out.append(("thought", "thinking" if part[2] else "thinking…", part[1].strip() or "…"))
            elif not part[2]:
                out.append(("thinking", "thinking…" if live else "(stopped while thinking)"))
        if live and not out:
            out.append(("thinking", "…"))
        for i, part in enumerate(out):
            if part[0] == "tools":
                n = sum(1 for line in part[1] if line.startswith("> "))
                out[i] = ("tools", "tool · %d call%s" % (n, "" if n == 1 else "s"), tools_block(part[1]))
        return out

    def draw_part(self, part, key) -> dict:
        """One part's element in the current slot, and what a later update of it changes."""
        kind = part[0]
        if kind == "text":
            md = ui.markdown(part[1])
            return {"kind": kind, "el": md, "md": md}
        if kind == "thinking":
            label = ui.label(part[1]).classes("text-sm italic opacity-60 animate-pulse")
            return {"kind": kind, "el": label, "md": label}
        section = ui.expansion(part[1], icon="build" if kind == "tools" else "psychology",
                               value=self.opened.get(key, kind == "thought"),
                               on_value_change=lambda e, k=key: self.opened.__setitem__(k, bool(e.value))
                               ).props("dense").classes("w-full silvann-part")
        with section:
            md = ui.markdown(part[2]).classes("text-sm" + (" opacity-70" if kind == "thought" else ""))
        return {"kind": kind, "el": section, "md": md}

    def update_part(self, drawn: dict, part):
        if drawn["kind"] == "thinking":
            drawn["md"].text = part[1]
            return
        if drawn["kind"] != "text":
            drawn["el"].text = part[1]
        content = part[1] if drawn["kind"] == "text" else part[2]
        if drawn["md"].content != content:
            drawn["md"].set_content(content)

    def draw_answer(self, content: str, index: int, live: bool) -> list:
        drawn = [self.draw_part(p, (index, i)) for i, p in enumerate(self.shown_parts(content, live))]
        if not drawn:
            drawn.append({"kind": "empty", "el": ui.markdown("_(empty)_"), "md": None})
        return drawn

    def refresh_answer(self, index: int, live: bool):
        """An answer's message brought up to its text: each part's content changed in place, new parts added after it;
        from the first part whose kind changed (a think block closing), the rest drawn again."""
        msg, drawn = self.answers[index]["slot"], self.answers[index]["drawn"]
        parts = self.shown_parts(self.state["history"][index]["content"] or "", live)
        keep = 0
        while keep < len(drawn) and keep < len(parts) and drawn[keep]["kind"] == parts[keep][0]:
            self.update_part(drawn[keep], parts[keep])
            keep += 1
        for d in drawn[keep:]:
            d["el"].delete()
        del drawn[keep:]
        with msg:
            for i in range(keep, len(parts)):
                drawn.append(self.draw_part(parts[i], (index, i)))
            if not drawn and not live:
                drawn.append({"kind": "empty", "el": ui.markdown("_(empty)_"), "md": None})
        self.draw_made(index)
        self.follow_chat()

    # ⭐ WHAT THE MODEL MADE IN A TURN, A CHIP EACH (NN-202, ⚖ 2026-11-22: *"generated things should be in the same
    #   attached folder as the uploaded ones, so you can download them from the ui"*): a file the turn's calls made or
    #   replaced among the conversation's files — the server says which, with the call's answer and in the
    #   conversation's history — is a chip under the answer, as an attachment's is above the message box, and a click
    #   downloads it.
    def draw_made(self, index: int):
        """The chips of what an answer's turn made, under its parts: drawn where the files changed, and put back at the
        answer's end where parts were drawn after them."""
        held = self.answers[index]
        made = tuple(self.state["history"][index].get("made") or ())
        row = held.get("chips")
        if made == held.get("made", ()):
            if row is not None and held["slot"].default_slot.children[-1] is not row:
                row.move(held["slot"])
            return
        if row is not None:
            row.delete()
        held["made"], held["chips"] = made, None
        if not made:
            return
        with held["slot"]:
            with ui.row().classes("items-center gap-1 flex-wrap mt-1") as held["chips"]:
                for path in made:
                    chip = ui.chip(path.split("/")[-1], icon="download").props(
                        "dense clickable color=green-2 text-color=green-9")
                    chip.tooltip("The model made %s in this turn. It is kept with this conversation's files, beside what "
                                 "you sent: click to download it." % path)
                    chip.on("click", lambda _, p=path: self.download_made(p))

    async def download_made(self, path: str):
        """A file the model made among the conversation's files, downloaded: by a one-time address of this UI's, which
        streams it from the server (▶ client.py's `file_ticket`) — said, where the file is there no more."""
        cid = self.state.get("active_id")
        if not cid:
            return
        try:
            if not any(f["path"] == "attached/" + path for f in await self.client.files(cid)):
                self.safe_notify("%s is no longer among this conversation's files" % path.split("/")[-1], type="warning")
                return
            self.page_client.download(self.client.file_ticket(cid, "attached/" + path), path.split("/")[-1])
        except Exception as ex:                             # noqa: BLE001 — said, not raised into the page
            self.safe_notify(f"the download failed: {ex}", type="negative")

    def follow_chat(self, force: bool = False):
        """The chat's view to its bottom where it is at the bottom already, or `force`d: a reader scrolled up stays put."""
        try:
            self.page_client.run_javascript("silvannChatFollow(%d, %s)" % (self.chat_scroll.id, "true" if force else "false"))
        except Exception:                                  # a page gone, its answer finishing
            pass

    def refresh_live(self, live: bool = True):
        """The streaming answer brought up to its text; `live` False settles it as the answer it ended as."""
        self.live_answer["dirty"] = False
        index = self.live_answer.get("index")
        if index is None or index not in self.answers or index != len(self.state["history"]) - 1:
            self.render_chat()
            return
        self.refresh_answer(index, live)

    def mark_live(self):
        """The streaming answer's text has grown: drawn by the next tick of the live timer — this page's, and any other
        page of this browser's following it."""
        self.live_answer["dirty"] = True
        live = LIVE.get(self.browser_id)
        if live is not None:
            live["seen"] += 1

    def follow_answer(self, live: dict):
        """This page's copy of the streaming answer brought up to its record's (▶ `LIVE`), where the page shows that
        conversation with the turn."""
        h = self.state["history"]
        if live["cart"] != self.state["active_id"] or len(h) != len(live["before"]) + len(live["turn"]):
            return
        text = live["turn"][-1]["content"]
        if h[-1]["content"] != text:
            h[-1]["content"] = text
        made = live["turn"][-1].get("made")                  # what the turn's calls made, for its chips (▶ `draw_made`)
        if made and h[-1].get("made") != made:
            h[-1]["made"] = list(made)

    def live_tick(self):
        """Ten times a second: the streaming answer copied and drawn where it grew; on a page that followed an answer
        another page asked for, the answer settled once it ends."""
        live = _streaming(self.browser_id)
        if live is not None:
            if self.live_answer.get("dirty") or live["seen"] != self.live_answer.get("seen"):
                self.live_answer["seen"] = live["seen"]
                self.follow_answer(live)
                self.refresh_live()
        elif self.live_answer.get("dirty"):
            self.refresh_live()
        elif self.live_answer.get("index") is not None and self.state["streaming"]:
            last = LIVE.get(self.browser_id)                     # ended on another page's stream: its last text, its ask
            if last is not None:
                self.follow_answer(last)
                if last.get("ask") and last["cart"] == self.state["active_id"]:
                    self.state["ask"] = last["ask"]
            self.state["streaming"] = False
            self.turn.set_send_btn(False)
            self.finish_live()

    def finish_live(self):
        """A turn over: its answer settled in place and the tail drawn — nothing above it drawn again."""
        self.refresh_live(live=False)
        self.live_answer.clear()
        self.render_tail()

    def draw_message(self, index: int, m: dict):
        """One message of the conversation into the current slot; a streaming answer's is kept for its updates."""
        _sent = (m["role"] == "user")
        # sent=user gives the bubble its style; self-end/self-start does the actual L/R
        # alignment (the column's align-items:flex-start otherwise keeps both on the left),
        # capped at 85% width so the bubbles don't span the whole row.
        with ui.chat_message(name=m["role"], sent=_sent
                             ).classes("self-end" if _sent else "self-start"
                             ).style("max-width: 85%") as msg:
            if _sent:
                ui.markdown(m["content"] or "_(empty)_")
            else:
                live = bool(self.state["streaming"]) and index == len(self.state["history"]) - 1
                self.answers[index] = {"slot": msg, "drawn": self.draw_answer(m["content"] or "", index, live)}
                self.draw_made(index)
                if live:
                    self.live_answer["index"] = index

    def render_tail(self):
        """Below the messages: "…" while an answer streams, else the model's ask waiting on the user's answer."""
        self.chat_tail_col.clear()
        with self.chat_tail_col:
            if self.state["streaming"]:
                ui.label("…").classes("opacity-50 self-center animate-pulse")
            elif self.state.get("ask"):                       # the model's command waits on the user
                with ui.card().classes("self-start bg-amber-50 border border-amber-300") as ask_card:
                    ui.label("The model asks to %s" % self.state["ask"]["question"]).classes("text-sm font-mono")
                    with ui.row().classes("gap-2"):
                        for choice in self.state["ask"]["choices"]:
                            ui.button(ASK_LABELS.get(choice, choice),
                                      on_click=lambda _, c=choice: self.turn.answer_ask(c)
                                      ).props("dense " + ("flat color=negative" if choice == "refuse" else "color=primary"))
                # ⛳ the ask brought into view once the browser has drawn it: a scroll in the same update as the card runs
                #   before the card has a height, and stops at the bottom of what was there — the ask out of sight below
                ui.timer(0.2, lambda: (self.chat_scroll.scroll_to(percent=1.0),
                                       ui.run_javascript("getHtmlElement(%d).scrollIntoView({block: 'end'})" % ask_card.id)),
                         once=True)
        self.follow_chat()

    def add_messages(self, start: int):
        """The conversation's messages from `start` on added below the others — a turn's, as it is sent."""
        if self.empty_note.get("el") is not None:
            self.empty_note.pop("el").delete()
        with self.chat_history_col:
            for index in range(start, len(self.state["history"])):
                self.draw_message(index, self.state["history"][index])
        self.render_tail()

    def render_chat(self):
        self.chat_history_col.clear()
        self.live_answer.clear()
        self.answers.clear()
        self.empty_note.clear()
        if self.state["active_id"]:
            self.chat_header.text = self.state.get("active_name") or self.state["active_id"]
            self.chat_header.classes(replace="text-lg font-semibold flex-grow")
        else:
            self.chat_header.text = "(no conversation active — pick one from the menu or click '+ new')"
            self.chat_header.classes(replace="text-sm opacity-60 flex-grow")
        self.show_context()
        with self.chat_history_col:
            if not self.state["history"] and self.state["active_id"]:
                self.empty_note["el"] = ui.label("(empty — say hi)").classes("opacity-50 text-sm self-center")
            for index, m in enumerate(self.state["history"]):
                self.draw_message(index, m)
        self.render_tail()
