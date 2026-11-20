"""The conversations: the drawer's list, one opened, made, renamed, exported, locked, deleted."""
import asyncio
import re
from typing import Optional

import httpx
from nicegui import ui

from .part import Part
from .session import _streaming, _with_turn


class Conversations(Part):
    """The conversations' list and what is done to one."""

    def wire(self):
        """The Activate button and the lock."""
        self.starting: dict = {"busy": False, "server": None}   # a first conversation in the making; the server it was looked for on

        self.chat.bind_btn.on("click", self.bind_active)

        self.chat.protected_icon.on("click", self.open_password_dialog)

    async def activate_cart(self, cart_id: str):
        """Lazy activate — fetches metadata from the server (which is cheap, just reads the
        sidecar JSON) and renders the chat history. Does NOT load VRAM. The user explicitly
        clicks the 'Activate' banner to /bind."""
        is_protected = False
        try:
            for c in (self.state.get("_last_carts") or []):
                if c["id"] == cart_id:
                    is_protected = bool(c.get("protected"))
                    break
        except Exception:
            pass
        if is_protected:
            pw = await self.prompt_password(cart_id)
            if pw is None:
                return
            self.client.set_password(cart_id, pw)
        # Unbind any previously-bound cart for this session — the server only counts ownership
        # on bind, but we keep the UI's `active_bound` consistent.
        prev = self.state.get("active_id")
        if prev and prev != cart_id and self.state.get("active_bound"):
            try:
                await self.client.unbind(prev)
            except Exception:
                pass
            self.state["active_bound"] = False
        try:
            try:
                resp = await self.client.activate(cart_id)
            except httpx.HTTPStatusError as ex:
                if ex.response.status_code == 401:
                    pw = await self.prompt_password(cart_id)
                    if pw is None:
                        return
                    self.client.set_password(cart_id, pw)
                    resp = await self.client.activate(cart_id)
                else:
                    raise
            self.state["active_id"] = cart_id
            self.state["active_name"] = resp.get("name") or cart_id[:8]
            live = _streaming(self.browser_id)
            # the conversation whose answer this browser is streaming: with its turn so far — the server's history has
            #   the turn once it ends
            self.state["history"] = _with_turn(live) if live is not None and live["cart"] == cart_id else resp.get("history", [])
            self.chat.show_context(used=resp.get("used", resp.get("next_pos", 0)), context=resp.get("context"))
            new_thinking = bool(resp.get("enable_thinking", True))
            self.state["active_thinking_mode"] = new_thinking
            self.chat.thinking_mode_toggle.set_value(new_thinking)
            self.state["active_tools"] = bool(resp.get("tools", False))
            self.chat.tools_toggle.set_value(self.state["active_tools"])
            self.state["ask"] = resp.get("ask")
            self.state["tools_mode"] = resp.get("tools_mode")
            self.chat.tools_mode.set_value(self.state["tools_mode"])
            try:
                off = (await self.client.tools_settings())["tools"] == "no"    # the server's config turns it off
                self.chat.tools_toggle.set_enabled(not off)
                self.chat.tools_mode.set_enabled(not off)
            except Exception:
                pass                                       # a server before the tool
            self.state["protected_active"] = bool(resp.get("protected", False))
            self.chat.protected_icon.props(f"icon={'lock' if self.state['protected_active'] else 'lock_open'}")
            # bound_by_self happens if the cart is already bound to OUR session (e.g., we
            # navigated away and back within the heartbeat window). Treat that as already-bound.
            self.state["active_bound"] = bool(resp.get("bound_by_self", False))
            if resp.get("bound_by_other"):
                self.chat.bind_banner_label.text = "Conversation is held by another session. Click Activate to take it over (will fail if they're still pinging)."
            else:
                self.chat.bind_banner_label.text = "Conversation loaded. Type a message, or click Activate, to bring the session online (~3s, allocates VRAM)."
            self.chat.render_chat()
            self.chat.follow_chat(force=True)
            await self.render_convs()
            # the model that made it, as the server's list says — the chat is emptied when another serves (ping)
            self.state["active_model"] = next((c.get("model_id") for c in self.state.get("_last_carts") or []
                                               if c["id"] == cart_id), None)
        except httpx.HTTPStatusError as ex:
            if ex.response.status_code == 401:
                self.safe_notify("wrong password", type="negative")
                self.client.set_password(cart_id, None)
            else:
                self.safe_notify(f"activate failed: {ex}", type="negative")
        except Exception as e:
            self.safe_notify(f"activate failed: {e}", type="negative")

    async def start_conversation(self, carts: Optional[list] = None) -> bool:
        """A conversation made and opened without the dialog, named as '+ new' names it — the model's first when it has
        none saved, or the one a message typed with none open goes to. True once it is the active one."""
        if self.starting["busy"]:
            return False
        self.starting["busy"] = True
        try:
            carts = await self.client.list_cartridges() if carts is None else carts
            resp = await self.client.create(name=f"chat {len(carts) + 1}")
            await self.activate_cart(resp["id"])
            return self.state["active_id"] == resp["id"]
        except Exception as ex:
            self.safe_notify(f"couldn't start a conversation: {ex}", type="negative")
            return False
        finally:
            self.starting["busy"] = False

    async def first_conversation(self, model: Optional[str]):
        """⭐ A MODEL WITH NO CONVERSATION SAVED GETS ONE, OPEN, so a user's first instinct — to type — works: looked for
        once a server answers (a model started here, the container's, one already up), again when the model changes."""
        here = (self.state["endpoint"], model)
        if self.state["active_id"] or self.state["streaming"] or self.starting["busy"] or self.starting["server"] == here:
            return
        try:
            carts = await self.client.list_cartridges()
        except Exception:
            return
        self.starting["server"] = here
        if not carts:
            await self.start_conversation(carts)

    async def bind_active(self):
        """Bring the active cartridge online — load it into VRAM, claim ownership. The
        'Activate' banner button calls this."""
        cart_id = self.state.get("active_id")
        if not cart_id:
            self.safe_notify("no conversation selected", type="warning")
            return
        self.chat.bind_btn.disable()
        try:
            resp = await self.client.bind(cart_id)
            self.state["active_bound"] = True
            self.state["active_name"] = resp.get("name") or self.state["active_name"]
            self.safe_notify("session online", type="positive")
        except httpx.HTTPStatusError as ex:
            if ex.response.status_code == 423:
                self.safe_notify("In use by another session — wait or try another conversation",
                                 type="negative")
            elif ex.response.status_code == 507:
                self.safe_notify("Out of VRAM — close another conversation first (or wait 2 min for idle eviction)",
                                 type="negative")
            elif ex.response.status_code == 401:
                self.safe_notify("wrong password — re-open the conversation to re-prompt",
                                 type="negative")
            else:
                self.safe_notify(f"activate failed: {ex.response.status_code} {ex.response.text}",
                                 type="negative")
        except Exception as e:
            self.safe_notify(f"activate failed: {e}", type="negative")
        finally:
            self.chat.bind_btn.enable()

    def toggle_conv_menu(self, cart_id: str):
        # Independent of activation. Toggle which conv's inline drawer is open (one at a time).
        if self.state.get("expanded_cart_id") == cart_id:
            self.state["expanded_cart_id"] = None
        else:
            self.state["expanded_cart_id"] = cart_id
        asyncio.create_task(self.render_convs())

    async def export_chat_history(self, cart_id: str, name: str):
        """Build a markdown transcript from the live state (if this is the active conv) or fetch
        history via activate. Downloads as <name>.md."""
        history = self.state["history"] if self.state["active_id"] == cart_id else []
        if not history:
            try:
                resp = await self.client.activate(cart_id)
                self.state["active_id"] = cart_id
                self.state["active_name"] = resp.get("name") or cart_id[:8]
                self.state["history"] = resp.get("history", [])
                history = self.state["history"]
                self.chat.render_chat()
                await self.render_convs()
            except Exception as e:
                self.safe_notify(f"couldn't load chat: {e}", type="negative")
                return
        lines = [f"# {name}\n"]
        for m in history:
            lines.append(f"## {m['role']}\n")
            lines.append(m["content"] or "")
            lines.append("")
        text = "\n".join(lines).encode("utf-8")
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name or cart_id)
        # Bypass context.client (which needs a slot stack that event-handler tasks lack) — call
        # the captured page_client directly.
        self.page_client.download(text, f"{safe}.md", "text/markdown")

    async def create_cart(self):
        # Open a dialog with the conversation's name, the model's context window and what the conversation takes in RAM
        # once put away — the server reports both via /estimate.
        try:
            existing = await self.client.list_cartridges()
        except Exception as e:
            self.safe_notify(f"couldn't reach server: {e}", type="negative")
            return
        try:
            est0 = await self.client.estimate_vram()
        except httpx.HTTPStatusError as ex:
            self.safe_notify(f"estimate failed: {ex.response.text}", type="negative")
            return
        except Exception as e:
            self.safe_notify(f"couldn't reach server: {e}", type="negative")
            return
        # ⭐ THE CONTEXT IS THE MODEL'S, set in its config when the server started — a conversation is created at that
        #   size and costs the card nothing more (its cache is allocated once, at boot, and shared). What the dialog
        #   shows is that window and what the conversation takes in RAM once it is put away, at its fullest.
        max_tokens = int(est0.get("max_tokens", 0))
        gb = float(est0.get("gb", 0.0))
        # the system prompt is the server's; its config's `user_system_prompt` lets a conversation's creator give one
        try:
            own_system = bool((await self.client.health()).get("user_system_prompt"))
        except Exception:
            own_system = False

        dlg = ui.dialog()
        with dlg, ui.card().classes("min-w-[420px] gap-2"):
            ui.label("New conversation").classes("text-lg font-bold")
            name_in = ui.input("Name", value=f"chat {len(existing) + 1}").classes("w-full")
            system_in = ui.textarea("System prompt", placeholder="empty: the server's own").props("autogrow") \
                .classes("w-full") if own_system else None
            if max_tokens:
                ui.label(f"Context window: {max_tokens:,} tokens — the model's, set in its config").classes("text-sm")
                ui.label(f"Saved, it takes up to {gb:.2f} GB of RAM").classes("text-xs opacity-70")
                cards = int(est0.get("num_gpus", 1) or 1)
                if cards > 1 and est0.get("per_card"):
                    ui.label(f"Over {cards} cards: {float(est0['per_card']['gb']):.2f} GB of its cache on each"
                             ).classes("text-xs opacity-70")

            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                ui.button("Cancel", on_click=dlg.close).props("flat")
                create_btn = ui.button("Create").props("color=primary unelevated")

            async def do_create():
                create_btn.disable()
                try:
                    resp = await self.client.create(name=name_in.value or None,
                                                    system=(system_in.value or None) if system_in else None)
                    dlg.close()
                    await self.activate_cart(resp["id"])
                except httpx.HTTPStatusError as ex:
                    self.safe_notify(f"create failed: {ex.response.text}", type="negative")
                    create_btn.enable()
                except Exception as ex:
                    self.safe_notify(f"create failed: {ex}", type="negative")
                    create_btn.enable()

            create_btn.on_click(do_create)

        dlg.open()

    async def rename_cart(self, cart_id: str, new_name: str):
        try:
            await self.client.rename(cart_id, new_name)
            if self.state["active_id"] == cart_id:
                self.state["active_name"] = new_name
                self.chat.render_chat()
            await self.render_convs()
        except Exception as e:
            self.safe_notify(f"rename failed: {e}", type="negative")

    async def delete_cart(self, cart_id: str):
        try:
            await self.client.delete(cart_id)
            if self.state["active_id"] == cart_id:
                self.state["active_id"] = None
                self.state["history"] = []
                self.chat.render_chat()
            await self.render_convs()
        except Exception as e:
            self.safe_notify(f"delete failed: {e}", type="negative")

    async def export_cart(self, cart_id: str):
        # ⛳ THROUGH AN HTTP ROUTE, NOT THE WEBSOCKET: an archive can be a few hundred MB (a conversation's recurrent
        #   state and cache), and sent as bytes over the page's websocket it drops the connection
        try:
            self.page_client.download(self.client.export_ticket(cart_id), f"{cart_id}.silvann.tar", "application/x-tar")
        except Exception as e:
            self.safe_notify(f"export failed: {e}", type="negative")

    async def on_upload(self, e):
        try:
            # NiceGUI 3.x upload event shape (see on_attach_upload note).
            blob = await e.file.read()
            resp = await self.client.import_bytes(blob, filename=e.file.name or "cart.tar")
            self.safe_notify(f"imported {resp['id']}", type="positive")
            await self.render_convs()
        except httpx.HTTPStatusError as ex:
            self.safe_notify(f"import refused: {ex.response.text}", type="negative")
        except Exception as ex:
            self.safe_notify(f"import failed: {ex}", type="negative")

    async def prompt_password(self, cart_id: str) -> Optional[str]:
        """Modal dialog asking for the password to unlock a protected cartridge. Returns the
        entered string (may be empty) or None if cancelled.

        Wraps the dialog construction in `with page_client:` so element creation works even when
        this is called from an async event-handler task (which doesn't auto-inherit the page's
        slot stack)."""
        result: dict = {"value": None}
        with self.page_client:
            dlg = ui.dialog()
            with dlg, ui.card():
                ui.label(f"Enter password for {cart_id}").classes("font-bold")
                inp = ui.input(password=True, password_toggle_button=True).classes("w-80")
                with ui.row().classes("justify-end w-full"):
                    ui.button("cancel", on_click=dlg.close).props("flat")
                    def ok():
                        result["value"] = inp.value
                        dlg.close()
                    ui.button("unlock", on_click=ok).props("color=primary")
        dlg.open()
        await dlg
        return result["value"]

    def open_password_dialog(self):
        """Opened by clicking the lock icon. Set a new password (when unprotected), change it
        (when protected, enter new+confirm), or remove protection. Saves happen automatically
        after every turn so there's no separate save action here."""
        if not self.state["active_id"]:
            self.safe_notify("activate a conversation first", type="warning")
            return
        cart_id = self.state["active_id"]
        is_protected = bool(self.state.get("protected_active", False))
        with self.page_client, ui.dialog() as dlg, ui.card().classes("w-96"):
            if is_protected:
                ui.label("Change password").classes("font-bold")
                ui.label("This conversation is locked. Enter a new password to change it, or click "
                         "Remove to make it public.").classes("text-xs opacity-70")
            else:
                ui.label("Set a password").classes("font-bold")
                ui.label("Lock this conversation so other users on the same server can't read it. "
                         "You'll be prompted for the password whenever you open it from a fresh "
                         "tab/device.").classes("text-xs opacity-70")
            pw1 = ui.input("new password", password=True, password_toggle_button=True
                           ).classes("w-full")
            pw2 = ui.input("confirm", password=True, password_toggle_button=True
                           ).classes("w-full")
            err = ui.label("").classes("text-red-500 text-sm")

            async def apply(password: Optional[str]):
                try:
                    resp = await self.client.save(cart_id, password=password)
                except httpx.HTTPStatusError as ex:
                    err.text = f"server refused: {ex.response.status_code} {ex.response.text}"
                    return
                except Exception as ex:
                    err.text = f"failed: {ex}"
                    return
                if password:
                    self.client.set_password(cart_id, password)
                else:
                    self.client.set_password(cart_id, None)
                self.state["protected_active"] = bool(resp.get("protected", False))
                self.chat.protected_icon.props(f"icon={'lock' if self.state['protected_active'] else 'lock_open'}")
                self.safe_notify(
                    "password set" if self.state["protected_active"] else "password removed",
                    type="positive")
                dlg.close()
                await self.render_convs()

            async def submit_set():
                if not pw1.value:
                    err.text = "password cannot be empty"
                    return
                if pw1.value != pw2.value:
                    err.text = "passwords don't match"
                    return
                await apply(pw1.value)

            async def submit_remove():
                await apply("")

            with ui.row().classes("justify-end w-full mt-2 gap-2"):
                ui.button("cancel", on_click=dlg.close).props("flat")
                if is_protected:
                    ui.button("remove", on_click=submit_remove).props("flat color=red")
                ui.button(("change" if is_protected else "set"),
                          on_click=submit_set).props("color=primary")
        dlg.open()

    def open_rename_dialog(self, cart_id: str, cur: str):
        with self.page_client, ui.dialog() as dlg, ui.card():
            ui.label(f"Rename {cart_id}").classes("font-bold")
            inp = ui.input(value=cur).classes("w-80")
            with ui.row().classes("justify-end w-full"):
                ui.button("cancel", on_click=dlg.close).props("flat")
                async def ok():
                    await self.rename_cart(cart_id, inp.value)
                    dlg.close()
                ui.button("rename", on_click=ok)
        dlg.open()

    def open_delete_dialog(self, cart_id: str):
        with self.page_client, ui.dialog() as dlg, ui.card():
            ui.label(f"Delete {cart_id}?").classes("font-bold")
            ui.label("This is irreversible — the cartridge dir is removed from disk."
                     ).classes("text-sm opacity-70")
            with ui.row().classes("justify-end w-full"):
                ui.button("cancel", on_click=dlg.close).props("flat")
                async def ok():
                    await self.delete_cart(cart_id)
                    dlg.close()
                ui.button("delete", on_click=ok).props("color=red")
        dlg.open()

    # ----- Conversations sub-tree under the Chat menu -----
    async def render_convs(self):
        # Wrap the whole thing in page_client so element creation works regardless of which task
        # we're called from (event handlers don't auto-inherit the slot stack).
        with self.page_client:
            self.shell.convs_col.clear()
            with self.shell.convs_col:
                try:
                    carts = await self.client.list_cartridges()
                except Exception as e:
                    ui.label(f"server down: {e}").classes("text-red-500 text-xs px-2")
                    # Still show controls so user can create/import once server is up.
                    with ui.row().classes("w-full gap-1 px-2 py-1"):
                        ui.button("+ new", on_click=self.create_cart
                                  ).props("size=sm dense flat color=primary")
                    return
                # Stash for activate_cart's "is this cart protected?" lookup.
                self.state["_last_carts"] = carts
                if carts:
                    for c in carts:
                        active = (c["id"] == self.state["active_id"])
                        expanded = (c["id"] == self.state.get("expanded_cart_id"))
                        with ui.column().classes(
                            "w-full silvann-conv-item gap-0"
                            + (" silvann-conv-active" if active else "")
                        ):
                            with ui.row().classes("w-full items-center gap-1 no-wrap min-w-0"):
                                with ui.row().classes("flex-grow items-center gap-1 cursor-pointer min-w-0"
                                                       ).on("click",
                                                            lambda cid=c["id"]: asyncio.create_task(self.activate_cart(cid))):
                                    if c.get("protected"):
                                        ui.icon("lock", size="xs").classes("text-slate-500")
                                    elif c["loaded"]:
                                        ui.icon("memory", size="xs").classes("text-green-500")
                                    else:
                                        ui.icon("chat_bubble_outline", size="xs").classes("opacity-50")
                                    ui.label(c["name"] or c["id"][:8]).classes(
                                        "text-sm flex-grow truncate"
                                    ).style("min-width: 0;")
                                    if not c["compatible"]:
                                        ui.icon("warning", size="xs").classes("text-orange-500")
                                ui.button(icon="expand_less" if expanded else "expand_more",
                                          on_click=lambda cid=c["id"]: self.toggle_conv_menu(cid)
                                          ).props("flat dense round size=xs").classes("opacity-70")
                            if expanded:
                                with ui.column().classes("w-full gap-0 pl-5 py-1"):
                                    ui.button("Rename", icon="edit",
                                              on_click=lambda cid=c["id"], n=c["name"]:
                                                  self.open_rename_dialog(cid, n)
                                              ).props("flat dense size=sm align=left").classes("w-full justify-start")
                                    ui.button("Export full session", icon="archive",
                                              on_click=lambda cid=c["id"]:
                                                  asyncio.create_task(self.export_cart(cid))
                                              ).props("flat dense size=sm align=left").classes("w-full justify-start")
                                    ui.button("Export chat history (.md)", icon="description",
                                              on_click=lambda cid=c["id"], n=c["name"]:
                                                  asyncio.create_task(self.export_chat_history(cid, n))
                                              ).props("flat dense size=sm align=left").classes("w-full justify-start")
                                    ui.button("Delete", icon="delete",
                                              on_click=lambda cid=c["id"]:
                                                  self.open_delete_dialog(cid)
                                              ).props("flat dense size=sm align=left color=red").classes("w-full justify-start")
                else:
                    ui.label("(no conversations yet)").classes("opacity-50 text-xs px-2 py-1")
                ui.separator().classes("my-1")
                with ui.column().classes("w-full gap-1 px-1 py-1"):
                    ui.button("+ new", on_click=self.create_cart
                              ).props("color=primary unelevated").classes("w-full")
                    ui.upload(on_upload=self.on_upload, auto_upload=True, max_files=1
                              ).props('label="import" flat').classes("w-full")
