"""Another machine's server with users of its own: its sign-in, asked for in a dialog."""
import asyncio

import httpx
from nicegui import ui

from .part import Part
from .sign_in import LOCAL_USERS, _said, fresh_session


class RemoteSignIn(Part):
    """The sign-in dialog of a server elsewhere that has users."""

    async def build(self):
        """The dialog, the signed-in row, and the dialog opened where the server asks for a sign-in."""
        # ⭐ ANOTHER MACHINE'S SERVER, WHOSE ACCOUNTS ARE ITS OWN (NN-150): its sign-in, asked for in a dialog when it answers
        #   401 — a token this browser keeps for that server (`server_tokens`). This machine's own sign-in is a page of its
        #   own (▶ sign_in_page): a page whose sign-in ended is drawn again, and that is the page it gets.
        self.sign_in_dialog = ui.dialog().props("persistent").mark("sign-in")
        with self.sign_in_dialog, ui.card().classes("w-80 gap-2"):
            ui.label("Sign in").classes("text-lg")
            self.sign_in_about = ui.label("").classes("text-sm opacity-70")
            self.sign_in_name = ui.input("name").classes("w-full")
            self.sign_in_password = ui.input("password", password=True, password_toggle_button=True).classes("w-full")
            self.sign_in_said = ui.label("").classes("text-sm text-red-700")

            self.sign_in_password.on("keydown.enter", lambda: asyncio.create_task(self.do_sign_in()))
            ui.button("Sign in", on_click=self.do_sign_in).classes("w-full")

            with ui.column().classes("w-full gap-1") as self.sso_box:
                ui.separator()
                self.sso_button = ui.button("Sign in", on_click=self.do_sso).props("outline").classes("w-full").mark("sign-on")
            self.sso_box.set_visibility(False)

        self.client.on_signed_out = self.need_sign_in

        with self.chat.chat_header.parent_slot.parent:
            with ui.row().classes("items-center gap-1 no-wrap") as self.signed_row:
                signed_label = ui.label("").classes("text-xs opacity-70")
                ui.button("sign out", on_click=self.do_sign_out).props("flat dense size=sm")
        self.signed_row.set_visibility(False)
        try:
            h_users = await self.client.health()
        except Exception:
            h_users = {}
        self.asking = False
        if h_users.get("users") and not h_users.get("user"):     # a server whose users are not this machine's
            self.asking = True
            await self.need_sign_in()
        elif self.me is not None or h_users.get("user"):
            # this machine's account, and the one a server elsewhere answers this browser as, where it is another's
            there = h_users.get("user") if (self.state.get("server_tokens") or {}).get(self.client.endpoint) else None
            signed_label.text = "signed in as %s" % (self.me or h_users["user"]) + (
                " · %s at the server" % there if self.me is not None and there else "")
            self.signed_row.set_visibility(True)

    async def do_sign_in(self):
        try:
            got = await self.client.sign_in((self.sign_in_name.value or "").strip().lower(), self.sign_in_password.value or "")
        except httpx.HTTPStatusError as ex:
            self.sign_in_said.text = _said(ex.response)   # a wrong password, the minute after one, a machine in use
            return
        except httpx.HTTPError as ex:
            self.sign_in_said.text = "the server does not answer: %s" % ex
            return
        self.state.setdefault("server_tokens", {})[self.client.endpoint] = got["token"]
        self.state["server_user"] = got["name"]
        self.sign_in_dialog.close()
        ui.navigate.reload()                          # every panel again, as the signed-in user

    # the server's provider (NN-157): its page, then back to /_silvann/sso with a code the server checks
    async def do_sso(self):
        try:
            got = await self.client.sso_start()
        except httpx.HTTPStatusError as ex:
            self.sign_in_said.text = _said(ex.response)
            return
        except httpx.HTTPError as ex:
            self.sign_in_said.text = "the server does not answer: %s" % ex
            return
        self.state["sso_state"] = got["state"]             # the page coming back checks the browser is this one
        ui.navigate.to(got["url"])

    async def need_sign_in(self):
        if self.me is not None and LOCAL_USERS.signed(self.state.get("user_token"), use=False) is None:
            ui.navigate.reload()                          # this machine's sign-in ended: its own page (▶ sign_in_page)
            return
        (self.state.get("server_tokens") or {}).pop(self.client.endpoint, None)
        self.sign_in_about.text = "The server at %s asks for a sign-in: an account of its own machine's." % self.client.endpoint
        self.client.use_token(self.page.token_for(self.client.endpoint))
        self.signed_row.set_visibility(False)
        self.sign_in_said.text = ""
        try:
            provider = await self.client.sso()
        except httpx.HTTPError:
            provider = {}
        if provider.get("on"):
            self.sso_button.text = "Sign in with %s" % provider.get("label", "your organisation")
        self.sso_box.set_visibility(bool(provider.get("on")))
        self.sign_in_dialog.open()

    async def do_sign_out(self):
        if (self.state.get("server_tokens") or {}).get(self.client.endpoint):   # a server elsewhere: its sign-in ended there
            try:
                await self.client.sign_out()
            except httpx.HTTPError:
                pass
            self.state["server_tokens"].pop(self.client.endpoint, None)
        if self.me is not None and self.state.get("user_token"):
            # every process's — and on a single user's machine the account's every sign-in: it leaves the machine (▶ users.py)
            await asyncio.to_thread(LOCAL_USERS.sign_out, self.state["user_token"])
            fresh_session(self.state)                          # nothing of theirs left for the next on this browser
        ui.navigate.reload()
