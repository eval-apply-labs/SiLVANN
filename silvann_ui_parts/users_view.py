"""The accounts of this machine: the switch between single and multi user, the Users view, the Account view."""
import asyncio
import os

from nicegui import ui

from .local import LISTEN
from .part import Part
from .run import _this_address, listen_host, restart_ui, start_again_later
from .server_process import SERVER
from .sign_in import LOCAL_USERS, USERS_MOD, gate


class UsersMode(Part):
    """Single user or multi user: the switch, in the Options view."""

    def build(self):
        """The "Single user or multi user" section."""
        # ⭐ SINGLE USER OR MULTI USER (NN-196, ⚖ 2026-11-18 *"the admin consolle has the single user vs multi user
        #   option in the config panel"*; NN-197: the machine's mode, its accounts the same in both)
        ui.separator().classes("mt-4")
        ui.label("Single user or multi user").classes("font-bold")
        windows_note = (" On Windows a model that works in its account's folder runs no commands — (powershell …), "
                        "(bash …), the tools it adds — as Windows cannot hold them there; an admin's on a single user's "
                        "machine runs them." if os.name == "nt" else "")
        if LOCAL_USERS is None:
            ui.label("This UI has no accounts of its own to switch: it talks to a server elsewhere, whose accounts are "
                     "its machine's.").classes("text-xs opacity-70 max-w-2xl whitespace-normal")
        elif not self.multi:
            ui.label("Single user: one account at a time signs in, and has this machine until it signs out, ten "
                     "minutes pass with nothing asked of it, or SiLVANN "
                     "is opened again; an admin's model reaches the machine as the tools file lets it, and SiLVANN is "
                     "reached from this machine alone. Multi user: the accounts are signed in side by side, each one's "
                     "model works in its own folder — yours too — and SiLVANN listens on your network. The switch stops "
                     "the model and starts SiLVANN again; you stay signed in." + windows_note
                     ).classes("text-xs opacity-70 max-w-2xl whitespace-normal")
            self.sw_btn = ui.button("switch to multi user", icon="group").props("dense outline")
            self.sw_said = ui.label("").classes("text-sm text-red-700")

            self.sw_btn.on("click", self.do_multi)
        else:
            port = LISTEN.get("port") or 8080
            if LISTEN.get("host") in ("127.0.0.1", "localhost", "::1"):
                reach = ("This UI listens on this machine alone (--private, or the address --host names): other "
                         "machines cannot reach it.")
            elif os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
                reach = ("SiLVANN runs in a container: other machines reach it once the host publishes its port on "
                         "the network — `setup.sh update` on the host (docker/README.md).")
            else:
                reach = ("Others on your network reach SiLVANN at http://%s:%d once this machine's firewall lets port "
                         "%d in." % (_this_address(), port, port))
            ui.label("Multi user: the accounts are signed in side by side, each one's model in its own folder; an admin "
                     "changes these Options and the accounts (Users). " + reach + windows_note
                     ).classes("text-xs opacity-70 max-w-2xl whitespace-normal")
            back_btn = ui.button("back to single user", icon="person").props("dense outline")

            back_btn.on("click", self.do_single)

    async def confirmed(self, title, text, button) -> bool:
        with self.page_client, ui.dialog() as dlg, ui.card().classes("gap-2"):
            ui.label(title).classes("font-bold")
            ui.label(text).classes("text-xs opacity-70 max-w-md whitespace-normal")
            with ui.row().classes("justify-end w-full gap-2"):
                ui.button("cancel", on_click=lambda: dlg.submit(False)).props("flat")
                ui.button(button, on_click=lambda: dlg.submit(True)).props("color=primary")
        return bool(await dlg)

    async def do_switch(self, switch, said):
        """The model stopped, the mode switched (▶ users.Users.multi, .single), and SiLVANN as it is in that mode."""
        if not gate(self.state)[2]:
            ui.navigate.reload()
            return
        again = SERVER.args if SERVER.is_running() else None
        await asyncio.to_thread(SERVER.stop)
        try:
            await asyncio.to_thread(switch)
        except USERS_MOD.Refused as ex:
            self.safe_notify("not switched: %s" % ex, type="negative")
            if again:
                SERVER.start(*again)
            return
        self.switched(again, said)

    async def do_multi(self):
        if await self.confirmed("Switch to multi user?",
                                "The accounts are signed in side by side from now on, each one's model in its own "
                                "folder — yours too, which then reaches nothing else of this machine. SiLVANN listens on "
                                "your network. The model stops and SiLVANN starts again; you stay signed in.",
                                "switch to multi user"):
            await self.do_switch(LOCAL_USERS.multi, "multi user: the accounts side by side")

    async def do_single(self):
        if await self.confirmed("Back to single user?",
                                "One account at a time from now on: everyone is signed out, and the next to sign in "
                                "has this machine until it signs out or SiLVANN is opened again. An admin's model "
                                "reaches the machine as the tools file lets it. SiLVANN is reached from this machine "
                                "alone. The model stops and SiLVANN starts again.",
                                "back to single user"):
            await self.do_switch(LOCAL_USERS.single, "single user: sign in again")

    def switched(self, again, said):
        """After the switch: the UI started again where its address changes (▶ restart_ui), the model the switch stopped
        started by the next one; else that model started again here, and every page — this one now, the others at their
        next look (▶ gate) — drawn afresh."""
        self.safe_notify(said, type="positive")
        if listen_host(LISTEN.get("named"), LISTEN.get("private"))[0] != LISTEN.get("host"):
            start_again_later(again)
            asyncio.get_running_loop().call_later(1.0, restart_ui)
            return
        if again:
            SERVER.start(*again)
        ui.navigate.reload()


class UsersView(Part):
    """The Users view: who signs in to this machine, an admin's to change."""

    def build(self):
        """The Users view's widgets, an admin's alone."""
        # --- Users view (NN-196): who signs in to this machine, an admin's to change ---
        self.users_view = ui.column().classes("silvann-main w-full p-4 gap-2 overflow-auto")
        with self.users_view:
            if self.me is not None and self.admin:
                ui.label("Users").classes("text-lg font-bold")
                ui.label(("This machine is single user: one of these accounts at a time signs in, and has it until it "
                          "signs out or SiLVANN is opened again. " if not self.multi else "")
                         + "Each account signs in and sees its own conversations. An admin also changes the Options — "
                         "the models, the configs, the tools — and these accounts"
                         + ("; on this single user's machine an admin's model reaches the machine as the tools file "
                            "lets it, and any other account's works in its own folder. " if not self.multi else
                            ", and each one's model works in its own folder. ")
                         + "An account made at a sign-on (sso.json) signs in with its organisation's account.").classes(
                    "text-xs opacity-70 max-w-2xl whitespace-normal")

                @ui.refreshable
                def users_table():
                    with ui.grid(columns="auto auto auto auto").classes("items-center gap-x-6 gap-y-1"):
                        for head in ("user", "", "signs in with", ""):
                            ui.label(head).classes("text-xs opacity-70")
                        for n in LOCAL_USERS.names():
                            ui.label(n + (" (you)" if n == self.me else "")).classes("font-mono text-sm")
                            ui.label("admin" if LOCAL_USERS.admin(n) else "").classes("text-xs text-green-700")
                            ways = (["a password"] if LOCAL_USERS.has_password(n) else []) + [
                                "%s's account %s" % (iss, email or sub) for iss, sub, email in LOCAL_USERS.accounts(n)]
                            ui.label(", ".join(ways) or "nothing yet").classes("text-xs")
                            with ui.row().classes("gap-1 no-wrap"):
                                ui.button("not admin" if LOCAL_USERS.admin(n) else "make admin",
                                          on_click=lambda _, n=n: self.toggle_admin(n)).props("dense flat size=sm")
                                if LOCAL_USERS.has_password(n) or not LOCAL_USERS.accounts(n):
                                    ui.button("password", on_click=lambda _, n=n: self.set_password(n)
                                              ).props("dense flat size=sm")
                                ui.button("remove", on_click=lambda _, n=n: self.remove_user(n)
                                          ).props("dense flat size=sm color=negative")
                self.users_table = users_table

                self.users_table()
                ui.separator().classes("mt-2")
                ui.label("Add a user").classes("font-bold")
                with ui.row().classes("items-end gap-2"):
                    self.add_name = ui.input("name").props("dense").classes("w-48")
                    self.add_pw1 = ui.input("password", password=True, password_toggle_button=True).props("dense").classes("w-48")
                    self.add_pw2 = ui.input("again", password=True).props("dense").classes("w-40")
                    self.add_admin = ui.checkbox("an admin")
                    add_btn = ui.button("add", icon="person_add").props("dense outline")
                self.add_said = ui.label("").classes("text-sm text-red-700")

                add_btn.on("click", self.do_add)

    def admin_now(self) -> bool:
        if gate(self.state)[2]:
            return True
        self.safe_notify("an admin's alone", type="negative")
        ui.navigate.reload()
        return False

    def toggle_admin(self, n):
        if not self.admin_now():
            return
        try:
            LOCAL_USERS.set_admin(n, not LOCAL_USERS.admin(n))
        except USERS_MOD.Refused as ex:
            self.safe_notify(str(ex), type="warning")
            return
        if n == self.me and not LOCAL_USERS.admin(self.me):
            ui.navigate.reload()                   # an admin no more: this page's Options and Users go
            return
        self.users_table.refresh()

    async def set_password(self, n):
        if not self.admin_now():
            return
        with self.page_client, ui.dialog() as dlg, ui.card().classes("gap-2"):
            ui.label("A new password for %s" % n).classes("font-bold")
            ui.label("Their sign-ins end: they sign in again with it.").classes("text-xs opacity-70")
            p1 = ui.input("new password", password=True, password_toggle_button=True).classes("w-72")
            p2 = ui.input("again", password=True).classes("w-72")
            err = ui.label("").classes("text-red-500 text-sm")

            def apply():
                if not p1.value or p1.value != p2.value:
                    err.text = "the two passwords differ" if p1.value else "a password, please"
                    return
                dlg.submit(p1.value)
            with ui.row().classes("justify-end w-full gap-2"):
                ui.button("cancel", on_click=lambda: dlg.submit(None)).props("flat")
                ui.button("set", on_click=apply).props("color=primary")
        pw = await dlg
        if not pw or not self.admin_now():
            return
        await asyncio.to_thread(LOCAL_USERS.passwd, n, pw)
        if n == self.me:                                # your own: this browser signed in again with it
            self.state["user_token"] = await asyncio.to_thread(LOCAL_USERS.issue, self.me)
            self.client.use_token(self.page.token_for(self.state["endpoint"]))
        self.safe_notify("%s's password set" % n, type="positive")
        self.users_table.refresh()

    async def remove_user(self, n):
        if not self.admin_now():
            return
        held = LOCAL_USERS.conversations(n)
        with self.page_client, ui.dialog() as dlg, ui.card().classes("gap-2"):
            ui.label("Remove %s?" % n).classes("font-bold")
            ui.label("Their folder goes: %s." % ("their %d conversation%s, their workspace and settings with it"
                                                  % (held, "" if held == 1 else "s") if held else
                                                  "their workspace and settings")).classes("text-xs opacity-70")
            with ui.row().classes("justify-end w-full gap-2"):
                ui.button("cancel", on_click=lambda: dlg.submit(False)).props("flat")
                ui.button("remove", on_click=lambda: dlg.submit(True)).props("color=negative")
        if not await dlg or not self.admin_now():
            return
        try:
            await asyncio.to_thread(LOCAL_USERS.remove, n, True)
        except USERS_MOD.Refused as ex:
            self.safe_notify(str(ex), type="warning")
            return
        self.safe_notify("%s removed" % n, type="info")
        if n == self.me:
            ui.navigate.reload()
            return
        self.users_table.refresh()

    async def do_add(self):
        if not self.admin_now():
            return
        name = (self.add_name.value or "").strip().lower()
        if not self.add_pw1.value or self.add_pw1.value != self.add_pw2.value:
            self.add_said.text = "the two passwords differ" if self.add_pw1.value else "a user needs a password"
            return
        try:
            await asyncio.to_thread(LOCAL_USERS.add, name, self.add_pw1.value, bool(self.add_admin.value))
        except USERS_MOD.Refused as ex:
            self.add_said.text = str(ex)
            return
        self.add_said.text = ""
        self.add_name.value = self.add_pw1.value = self.add_pw2.value = ""
        self.add_admin.value = False
        self.safe_notify("%s added: they sign in with that password" % name, type="positive")
        self.users_table.refresh()


class AccountView(Part):
    """The Account view: the signed-in user's own."""

    def build(self):
        """The Account view's widgets."""
        # --- Account view (NN-196): the signed-in user's own ---
        self.account_view = ui.column().classes("silvann-main w-full p-4 gap-2 overflow-auto")
        with self.account_view:
            if self.me is not None:
                ui.label("Account").classes("text-lg font-bold")
                ui.label("Signed in as %s%s." % (self.me, ", an admin" if self.admin else "")).classes("text-sm")
                for iss, sub, email in LOCAL_USERS.accounts(self.me):
                    ui.label("You also sign in with %s's account %s." % (iss, email or sub)).classes("text-xs opacity-70")
                if LOCAL_USERS.has_password(self.me):
                    ui.label("Your password").classes("font-bold mt-2")
                    with ui.row().classes("items-end gap-2"):
                        self.acc_now = ui.input("the one you have", password=True).props("dense").classes("w-48")
                        self.acc_pw1 = ui.input("a new one", password=True, password_toggle_button=True).props("dense"
                                                                                                        ).classes("w-48")
                        self.acc_pw2 = ui.input("again", password=True).props("dense").classes("w-40")
                        acc_btn = ui.button("change", icon="key").props("dense outline")
                    self.acc_said = ui.label("").classes("text-sm text-red-700")

                    acc_btn.on("click", self.change_password)
                else:
                    ui.label("You sign in with your organisation's account; an admin can give you a password as well."
                             ).classes("text-xs opacity-70")
                ui.button("sign out", icon="logout", on_click=lambda: asyncio.create_task(self.remote.do_sign_out())
                          ).props("dense flat").classes("mt-2")

    async def change_password(self):
        if not self.acc_pw1.value or self.acc_pw1.value != self.acc_pw2.value:
            self.acc_said.text = "the two new passwords differ" if self.acc_pw1.value else "a new password, please"
            return
        try:                                               # checked as a sign-in's is: a wrong one costs a minute
            theirs = await asyncio.to_thread(LOCAL_USERS.password_ok, self.me, self.acc_now.value or "",
                                             self.page_client.ip or None)
        except USERS_MOD.Wait as ex:
            self.acc_said.text = str(ex)
            return
        if not theirs:
            self.acc_said.text = "that is not your password: the next try in %d seconds" % USERS_MOD.WAIT
            return
        await asyncio.to_thread(LOCAL_USERS.passwd, self.me, self.acc_pw1.value)
        self.state["user_token"] = await asyncio.to_thread(LOCAL_USERS.issue, self.me)   # signed in again with it
        self.client.use_token(self.page.token_for(self.state["endpoint"]))
        self.acc_now.value = self.acc_pw1.value = self.acc_pw2.value = self.acc_said.text = ""
        self.safe_notify("your password changed: your other sign-ins ended", type="positive")
