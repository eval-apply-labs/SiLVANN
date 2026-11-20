"""The accounts of this machine: the switch between single and multi user, the Users view, the Account view."""
import asyncio
import os

from nicegui import ui

from .local import LISTEN
from .part import Part
from .run import _this_address, listen_host, restart_ui, start_again_later
from .server_process import SERVER
from .sign_in import LOCAL_USERS, USERS_MOD, _runtime_module, gate

SPACE = _runtime_module("space") if LOCAL_USERS is not None else None      # an account's space and its files (NN-200)


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
                    with ui.grid(columns="auto auto auto auto auto").classes("items-center gap-x-6 gap-y-1"):
                        for head in ("user", "", "signs in with", "space", ""):
                            ui.label(head).classes("text-xs opacity-70")
                        for n in LOCAL_USERS.names():
                            ui.label(n + (" (you)" if n == self.me else "")).classes("font-mono text-sm")
                            ui.label("admin" if LOCAL_USERS.admin(n) else "").classes("text-xs text-green-700")
                            ways = (["a password"] if LOCAL_USERS.has_password(n) else []) + [
                                "%s's account %s" % (iss, email or sub) for iss, sub, email in LOCAL_USERS.accounts(n)]
                            ui.label(", ".join(ways) or "nothing yet").classes("text-xs")
                            if SPACE is not None:              # what the account's folder holds, of the most it may
                                limit = LOCAL_USERS.space(n)
                                ui.button("%s of %s" % (SPACE.said(SPACE.used(LOCAL_USERS, n)),
                                                        SPACE.said(limit) if limit else "no limit"),
                                          on_click=lambda _, n=n: self.set_space(n)).props("dense flat no-caps size=sm")
                            else:
                                ui.label("")
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

    async def set_space(self, n):
        """An account's space (NN-200, ⚖ 2026-11-20 *"maybe the admin can assign a per user allowed space, and the system
        can refuse to save things past that"*): the most its folder may hold — a size, or none."""
        if not self.admin_now():
            return
        with self.page_client, ui.dialog() as dlg, ui.card():
            ui.label("%s's space" % n).classes("font-bold")
            ui.label("The most the account's folder may hold — its conversations, their states and files, its workspace. "
                     "Past it the server saves no more for the account: a file sent, a picture, a new conversation, a "
                     "message; what it has is still read, and deleted or archived in its Account. A size — 5GB, 500MB — "
                     "or none.").classes("text-xs opacity-70 max-w-md whitespace-normal")
            limit = LOCAL_USERS.space(n)
            box = ui.input("space", value=SPACE.said(limit).replace(" ", "") if limit else "none").props("dense").classes("w-40")
            said = ui.label("").classes("text-sm text-red-700")

            def ok():
                try:
                    LOCAL_USERS.set_space(n, SPACE.parse(box.value))
                    dlg.close()
                    self.users_table.refresh()
                except Exception as ex:                     # noqa: BLE001 — a size not read as one: said
                    said.text = str(ex)
            with ui.row().classes("justify-end w-full"):
                ui.button("cancel", on_click=dlg.close).props("flat")
                ui.button("set", on_click=ok).props("color=primary")
        dlg.open()

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
                if SPACE is not None:
                    self.build_space()

    # ⭐ THE ACCOUNT'S SPACE (NN-200, ⚖ 2026-11-20: *"the user account can show how much available space you have, what
    #   is occupying it"* · *"can be seen/modified in the user space anyway by deleting the conversation or deleting the
    #   file from the conversation via the ui, that allows you to do a mini file explorer"*): what the account's folder
    #   holds on this machine — each conversation's transcript, state and files, and the workspace, whose attached/ is the
    #   shared folder every conversation of the account reads — with what is deleted, archived and uploaded from here.
    def build_space(self):
        self.files_at = None                               # (where, path): "shared" or a conversation's id, a folder in it
        ui.separator().classes("mt-3")
        ui.label("Your space").classes("font-bold")
        ui.label("What your account keeps on this machine. A conversation holds its transcript, its state — what the "
                 "model holds of it, the larger part — and its own files: what you sent to it and the pictures made in it, "
                 "private to it and deleted with it. Archiving a conversation gives up its state and keeps the rest: it "
                 "is still shown, no more is sent to it, and it is not opened again. The shared folder is attached/ in your workspace: what you "
                 "put there every conversation of yours can read.").classes("text-xs opacity-70 max-w-2xl whitespace-normal")

        @ui.refreshable
        def space_table():
            u = SPACE.usage(LOCAL_USERS, self.me)
            if u["limit"]:
                ui.linear_progress(value=min(1.0, u["used"] / u["limit"]), show_value=False).classes("max-w-2xl")
                ui.label("%s used of %s — %s left" % (SPACE.said(u["used"]), SPACE.said(u["limit"]), SPACE.said(u["left"]))
                         ).classes("text-sm" + (" text-red-700" if u["left"] == 0 else ""))
            else:
                ui.label("%s used — no limit is set for this account" % SPACE.said(u["used"])).classes("text-sm")
            with ui.grid(columns="auto auto auto auto auto auto").classes("items-center gap-x-5 gap-y-1"):
                for head in ("conversation", "transcript", "state", "files", "all", ""):
                    ui.label(head).classes("text-xs opacity-70")
                for c in u["conversations"]:
                    with ui.row().classes("items-center gap-1 no-wrap"):
                        ui.label(c["name"]).classes("text-sm")
                        ui.label(c["model"]).classes("text-xs opacity-50")
                        if c["archived"]:
                            ui.label("archived").classes("text-xs text-slate-600 border rounded px-1")
                    for part in ("transcript", "state", "files", "bytes"):
                        ui.label(SPACE.said(c[part])).classes("text-xs font-mono")
                    with ui.row().classes("gap-1 no-wrap"):
                        ui.button("files", on_click=lambda _, c=c: self.open_files(c["id"], "", c["name"])
                                  ).props("dense flat size=sm")
                        if not c["archived"]:
                            ui.button("archive", on_click=lambda _, c=c: self.archive_conversation(c)
                                      ).props("dense flat size=sm")
                        ui.button("delete", on_click=lambda _, c=c: self.delete_conversation(c)
                                  ).props("dense flat size=sm color=negative")
                ui.label("shared folder — attached/ in your workspace").classes("text-sm")
                ui.label("").classes("text-xs")
                ui.label("").classes("text-xs")
                ui.label(SPACE.said(u["shared"])).classes("text-xs font-mono")
                ui.label(SPACE.said(u["shared"])).classes("text-xs font-mono")
                ui.button("files", on_click=lambda _: self.open_files("shared", "", "the shared folder")
                          ).props("dense flat size=sm")
                for name, n in (("the rest of your workspace — what the model wrote there", max(0, u["workspace"] - u["shared"])),
                                ("your settings and the rest", u["other"])):
                    ui.label(name).classes("text-sm")
                    for _ in range(3):
                        ui.label("")
                    ui.label(SPACE.said(n)).classes("text-xs font-mono")
                    ui.label("")
            ui.button("measure again", icon="refresh", on_click=self.refresh_space).props("dense flat size=sm")
        self.space_table = space_table
        space_table()
        self.space_said = ui.label("").classes("text-sm text-red-700")

        @ui.refreshable
        def files_panel():
            if self.files_at is None:
                return
            where, path, title = self.files_at
            ui.separator().classes("mt-2")
            with ui.row().classes("items-center gap-2"):
                ui.label("Files — %s%s" % (title, " / " + path if path else "")).classes("font-bold")
                if path:
                    ui.button("up", icon="arrow_upward", on_click=lambda _: self.open_files(
                        where, "/".join(path.split("/")[:-1]), title)).props("dense flat size=sm")
                ui.button("close", on_click=lambda _: self.close_files()).props("dense flat size=sm")
            entries = self.files_listed(where, path)
            if not entries:
                ui.label("nothing here").classes("text-xs opacity-60")
            with ui.grid(columns="auto auto auto").classes("items-center gap-x-5 gap-y-0"):
                for e in entries:
                    if e["folder"]:
                        ui.button(e["name"] + "/", icon="folder", on_click=lambda _, e=e: self.open_files(where, e["path"], title)
                                  ).props("dense flat no-caps size=sm align=left")
                    else:
                        ui.label(e["name"]).classes("text-sm font-mono")
                    ui.label(SPACE.said(e["bytes"])).classes("text-xs font-mono")
                    with ui.row().classes("gap-1 no-wrap"):
                        if not e["folder"]:
                            ui.button("download", on_click=lambda _, e=e: self.download_file(where, e["path"])
                                      ).props("dense flat size=sm")
                        if self.deletable(where, e["path"]):
                            ui.button("delete", on_click=lambda _, e=e: self.delete_file(where, e)
                                      ).props("dense flat size=sm color=negative")
            if where == "shared":
                with ui.row().classes("items-end gap-2 mt-1"):
                    ui.upload(on_upload=self.upload_shared, auto_upload=True, multiple=True, max_file_size=100 * 1024 * 1024
                              ).props('label="upload here" flat dense').classes("w-64")
                    self.new_folder = ui.input("a new folder").props("dense").classes("w-48")
                    ui.button("make", on_click=self.make_folder).props("dense outline size=sm")
            else:
                ui.label("A conversation's files are sent to it in its chat, with the 📎; here they are downloaded and "
                         "deleted.").classes("text-xs opacity-60")
        self.files_panel = files_panel
        files_panel()

    def refresh_space(self, *_):
        SPACE.forget(LOCAL_USERS, self.me)
        self.space_table.refresh()
        self.files_panel.refresh()

    def roots(self, where):
        """The folders the explorer shows of `where`: the shared folder — attached/ in the account's workspace — or a
        conversation's own files, its attached/ and its pictures/, by the names they have in its folder."""
        if where == "shared":
            return os.path.join(LOCAL_USERS.workspace(self.me), SPACE.ATTACHED)
        for c in SPACE.conversations(LOCAL_USERS, self.me):
            if c["id"] == where:
                return os.path.join(LOCAL_USERS.folder(self.me), "user_sessions", c["model"], c["id"])
        raise USERS_MOD.Refused("that conversation is not here any more")

    def files_listed(self, where, path):
        try:
            root = self.roots(where)
            if where == "shared" or path:
                return SPACE.listing(root, path)
            # a conversation's top: its two folders of files, and nothing else of it
            return [e for e in SPACE.listing(root, "") if e["folder"] and e["name"] in (SPACE.ATTACHED, SPACE.PICTURES)]
        except Exception as ex:                             # noqa: BLE001 — said, not raised into the page
            self.space_said.text = str(ex)
            return []

    def deletable(self, where, path):
        """A file or a folder of the shared folder; of a conversation, what is under its attached/ or pictures/."""
        return where == "shared" or "/" in path

    def inside(self, where, path):
        """`path` of `where` as a file's own — refused where it leaves the folder, or a conversation's files."""
        root = self.roots(where)
        if where != "shared" and path.split("/")[0] not in (SPACE.ATTACHED, SPACE.PICTURES):
            raise USERS_MOD.Refused("of a conversation, its files alone are shown here")
        return SPACE.within(root, path)

    def open_files(self, where, path, title):
        self.files_at = (where, path, title)
        self.space_said.text = ""
        self.files_panel.refresh()

    def close_files(self):
        self.files_at = None
        self.files_panel.refresh()

    def download_file(self, where, path):
        try:
            ui.download(self.inside(where, path))
        except Exception as ex:                             # noqa: BLE001
            self.space_said.text = str(ex)

    async def delete_file(self, where, entry):
        if not await self.confirmed_here("Delete %s?" % entry["name"],
                                         "It is removed from the disk%s." % (", with everything in it" if entry["folder"] else ""),
                                         "delete"):
            return
        try:
            import shutil
            full = self.inside(where, entry["path"])
            if os.path.isdir(full) and not os.path.islink(full):
                await asyncio.to_thread(shutil.rmtree, full)
            else:
                os.remove(full)
            self.refresh_space()
        except Exception as ex:                             # noqa: BLE001
            self.space_said.text = str(ex)

    async def upload_shared(self, e):
        """A file into the shared folder, where the explorer is — refused in words where the account's space is full."""
        try:
            where, path, _ = self.files_at
            data = await e.file.read()
            SPACE.room(LOCAL_USERS, self.me, len(data), "the file %s" % e.file.name)
            await asyncio.to_thread(SPACE.keep, SPACE.within(self.roots("shared"), path), e.file.name, data)
            self.safe_notify("%s is in the shared folder: every conversation of yours can read it" % e.file.name, type="positive")
            self.refresh_space()
        except Exception as ex:                             # noqa: BLE001
            self.space_said.text = str(ex)
            self.safe_notify(str(ex), type="negative")

    def make_folder(self, *_):
        try:
            where, path, _ = self.files_at
            name = SPACE.file_name(self.new_folder.value or "")
            os.makedirs(SPACE.within(self.roots("shared"), (path + "/" if path else "") + name), exist_ok=True)
            self.refresh_space()
        except Exception as ex:                             # noqa: BLE001
            self.space_said.text = str(ex)

    async def confirmed_here(self, title, text, button) -> bool:
        with self.page_client, ui.dialog() as dlg, ui.card():
            ui.label(title).classes("font-bold")
            ui.label(text).classes("text-sm opacity-70 max-w-md whitespace-normal")
            with ui.row().classes("justify-end w-full"):
                ui.button("cancel", on_click=lambda: dlg.submit(False)).props("flat")
                ui.button(button, on_click=lambda: dlg.submit(True)).props("color=primary")
        return bool(await dlg)

    async def through_server(self):
        """Whether a conversation is changed through the server: True where this page's server answers — it holds the
        conversations it has loaded, so it is asked; False where no server runs on this data folder, and the folder is
        changed here. Where one runs that this page does not reach, a Refused saying so."""
        try:
            await self.client.health()
            return True
        except Exception:                                   # noqa: BLE001 — no server at this page's endpoint
            if LOCAL_USERS.serving():
                raise USERS_MOD.Refused("a server runs on this data folder and this page is not connected to it: connect "
                                        "to it, or stop it, first")
            return False

    async def delete_conversation(self, c):
        if not await self.confirmed_here("Delete %s?" % c["name"], "The conversation's folder is removed from the disk: its "
                                         "transcript, its state, and the files sent to it and made in it.", "delete"):
            return
        try:
            if await self.through_server():
                await self.client.delete(c["id"])
            else:
                import shutil
                await asyncio.to_thread(shutil.rmtree, self.roots(c["id"]))
            if self.files_at and self.files_at[0] == c["id"]:
                self.files_at = None
            if self.state.get("active_id") == c["id"]:      # the conversation shown is gone: the chat emptied
                self.state["active_id"] = None
                self.state["history"] = []
                self.chat.set_archived(False)
                self.chat.render_chat()
            self.refresh_space()
            await self.conversations.render_convs()
        except Exception as ex:                             # noqa: BLE001
            self.space_said.text = "not deleted: %s" % ex

    async def archive_conversation(self, c):
        if not await self.confirmed_here("Archive %s?" % c["name"], "Its state is given up — %s of the %s it takes — and its "
                                         "transcript and files are kept. It is still shown, and no more is sent to it. "
                                         "This is final: an archived conversation is not opened again."
                                         % (SPACE.said(c["state"]), SPACE.said(c["bytes"])), "archive"):
            return
        try:
            if await self.through_server():
                await self.client.archive(c["id"])
            else:
                import json
                import time
                folder = self.roots(c["id"])
                await asyncio.to_thread(SPACE.archive, os.path.join(folder, "cartridge.npz"))
                side, meta = os.path.join(folder, "server_meta.json"), {}
                if os.path.isfile(side):
                    with open(side) as f:
                        meta = json.load(f)
                meta["archived"] = time.time()
                with open(side, "w") as f:
                    json.dump(meta, f, indent=2)
            if self.state.get("active_id") == c["id"]:
                self.state["active_bound"] = False
                self.chat.set_archived(True)
            self.refresh_space()
            await self.conversations.render_convs()
        except Exception as ex:                             # noqa: BLE001
            self.space_said.text = "not archived: %s" % ex

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
