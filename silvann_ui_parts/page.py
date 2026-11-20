"""A page of SiLVANN: what it is made of, and the order it is built in."""
import asyncio
import ipaddress
from typing import Optional
from urllib.parse import urlsplit

from nicegui import app, ui

import silvann_hosts

from .chat import ChatView
from .client import Client, unattended
from .connection import Connection
from .conversations import Conversations
from .help import HelpView
from .local import LOCAL_APP
from .models_panel import ModelsPanel
from .options import OptionsView
from .remote_sign_in import RemoteSignIn
from .run import _this_address
from .server_process import SERVER
from .session import PAGES, _session_default, _streaming, _with_turn
from .shell import Shell
from .sign_in import LOCAL_USERS, first_account_page, gate, sign_in_page
from .turn import Turn
from .users_view import AccountView, UsersMode, UsersView


def on_this_machine(endpoint: str) -> bool:
    """Whether a server's address is this machine's own: loopback, the machine's address on its network, or one of its
    names."""
    host = (urlsplit(endpoint).hostname or "").lower()
    try:
        return ipaddress.ip_address(host).is_loopback or host == _this_address()
    except ValueError:
        return host == "localhost" or host.endswith(".localhost") or host in silvann_hosts.own_names()


class Page:
    """One browser page of SiLVANN: its state, its client, who it is for, and its parts."""

    def make_parts(self):
        """The parts, made once `setup` has said the page is to be drawn: each takes the page's own names."""
        self.shell = Shell(self)
        self.chat = ChatView(self)
        self.options = OptionsView(self)
        self.models = ModelsPanel(self)
        self.users_mode = UsersMode(self)
        self.users = UsersView(self)
        self.account = AccountView(self)
        self.help = HelpView(self)
        self.connection = Connection(self)
        self.conversations = Conversations(self)
        self.turn = Turn(self)
        self.remote = RemoteSignIn(self)
        for part in (self.shell, self.chat, self.options, self.models, self.users_mode, self.users, self.account, self.help, self.connection, self.conversations, self.turn, self.remote):
            part.meet()

    async def setup(self):
        """Whose page this is and what it talks to — False where SiLVANN is not drawn: the sign-in page instead, or a browser that never connected."""
        self.state = app.storage.user.setdefault("silvann", _session_default())
        for k, v in _session_default().items():
            self.state.setdefault(k, v)
        self.browser_id = app.storage.browser["id"]
        # "streaming" true only while this UI streams the browser's answer: a UI that crashed mid-stream leaves it set in the
        # browser's storage, which would make the send button stop() forever
        self.live = _streaming(self.browser_id)
        self.state["streaming"] = self.live is not None
        if self.live is not None and self.live["cart"] == self.state.get("active_id"):
            self.state["history"] = _with_turn(self.live)                # the turn so far, rather than what the file last kept of it
        self.state["active_bound"] = False
        # ⭐ WHO THIS PAGE IS FOR (NN-196, NN-197, ▶ gate): what it builds follows — the machine's Options and its Users an
        #   admin's alone, never built for anyone else (a hidden panel's handlers would still answer), and a check every few
        #   seconds reloads the page when its answer changes (a sign-in ended, an admin made a user, the machine switched)
        self.mode, self.me, self.admin = gate(self.state)
        self.multi = self.mode == "multi"
        self.built_for = (self.mode, self.me, self.admin)
        if self.mode == "first":
            await first_account_page(self.state)                  # the machine's first account, and nothing else of SiLVANN
            return False
        if self.mode is not None and self.me is None:
            self.state.pop("user_token", None)                    # ended, or never this data folder's
            self.state.pop("user_name", None)
            await sign_in_page(self.state)                        # and nothing else of SiLVANN
            return False
        if self.state.get("view") in ("options", "users") and not self.admin or self.state.get("view") == "account" and self.me is None:
            self.state["view"] = "chat"
        if LOCAL_APP["on"]:
            # the app over this machine's models: its local panel, and the options view until a server is up
            self.state["server_mode"] = "local"
            if not SERVER.is_running() and self.admin:
                self.state["view"] = "options"
        # Per-browser-tab UUID. app.storage.tab requires the WS handshake to be done first —
        # await client.connected() before touching it. Survives reloads within the tab, dies on close.
        import uuid as _uuid
        from nicegui import context as _ng_context
        # ⛳ a page whose websocket never connects (a browser reconnecting after a dropped connection) is left quietly —
        #   the browser loads it again once it is back — rather than raising on the tab storage it cannot have
        try:
            await _ng_context.client.connected(timeout=30)
            tab_store = app.storage.tab
        except (RuntimeError, TimeoutError, asyncio.TimeoutError):
            return False
        if "web_user_id" not in tab_store:
            tab_store["web_user_id"] = _uuid.uuid4().hex
        self.web_user_id = tab_store["web_user_id"]
        # Share the in-state passwords dict by reference so client.set_password() and the UI's
        # storage stay in sync — no separate copy to manage.
        self.client = Client(self.state["endpoint"],
                             passwords=self.state.setdefault("passwords", {}),
                             web_user_id=self.web_user_id)

        self.client.use_token(self.token_for(self.state["endpoint"]))

        # Capture the page-level NiceGUI Client now (while the slot stack is set up) so event-handler
        # tasks — which run with an empty slot stack — can still trigger downloads/notifications by
        # going through this reference instead of through context.client.
        from nicegui import context as _ng_context
        self.page_client = _ng_context.client
        return True

    async def start(self):
        """The page as it opens: its view, its conversation, the server asked, and the timers that keep it current."""
        # Initial render
        self.shell.update_view()
        self.chat.render_chat()
        await self.connection.ping()
        self.live = _streaming(self.browser_id)
        if self.live is not None and self.live["cart"] == self.state["active_id"]:
            # back on another tab while this browser's answer streams: the conversation claimed for this one, so the server
            #   does not put it away as the tab that went's
            await self.conversations.bind_active()
        if not self.remote.asking:
            await self.conversations.render_convs()
        # a streaming answer drawn at most ten times a second, whatever the tokens' rate
        ui.timer(0.1, self.chat.live_tick)

        ui.timer(5.0, self._periodic_ping)

        ui.timer(30.0, self._periodic_heartbeat)
        self._periodic_heartbeat()                                  # a page back: the server told now, not in 30 s
        ui.on("silvann_active", self._active)                       # the person at the page (▶ head.py)

        PAGES.setdefault(self.web_user_id, set()).add(self.page_client.id)

        self.page_client.on_delete(self._page_gone)
        ui.timer(0.5, lambda: self.chat.follow_chat(force=True), once=True)   # the conversation opened at its bottom, once drawn

    def token_for(self, endpoint: str) -> Optional[str]:
        """The token a request to `endpoint` carries: one that server gave this browser (another machine's, whose accounts
        are its own), else — for a server on this machine — this data folder's sign-in (▶ gate), the same token its server
        checks. ⛔ A server elsewhere is never sent this machine's sign-in: whoever holds it is signed in here."""
        theirs = (self.state.get("server_tokens") or {}).get(endpoint.rstrip("/"))
        if theirs:
            return theirs
        return self.state.get("user_token") if self.me is not None and on_this_machine(endpoint) else None

    def safe_notify(self, msg: str, type: str = "info"):
        """ui.notify needs a slot context that event-handler tasks don't always inherit in
        NiceGUI 3.x. Fall back to a log line if the notify can't render."""
        try:
            ui.notify(msg, type=type)
        except RuntimeError:
            print(f"[ui notify suppressed] {msg}", flush=True)

    async def _active(self):
        """⭐ THE PERSON IS AT THE PAGE — a key, the pointer, a touch, said by the browser a few times a minute at most
        (▶ head.py): a use of this machine's sign-in, and of the one a server elsewhere gave this browser. Reading an
        answer or writing a long message is no idleness; a page left alone is, whatever it asks by itself."""
        if self.me is not None:
            await asyncio.to_thread(LOCAL_USERS.signed, self.state.get("user_token"))
        if (self.state.get("server_tokens") or {}).get(self.client.endpoint):
            try:
                await self.client.used()
            except Exception:                                        # noqa: BLE001 — the server silent: its dot says so
                pass

    async def _by_itself(self, asked):
        """`asked()` as the page's own asking, no use of its sign-in (▶ client.py's `unattended`)."""
        with unattended():
            return await asked()

    # Keep the status dot live — re-ping every 5s.
    def _periodic_ping(self):
        asyncio.create_task(self._by_itself(self.connection.ping))
        # a sign-in ended, an admin made a user, the machine switched — looked at, which is no use of the sign-in
        if gate(self.state, use=False) != self.built_for:
            ui.navigate.reload()

    # Heartbeat: tell the server we're still here so the idle-eviction loop doesn't unbind us.
    # Every 30s — well under the 2-min eviction window (gives ~4 missed pings of slack).
    async def _heartbeat(self):
        try:
            with unattended():
                resp = await self.client.ping()
            # If our active cart isn't in the server's owned list, the server evicted us —
            # flip the UI back to banner-mode so the user can Activate again.
            if self.state.get("active_bound") and self.state.get("active_id"):
                if self.state["active_id"] not in (resp.get("owned") or []):
                    self.state["active_bound"] = False
                    self.chat.bind_banner_label.text = "Session evicted by server (idle). Click Activate to reload."
        except Exception:
            pass

    def _periodic_heartbeat(self):
        asyncio.create_task(self._heartbeat())

    async def _page_gone(self):
        """The browser left and did not come back within NiceGUI's reconnect window: the server told, so that with seats
        it puts the conversation away soon after its last answer, rather than two minutes after the last heartbeat."""
        here = PAGES.get(self.web_user_id, set())
        here.discard(self.page_client.id)
        if here:                                           # this tab's page reloaded: the new one is there
            return
        PAGES.pop(self.web_user_id, None)
        try:
            with unattended():
                await self.client.gone()
        except Exception:
            pass


@ui.page("/")
async def page():
    p = Page()
    if not await p.setup():
        return
    p.make_parts()
    p.shell.build()
    p.chat.build()
    p.options.build()
    p.users.build()
    p.account.build()
    p.help.build()
    p.shell.wire()
    p.connection.wire()
    p.chat.wire()
    p.conversations.wire()
    p.turn.wire()
    p.options.wire()
    await p.remote.build()
    await p.start()
