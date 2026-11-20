"""Who a page is for: this data folder's users, the sign-in page, and the page a provider sends the browser back to."""
import asyncio
import secrets
import sys

import httpx
from fastapi.responses import RedirectResponse
from nicegui import app, ui

from .head import SILVANN_TILE
from .local import DATA_DIR, LOCAL_AVAILABLE, RUNTIME_DIR
from .session import _session_default


def _runtime_module(name: str):
    """silvann_runtime.<name> — users.py, sso.py: the standard library's and httpx's alone — without the package's own
    __init__, which boots the engine: a stand-in package whose path is the runtime's folder."""
    import importlib
    import types
    if "silvann_runtime" not in sys.modules:
        pkg = types.ModuleType("silvann_runtime")
        pkg.__path__ = [str(RUNTIME_DIR / "silvann_runtime")]
        sys.modules["silvann_runtime"] = pkg
    return importlib.import_module("silvann_runtime." + name)


# ⭐ THE ACCOUNTS OF THIS DATA FOLDER, READ BY THE UI ITSELF (NN-196): a sign-in checked here whether a model runs or not
#   — the same sign-ins the server checks (users/.sign_ins.json) — the machine's first account made here where it has
#   none (NN-197), and the admin's panels built for an admin alone. None where the UI has no server and runtime beside
#   it (a UI that talks to another machine's server): that server's sign-in, asked for in a dialog.
try:
    USERS_MOD = _runtime_module("users") if LOCAL_AVAILABLE else None
    LOCAL_USERS = USERS_MOD.Users(DATA_DIR) if USERS_MOD is not None else None
except Exception as _ex:                              # noqa: BLE001 — a runtime that does not import: the server's sign-in
    print("[ui] the users of %s not read here (%s): the server's sign-in" % (DATA_DIR, _ex), flush=True)
    USERS_MOD = LOCAL_USERS = None
_SSO: dict = {}


def local_sso():
    """The data folder's provider (sso.json), its sign-ons begun and finished in this process (▶ silvann_runtime/sso.py)."""
    if "sign_on" not in _SSO:
        _SSO["sign_on"] = _runtime_module("sso").SignOn(LOCAL_USERS, DATA_DIR)
    return _SSO["sign_on"]


def gate(state: dict, use: bool = True) -> tuple:
    """⭐ WHO A PAGE IS FOR (NN-196, NN-197 ⚖ 2026-11-19 *"now __local disappears, as it is a normal account with a
    password"*): (the machine's mode, the account signed in or None, an admin). Everyone signs in, on a single user's
    machine as on a multi user's: the browser's sign-in, checked here; the machine's settings, its accounts and its
    models an admin's. The mode is "single" or "multi" (▶ silvann_runtime/users.py), "first" on a machine with no
    account yet — its first page makes one (▶ first_account_page) — and None for a UI with no accounts of its own (no
    runtime beside it): the Options its own, and the server's sign-in the server's. A page opened is a use of its
    sign-in, whose ten minutes count from it; `use` false is a page looking by itself, which keeps none alive."""
    if LOCAL_USERS is None:
        return None, None, True
    if not LOCAL_USERS.ready():
        return "first", None, False
    me = LOCAL_USERS.signed(state.get("user_token"), use=use)
    return LOCAL_USERS.mode(), me, LOCAL_USERS.admin(me)


def _logo():
    ui.html('<span class="silvann-logo" style="font-size: 2rem">'
            + SILVANN_TILE.replace('<svg ', '<svg class="silvann-tile" style="width: 3rem; height: 3rem" ', 1)
            + '<span>LVANN</span></span>')


# what a browser keeps of its own whoever signs in on it: its look, and where it sends conversations
BROWSER_OWN = ("dark", "show_thinking", "convs_expanded", "endpoint", "server_mode", "server_port")


def fresh_session(state: dict):
    """⛔ THE BROWSER'S STATE BACK TO ITS DEFAULTS, ITS OWN CHOICES KEPT (NN-196): at its sign-in page, a sign-in and a
    sign-out — so the next person on the same browser finds nothing of the last one's: their open conversation and its
    messages, the conversations they unlocked with a password, the panel they had open."""
    kept = {k: state[k] for k in BROWSER_OWN if k in state}
    state.clear()
    state.update(_session_default())
    state.update(kept)


async def first_account_page(state: dict):
    """⭐ THE MACHINE'S FIRST ACCOUNT, MADE AT ITS FIRST PAGE (NN-197, ⚖ 2026-11-19 *"maybe we need to create the account
    at install time"*): a machine with no account is given this and nothing else of SiLVANN — where the installer had no
    terminal to ask in (a container, the portable app), or its question went unanswered. The account is the machine's
    admin, and what the machine held before accounts — a single user's conversations, workspace and definitions — is
    its own from now on (▶ users.Users.first). While this page is the one answered, it is answered under this
    machine's address and names alone (▶ silvann_hosts.py)."""
    fresh_session(state)
    ui.dark_mode(bool(state.get("dark", False)))
    kept = (DATA_DIR / "users" / USERS_MOD.BEFORE).is_dir() or (DATA_DIR / "user_sessions").is_dir()
    with ui.column().classes("absolute-center items-center gap-6"):
        _logo()
        with ui.card().classes("w-96 gap-2").mark("first-account"):
            ui.label("This machine's account").classes("text-lg")
            ui.label("SiLVANN asks for a sign-in: here, in the terminal chat and from a script. Make this machine's "
                     "account — its admin, who changes its settings, its models and its accounts."
                     + (" The conversations already on this machine become this account's." if kept else "")
                     ).classes("text-sm opacity-70 whitespace-normal")
            name_in = ui.input("name").classes("w-full")
            pw1 = ui.input("password", password=True, password_toggle_button=True).classes("w-full")
            pw2 = ui.input("password, again", password=True).classes("w-full")
            said = ui.label("").classes("text-sm text-red-700")

            async def make():
                name = (name_in.value or "").strip().lower()
                if not USERS_MOD.NAME.match(name):
                    said.text = "a name is 1 to 32 of a-z, 0-9, _ and -, beginning with a letter or a digit"
                    return
                if not pw1.value or pw1.value != pw2.value:
                    said.text = "the two passwords differ" if pw1.value else "the account needs a password"
                    return
                try:
                    await asyncio.to_thread(LOCAL_USERS.first, name, pw1.value)
                    token = await asyncio.to_thread(LOCAL_USERS.sign_in, name, pw1.value)
                except USERS_MOD.Refused as ex:            # made meanwhile on another page, or its folder is taken
                    said.text = str(ex)
                    return
                fresh_session(state)
                state["user_token"], state["user_name"] = token, name
                ui.navigate.reload()                       # SiLVANN, as the machine's admin

            name_in.on("keydown.enter", lambda: pw1.run_method("focus"))
            pw1.on("keydown.enter", lambda: pw2.run_method("focus"))
            pw2.on("keydown.enter", make)
            ui.button("Make the account", on_click=make).classes("w-full")


async def sign_in_page(state: dict):
    """⭐ THE SIGN-IN, A PAGE OF ITS OWN (NN-196, ⚖ 2026-11-18 *"i would like a proper login screen and not a popup, so
    silvann does not load for not authenticated users"*): a browser not signed in is given this and nothing else of
    SiLVANN — no drawer, no conversations, no documents — and SiLVANN itself once it has signed in. On a single user's
    machine another account has, a right password is refused, saying who has it (▶ users.InUse); after a wrong one
    the name waits a minute at this browser's address, and the system's log has both (▶ users.Wait). The provider's
    button where the data folder names one (sso.json, ▶ silvann_runtime/sso.py)."""
    fresh_session(state)
    ui.dark_mode(bool(state.get("dark", False)))
    at = ui.context.client.ip or None                  # the browser's address, as this UI sees it
    with ui.column().classes("absolute-center items-center gap-6"):
        _logo()
        with ui.card().classes("w-80 gap-2").mark("sign-in"):
            ui.label("Sign in").classes("text-lg")
            ui.label("This machine is single user: one account at a time signs in, and has it until it signs out, "
                     "ten minutes pass with nothing asked of it, or SiLVANN is opened again."
                     if LOCAL_USERS.mode() == "single" else
                     "This machine has users: each signs in, and sees their own conversations. A sign-in is over "
                     "ten minutes after its last use.").classes("text-sm opacity-70 whitespace-normal")
            name_in = ui.input("name").classes("w-full")
            password_in = ui.input("password", password=True, password_toggle_button=True).classes("w-full")
            said = ui.label("").classes("text-sm text-red-700 whitespace-normal")

            async def sign_in():
                name = (name_in.value or "").strip().lower()
                try:
                    token = await asyncio.to_thread(LOCAL_USERS.sign_in, name, password_in.value or "", at)
                except USERS_MOD.Refused as ex:            # theirs, and the machine another account's now (InUse); or
                    said.text = str(ex)                    #   a wrong one given less than a minute ago (Wait)
                    password_in.value = ""
                    return
                if token is None:
                    said.text = "wrong name or password: the next try in %d seconds" % USERS_MOD.WAIT
                    password_in.value = ""
                    return
                fresh_session(state)
                state["user_token"], state["user_name"] = token, name
                ui.navigate.reload()                       # SiLVANN, as the signed-in user

            name_in.on("keydown.enter", lambda: password_in.run_method("focus"))
            password_in.on("keydown.enter", sign_in)
            ui.button("Sign in", on_click=sign_in).classes("w-full")
            try:
                provider = await asyncio.to_thread(local_sso().public)
            except Exception:                              # noqa: BLE001 — no provider to offer
                provider = {}
            if provider.get("on"):
                async def sign_on():                       # its page, then back to /_silvann/sso, checked there
                    try:
                        got = await asyncio.to_thread(local_sso().start)
                    except Exception as ex:                # noqa: BLE001 — the provider's refusal or silence, said
                        said.text = str(ex)
                        return
                    state["sso_state"] = got["state"]      # the page coming back checks the browser is this one
                    ui.navigate.to(got["url"])
                ui.separator()
                ui.button("Sign in with %s" % provider.get("label", "your organisation"), on_click=sign_on
                          ).props("outline").classes("w-full").mark("sign-on")


def _said(response) -> str:
    """What a server's refusal says: its `detail`, else its first words."""
    try:
        return str(response.json().get("detail") or response.text[:200])
    except ValueError:
        return response.text[:200]


@ui.page("/_silvann/sso")
async def sso_back(code: str = "", state: str = "", error: str = "", error_description: str = ""):
    """⭐ THE PROVIDER SENDS THE BROWSER BACK HERE (NN-157; sso.json's `redirect_uri` is this page's address): the state
    checked against the one this browser kept when it left — a browser sent here by someone else's sign-on is refused —
    then the code handed to the server, which checks the account itself and answers a token, as a password's sign-in
    does; the chat after."""
    session = app.storage.user.setdefault("silvann", _session_default())
    expected = session.pop("sso_state", None)
    said = ""
    if error:
        said = "the provider: %s" % (error_description or error)
    elif not expected or not secrets.compare_digest(state.encode(), str(expected).encode()):
        said = "this sign-on was not begun in this browser, or it was finished already: begin it again"
    elif LOCAL_USERS is not None and LOCAL_USERS.ready() \
            and LOCAL_USERS.signed(session.get("user_token"), use=False) is None:
        # this data folder's sign-in (NN-196): the provider's answer checked here, a token every process here takes
        try:
            name = await asyncio.to_thread(local_sso().finish, code, state)
            fresh_session(session)
            session["user_token"], session["user_name"] = await asyncio.to_thread(LOCAL_USERS.issue, name), name
            return RedirectResponse("/")
        except Exception as ex:                           # noqa: BLE001 — refused, or the provider silent: said
            said = str(ex)
    else:
        try:
            async with httpx.AsyncClient(timeout=30) as http:
                r = await http.post(f"{session['endpoint'].rstrip('/')}/users/sso/finish",
                                    json={"code": code, "state": state})
            if r.status_code == 200:
                got = r.json()
                session.setdefault("server_tokens", {})[session["endpoint"].rstrip("/")] = got["token"]
                session["server_user"] = got["name"]
                return RedirectResponse("/")
            said = _said(r)
        except httpx.HTTPError as ex:
            said = "the server does not answer: %s" % ex
    with ui.card().classes("absolute-center w-96 gap-2"):
        ui.label("Sign in").classes("text-lg")
        ui.label(said).classes("text-sm text-red-700").mark("sign-on-said")
        ui.link("back to SiLVANN", "/")
