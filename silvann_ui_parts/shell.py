"""The page's frame: the drawer, its rows, the view shown, the theme."""
import asyncio

from nicegui import ui

from .head import SILVANN_TILE
from .help import help_doc
from .part import Part


class Shell(Part):
    """The drawer and what it switches: the view shown, the theme."""

    def build(self):
        """The drawer: the mark, the menu's rows, the conversations' column."""
        # Layout: no top header. Left drawer defaults to mini (icons only); hovering expands it to
        # 280px as an overlay so it doesn't push the main area around. Main area is a flex column
        # holding a thin hamburger bar and three view columns (only one visible at a time).

        # ============================================================
        # LEFT DRAWER (always-on, mini by default, hover-expands to full width)
        # ============================================================
        self.drawer = ui.left_drawer(value=True
                                     ).props("bordered mini mini-to-overlay mini-width=60 width=280 "
                                             "behavior=desktop"
                                             ).classes("flex flex-col")
        # Hover-expand with a 333ms collapse debounce. Brief mouse-outs (e.g. drifting onto a
        # popup q-menu from the 3-dot button) won't flash the drawer shut.
        self._collapse = {"task": None}

        self.drawer.on("mouseenter", self._on_drawer_enter)
        self.drawer.on("mouseleave", self._on_drawer_leave)

        with self.drawer:
            # --- The brand row: silicon's square, then "LVANN" (hidden with the drawer closed, .silvann-brand-label) ---
            with ui.row().classes("silvann-brand-row items-center w-full px-3 py-3 gap-2"):
                ui.html('<span class="silvann-logo">' + SILVANN_TILE.replace('<svg ', '<svg class="silvann-tile" ', 1)
                        + '<span class="silvann-brand-label">LVANN</span></span>').classes("flex-grow")
                self.status_dot = ui.icon("circle").classes("text-gray-400").style("font-size: 12px")
                with self.status_dot:
                    self.status_tip = ui.tooltip("connecting…")
            ui.separator()

            # --- Nav rows (Dark mode / Help / Options / Chat): Chat last, so its conversations run down the drawer ---
            with ui.column().classes("w-full p-1 gap-0"):
                self.theme_row = ui.row().classes("silvann-menu-item w-full")
                with self.theme_row:
                    self.theme_icon = ui.icon("dark_mode")
                    self.theme_label = ui.label("Dark mode")

                self.help_row = ui.row().classes("silvann-menu-item w-full")
                with self.help_row:
                    ui.icon("help_outline")
                    ui.label("Help")

                self.opts_row = self.menu_row("settings", "Options") if self.admin else None
                # the accounts, an admin's; the account, each signed-in user's own
                self.users_row = self.menu_row("group", "Users") if self.me is not None and self.admin else None
                self.account_row = self.menu_row("account_circle", "Account") if self.me is not None else None

                self.chat_row = ui.row().classes("silvann-menu-item w-full")
                with self.chat_row:
                    ui.icon("chat")
                    ui.label("Chat").classes("flex-grow")
                    self.chat_chevron = ui.icon("expand_more")

                # Conversations sub-tree — populated by render_convs() once the page connects via WS.
                # Show a placeholder synchronously so the drawer isn't empty during the initial paint.
                self.convs_col = ui.column().classes("pl-3 pr-1 w-full gap-0 silvann-convs-col")
                with self.convs_col:
                    ui.label("(loading…)").classes("opacity-50 text-xs px-2 py-1")

            ui.space()
            ui.separator()

    def wire(self):
        """What the menu's rows do, and the theme."""
        self.chat_row.on("click", lambda: asyncio.create_task(self.on_chat_click()))
        for row, name in ((self.opts_row, "options"), (self.users_row, "users"), (self.account_row, "account")):
            if row is not None:
                row.on("click", lambda _=None, name=name: asyncio.create_task(self.select_other_view(name)))
        self.help_row.on("click", lambda: asyncio.create_task(self.select_other_view("help")))

        # light or dark, as this browser last chose — the row offers the other
        self.dark = ui.dark_mode(bool(self.state.get("dark", False)))

        self.show_theme()
        self.theme_row.on("click", self.on_theme_click)

    async def _collapse_after(self, delay: float):
        try:
            await asyncio.sleep(delay)
            self.drawer.props(add="mini")
        except asyncio.CancelledError:
            pass

    def _cancel_pending_collapse(self):
        t = self._collapse["task"]
        if t is not None and not t.done():
            t.cancel()
        self._collapse["task"] = None

    def _on_drawer_enter(self):
        self._cancel_pending_collapse()
        self.drawer.props(remove="mini")

    def _on_drawer_leave(self):
        self._cancel_pending_collapse()
        self._collapse["task"] = asyncio.create_task(self._collapse_after(0.1))

    # ⛔ A ROW BUILT ONLY FOR WHOEVER MAY OPEN IT (NN-196) — never one built and hidden: the active row's
    #   highlight replaces a row's classes (▶ update_view), NiceGUI's `hidden` among them
    def menu_row(self, icon, label):
        row = ui.row().classes("silvann-menu-item w-full")
        with row:
            ui.icon(icon)
            ui.label(label)
        return row

    # ============================================================
    # Rendering / view-switching helpers (close over everything above)
    # ============================================================
    def update_view(self):
        v = self.state["view"]
        expanded = bool(self.state.get("convs_expanded", True))
        self.chat.chat_view.set_visibility(v == "chat")
        self.options.opts_view.set_visibility(v == "options")
        self.users.users_view.set_visibility(v == "users")
        self.account.account_view.set_visibility(v == "account")
        self.help.help_view.set_visibility(v == "help")
        # Sub-tree is shown only when Chat is the active view AND the user hasn't collapsed it.
        self.convs_col.set_visibility(v == "chat" and expanded)
        self.chat_chevron.props(
            f"name={'expand_less' if (v == 'chat' and expanded) else 'expand_more'}")
        # Highlight active menu row.
        base = "silvann-menu-item w-full"
        for row, name in ((self.chat_row, "chat"), (self.opts_row, "options"), (self.users_row, "users"), (self.account_row, "account"),
                          (self.help_row, "help")):
            if row is not None:                            # a row this page was not built with (▶ menu_row)
                row.classes(replace=base + (" silvann-menu-active" if v == name else ""))

    async def on_chat_click(self):
        # Clicking Chat: if it's already the active view, toggle the sub-tree. Otherwise switch
        # to chat AND ensure the sub-tree is expanded (auto-show on view switch).
        if self.state["view"] == "chat":
            self.state["convs_expanded"] = not bool(self.state.get("convs_expanded", True))
            self.update_view()
        else:
            self.state["view"] = "chat"
            self.state["convs_expanded"] = True
            self.update_view()
            await self.conversations.render_convs()

    async def select_other_view(self, v: str):
        if v in ("options", "users") and not self.admin or v == "account" and self.me is None:
            return                                         # not built for this page (▶ gate)
        self.state["view"] = v
        self.update_view()
        if v == "options":
            await self.options.show_tools()

    def show_theme(self):
        on = bool(self.state.get("dark", False))
        self.theme_icon.name = "light_mode" if on else "dark_mode"
        self.theme_label.text = "Light mode" if on else "Dark mode"

    def on_theme_click(self):
        self.state["dark"] = not bool(self.state.get("dark", False))
        self.dark.set_value(self.state["dark"])
        self.show_theme()
        self.help.help_doc_md.set_content(help_doc(self.help.help_shown["path"], self.state["dark"]))      # its pictures in the new theme
