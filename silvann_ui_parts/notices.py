"""What this machine's settings cost a model on its card, said to its admin when a page opens."""
import asyncio

from nicegui import ui

from .help import USERS_GUIDE_DOC
from .local import LOCAL_AVAILABLE
from .part import Part
from .sign_in import _runtime_module


class Notices(Part):
    """Windows' settings that cost a model on its card — the NVIDIA driver's shared memory, a page file smaller than
    the card (▶ silvann_runtime/system.py's `notices`) — each a dialog to this machine's admin. A notice closed with
    "don't show again" stays closed in this browser for as long as the setting stays as it is: put right and wrong
    again, it is said again."""

    async def show(self):
        """The notices this machine has now, those this browser was not told to keep closed."""
        if not (self.admin and LOCAL_AVAILABLE):
            return                                                  # a server elsewhere: its own machine's to say
        try:
            found = await asyncio.to_thread(_runtime_module("system").notices)
        except Exception as ex:                                     # noqa: BLE001 — a reading that fails says nothing
            print("[ui] this machine's settings not read (%s)" % ex, flush=True)
            return
        closed = [i for i in self.state.get("notices_closed", []) if i in [n["id"] for n in found]]
        self.state["notices_closed"] = closed
        for notice in found:
            if notice["id"] not in closed:
                self.say(notice)

    def say(self, notice: dict):
        """One notice's dialog: what is amiss and how to put it right, the page that shows it, and its two ways out."""
        def keep_closed():
            self.state["notices_closed"] = self.state.get("notices_closed", []) + [notice["id"]]
            dlg.close()

        def guide():
            dlg.close()
            self.help.open_doc(USERS_GUIDE_DOC)

        with self.page_client, ui.dialog() as dlg, ui.card().classes("gap-3 max-w-xl").mark("notice-" + notice["id"]):
            ui.label(notice["title"]).classes("text-lg font-bold")
            ui.label(notice["text"]).classes("text-sm whitespace-pre-wrap")
            with ui.row().classes("w-full justify-end gap-2"):
                if notice.get("link"):
                    ui.button("NVIDIA's page on it", icon="open_in_new",
                              on_click=lambda: ui.navigate.to(notice["link"], new_tab=True)).props("flat")
                ui.button("the guide", icon="help_outline", on_click=guide).props("flat").tooltip(
                    "Running SiLVANN, on Windows")
                ui.button("don't show again", on_click=keep_closed).props("flat")
                ui.button("close", on_click=dlg.close).props("color=primary")
        dlg.open()
