"""The Options view's models: the ones this release can fetch, downloaded and updated."""
import asyncio
import os
import shutil

from nicegui import ui

from .downloads import DOWNLOADS, human_bytes, save_token, saved_token, token_owner
from .local import MODELS_DIR, list_models
from .part import Part


class ModelsPanel(Part):
    """The models this release can fetch: the list, a download's progress, the Hugging Face token."""

    def build(self):
        """The "Download models" section: the token, the list, a download's bar."""
        ui.label(f"Downloaded from Hugging Face into {MODELS_DIR}. Help → Running SiLVANN says what "
                 f"each needs; one larger than the card runs its experts on the CPU. A pack's name opens "
                 f"its page on Hugging Face, and 'made from' the original model's — its makers' card and "
                 f"the license the pack carries."
                 ).classes("text-xs opacity-70 max-w-2xl whitespace-normal")

        with ui.row().classes("items-end gap-2"):
            self.token_in = ui.input("Hugging Face token", password=True, password_toggle_button=True
                                     ).props("dense").classes("w-80")
            token_save = ui.button("save").props("dense outline")
            token_forget = ui.button("forget").props("dense flat")
        self.token_lbl = ui.label(self.token_state()).classes("text-xs opacity-70 max-w-2xl whitespace-normal")

        token_save.on_click(self.save_hf_token)
        token_forget.on_click(self.forget_hf_token)

        @ui.refreshable
        def models_table():
            if DOWNLOADS.known is None:
                ui.label("reading the list of models…").classes("text-xs opacity-70")
                return
            ui.label(f"{self._free_bytes() / 1e9:.0f} GB free on this disk").classes("text-xs opacity-70")
            with ui.grid(columns="auto auto auto auto").classes("items-center gap-x-6 gap-y-1"):
                for head in ("pack", "made from", "size", ""):
                    ui.label(head).classes("text-xs opacity-70")
                for m in DOWNLOADS.catalogue():
                    ui.link(m["name"], "https://huggingface.co/" + m["repo"], new_tab=True
                            ).classes("font-mono text-sm")
                    if m["base"]:
                        ui.link(m["base"], "https://huggingface.co/" + m["base"], new_tab=True
                                ).classes("text-sm")
                    else:
                        ui.label("")
                    ui.label(m["size"]).classes("text-sm")
                    if m["here"]:
                        self.here_state(m["name"])
                    elif DOWNLOADS.running() and DOWNLOADS.name == m["name"]:
                        ui.label("downloading…").classes("text-sm")
                    else:
                        ui.button("download", on_click=lambda _, n=m["name"]: self.start_download(n)
                                  ).props("dense outline")
        self.models_table = models_table
        self.models_table()

        if DOWNLOADS.known is None or DOWNLOADS.states is None:
            asyncio.create_task(self.read_catalogue())

        with ui.row().classes("items-center gap-3 w-full") as self.dl_row:
            self.dl_bar = ui.linear_progress(value=0, show_value=False).classes("w-80")
            self.dl_label = ui.label("").classes("text-xs font-mono")
            self.dl_stop = ui.button("stop the download", on_click=lambda: asyncio.to_thread(DOWNLOADS.cancel)
                                     ).props("dense flat color=negative")
        self.dl_row.set_visibility(False)
        self.dl_seen = {"running": DOWNLOADS.running(), "states": DOWNLOADS.states is not None}

        ui.timer(1.0, self.pump_download)

        ui.label(f"Models dir: {MODELS_DIR}").classes("text-xs opacity-70 mt-2")

    # the Hugging Face token: faster downloads than a client that is not signed in, and the repositories
    #   whose terms its account accepted
    def token_state(self):
        if os.environ.get("HF_TOKEN"):
            return "Hugging Face token: HF_TOKEN, from the environment the UI was started in"
        token = saved_token()
        if token:
            return f"Hugging Face token: saved ({token[:3]}…{token[-4:]})"
        return ("Hugging Face token: none — downloads go as a client that is not signed in, which Hugging "
                "Face serves more slowly. A token with read access: huggingface.co/settings/tokens")

    async def save_hf_token(self):
        token = (self.token_in.value or "").strip()
        if not token:
            self.safe_notify("paste a token first", type="warning")
            return
        try:
            who = await asyncio.to_thread(token_owner, token)
        except ValueError as e:
            self.safe_notify(f"not saved: {e}", type="negative")
            return
        save_token(token)
        self.token_in.value = ""
        self.token_lbl.set_text(self.token_state() + (f" — signed in as {who}" if who else
                                                      " — not checked: Hugging Face did not answer"))
        self.safe_notify("token saved", type="positive")

    def forget_hf_token(self):
        save_token("")
        self.token_lbl.set_text(self.token_state())
        self.safe_notify("token removed", type="info")

    def _free_bytes(self):
        d = MODELS_DIR
        while not d.exists() and d != d.parent:
            d = d.parent
        return shutil.disk_usage(d).free

    # a model that is here: "update" when its pack's version is above its own, else "done"
    def here_state(self, name):
        st = (DOWNLOADS.states or {}).get(name) or {}
        with ui.row().classes("items-center gap-2"):
            if DOWNLOADS.running() and DOWNLOADS.name == name:
                ui.label("updating…").classes("text-sm")
            elif DOWNLOADS.states is None:
                ui.label("checking…").classes("text-sm opacity-70")
            elif st.get("state") == "update":
                ui.button("update", on_click=lambda _, n=name: self.start_download(n)
                          ).props("dense outline")
            elif st.get("state") == "done":
                ui.label("done").classes("text-sm text-green-700")
            else:
                with ui.label("here").classes("text-sm text-green-700"):
                    ui.tooltip(st.get("why") or "not checked against its pack")
            if st.get("newer"):
                ui.label("its pack has a program for a later release — a newer SiLVANN is out "
                         "(silvann update)").classes("text-xs text-orange-600")

    async def read_states(self):
        if DOWNLOADS.reading:
            return
        DOWNLOADS.reading = True
        try:
            await asyncio.to_thread(DOWNLOADS.read_states)
        finally:
            DOWNLOADS.reading = False
        self.models_table.refresh()

    async def read_catalogue(self):
        try:
            await asyncio.to_thread(DOWNLOADS.catalogue)
        except Exception as e:
            self.safe_notify(f"the list of models could not be read: {e}", type="negative")
        self.models_table.refresh()
        if DOWNLOADS.states is None:
            await self.read_states()

    async def start_download(self, name):
        try:
            await asyncio.to_thread(DOWNLOADS.start, name)
            self.safe_notify(f"{'updating' if DOWNLOADS.updating else 'downloading'} {name}", type="info")
        except Exception as e:
            self.safe_notify(f"not downloaded: {e}", type="negative")
        self.models_table.refresh()

    def refresh_model_select(self):
        opts = list_models()
        keep = (self.options.model_select.value if self.options.model_select.value in opts
                else DOWNLOADS.name if DOWNLOADS.name in opts else (opts[0] if opts else None))
        self.options.model_select.set_options(opts, value=keep)

    def pump_download(self):
        running = DOWNLOADS.running()
        if DOWNLOADS.name and (running or DOWNLOADS.message):
            phase, have, total = DOWNLOADS.progress()
            self.dl_row.set_visibility(True)
            # an update in two runs of the bar: the files read against the pack's, then those fetched
            nothing = not running and DOWNLOADS.message == "done" and not total
            self.dl_bar.set_value(1.0 if nothing else min(1.0, have / total) if total else 0)
            self.dl_label.set_text(f"{DOWNLOADS.name}{' (update)' if DOWNLOADS.updating else ''}: "
                                   + ("nothing to fetch, every file the pack's" if nothing else
                                      f"{'checking' if phase == 'checking' else 'downloading'}… "
                                      f"{human_bytes(have)} of {human_bytes(total)}"
                                      + (" read" if phase == "checking" else ""))
                                   + ("" if running else f" — {DOWNLOADS.message}"))
            self.dl_stop.set_visibility(running)
        if running != self.dl_seen["running"]:
            self.dl_seen["running"] = running
            self.models_table.refresh()
            if not running:
                self.refresh_model_select()
                asyncio.create_task(self.read_states())     # what the folder holds now
        if (DOWNLOADS.states is not None) != self.dl_seen["states"]:
            self.dl_seen["states"] = DOWNLOADS.states is not None
            self.models_table.refresh()
