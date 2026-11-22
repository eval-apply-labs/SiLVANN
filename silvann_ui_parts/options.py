"""The Options view: this machine's model and config, its server started and stopped, the server the UI talks to, the server's tools."""
import asyncio
import json
from typing import Optional

from nicegui import ui

from .local import LOCAL_APP, LOCAL_AVAILABLE, MODELS_DIR, ensure_config, list_model_configs, list_models, model_configs_dir, model_default_config, new_config
from .client import unattended
from .part import Part
from .server_process import SERVER


class OptionsView(Part):
    """The Options view: the model and its config, the server's start and stop, its log."""

    def build(self):
        """The Options view's widgets, an admin's alone."""
        # --- Options view (was "Server" tab) ---
        self.opts_view = ui.column().classes("silvann-main w-full p-4 gap-2 overflow-auto")
        with self.opts_view:
            if self.admin:
                ui.label("Server control").classes("text-lg font-bold")

            # this machine's models and its server; which server the UI talks to is chosen at the bottom
            with ui.column().classes("w-full gap-2"):
                if not self.admin:
                    pass                                       # never built for a user who is not an admin (▶ gate)
                elif not LOCAL_AVAILABLE:
                    ui.label(
                        f"Local mode unavailable: this UI process can't find start_silvann_server.py "
                        f"beside it. Run the UI from main/ (the runtime root) to enable local control."
                    ).classes("text-orange-600 max-w-2xl")
                else:
                    # MODELS — the ones this release can fetch, which are here, and the one downloading
                    # collapsed once a model is here: downloading is the first visit's work, starting a server every visit's
                    with ui.expansion("Download models", icon="download", value=not list_models()).classes("w-full"):
                        self.models.build()

                    # MODEL selector — the self-contained folders under main/models/ (drives the config path).
                    model_options = list_models()
                    _init_model = (self.state["selected_model"] if self.state["selected_model"] in model_options
                                   else (model_options[0] if model_options else None))
                    self.state["selected_model"] = _init_model or ""
                    self.model_select = ui.select(
                        model_options, value=_init_model, label="Model",
                    ).classes("w-80")

                    # CONFIG selector — scoped to the selected model's configs/ folder.
                    ensure_config(_init_model)
                    _init_cfgs = list_model_configs(_init_model) if _init_model else []
                    _init_cfg = (self.state["selected_config"] if self.state["selected_config"] in _init_cfgs
                                 else ((_init_model and model_default_config(_init_model))
                                       or (_init_cfgs[0] if _init_cfgs else None)))
                    self.state["selected_config"] = _init_cfg or ""
                    # ⭐ A CONFIG IS AN OVERRIDE: the boot chooses every setting a config leaves out for this machine, so the
                    #   downloaded default.json — every setting "auto" — is the one to run; this section, collapsed, is
                    #   for naming one
                    self.cfg_box = ui.expansion("Config: %s" % (_init_cfg or "none"), icon="tune", value=False).classes("w-full")
                    with self.cfg_box:
                        ui.label("Every setting a config leaves out is chosen for this machine at each boot: a config names "
                                 "only what it overrides.").classes("text-xs opacity-70 max-w-2xl whitespace-normal")
                        self.config_select = ui.select(
                            _init_cfgs, value=_init_cfg, label="Config",
                        ).classes("w-80")
                        self.config_desc_label = ui.label(self._config_desc(_init_model, _init_cfg)).classes(
                            "text-xs opacity-70 max-w-2xl whitespace-normal")
                        with ui.row().classes("gap-2 items-end"):
                            ui.button("edit selected",
                                      on_click=lambda: self.open_config_editor(self.config_select.value)
                                      ).props("dense outline")
                            ui.button("new config",
                                      on_click=lambda: self.open_config_editor(None)
                                      ).props("dense outline")
                            ui.button("autotune", icon="tune",
                                      on_click=lambda: self.forget_autotune()
                                      ).props("dense outline").tooltip("Test the setups again at the next start")
                            ui.button("delete selected",
                                      on_click=lambda: self.delete_config(self.config_select.value)
                                      ).props("dense outline color=negative")
                            ui.button("config guide", icon="help_outline",
                                      on_click=lambda: self.help.open_config_guide()
                                      ).props("dense flat").tooltip("What every key and every model's option does")

                    self.config_select.on_value_change(self._on_config_change)

                    self.model_select.on_value_change(self._on_model_change)
                    with ui.row().classes("gap-2 items-end"):
                        self.port_in = ui.number(label="Port", value=self.state["server_port"],
                                                 min=1, max=65535).classes("w-32")
                    self.status_lbl = ui.label("status: ...").classes("font-mono text-sm")
                    # what the server chose for this machine, or why it did not start — said here, not only in the log
                    self.setup_lbl = ui.label("").classes("text-sm whitespace-pre-wrap")
                    with ui.row().classes("gap-2"):
                        self.start_btn = ui.button("start", icon="play_arrow"
                                                   ).props("color=positive")
                        self.stop_btn = ui.button("stop", icon="stop"
                                                  ).props("color=negative")
                    ui.label("Subprocess log").classes("font-bold mt-2 text-sm")
                    # Smaller + softer than a full-height terminal — the page no longer feels half-black.
                    self.log_pane = ui.log(max_lines=500).classes(
                        "w-full h-48 bg-slate-100 text-slate-800 font-mono text-xs p-2 border rounded"
                    )

            if self.admin:                                          # the rest of the Options, an admin's too
                # ⭐ THE SERVER THE UI TALKS TO: this machine's, at the port above, or another's at an address — a server
                #   elsewhere is started and stopped on its own machine
                ui.separator().classes("mt-4")
                with ui.row().classes("items-center gap-4"):
                    ui.label("Send conversations to:").classes("font-bold")
                    self.server_sel = ui.toggle({"local": "this machine", "remote": "remote"}, value=self.state["server_mode"])
                with ui.row().classes("items-end gap-2") as remote_row:
                    self.endpoint_in = ui.input(label="Address", value=self.state["endpoint"]).props("dense outlined").classes("w-96")
                    self.connect_btn = ui.button("connect").props("dense outline color=primary")
                remote_row.bind_visibility_from(self.server_sel, "value", lambda v: v == "remote")
                if not LOCAL_AVAILABLE:                           # no server beside this UI to start: another's only
                    self.server_sel.set_value("remote")
                    self.server_sel.disable()
                self.server_sel.on_value_change(lambda e: asyncio.create_task(self.connection.on_server_mode(e.value)))

                # ⭐ THE SERVER'S TOOLS (NN-127): what its model's REPL reaches on its machine, as the machine's tools file sets
                #   it up — the file is edited there (docs/tools.md); "Rebuild the tools" reads the skills again, gives each one
                #   installed since its entry, and checks again what each tool needs
                ui.separator().classes("mt-4")
                with ui.row().classes("items-center gap-2"):
                    ui.label("The server's tools").classes("font-bold")
                    ui.button("Rebuild the tools", icon="build", on_click=lambda: asyncio.create_task(self.on_rebuild_tools())
                              ).props("dense outline").tooltip(
                        "Read the skills again, give each one installed since its entry in the tools file, and check again "
                        "what each tool needs — a SearXNG to search with, a picture model to draw with")
                    ui.button(icon="refresh", on_click=lambda: asyncio.create_task(self.show_tools())
                              ).props("dense flat").tooltip("Ask the server again")
                self.tools_lbl = ui.label("").classes("text-xs font-mono whitespace-pre-wrap")

                self.users_mode.build()
                self.updates.build()                              # the version this copy runs, and the one after it

    def wire(self):
        """The server's status, start, stop and log."""
        # ----- Options view: server-control wiring (only if LOCAL_AVAILABLE) -----
        if LOCAL_AVAILABLE and self.admin:

            self.start_btn.on("click", self.do_start)
            self.stop_btn.on("click", self.do_stop)

            self.log_pane.clear()
            self.state["log_cursor"] = 0
            self.pump_log()
            ui.timer(0.5, self.pump_log)
            ui.timer(1.0, self.refresh_status)
            asyncio.create_task(self.refresh_status())

    def _config_desc(self, model, name):
        # Surface the config's "_comment" so users see what each setup does (VRAM,
        # topology, offload) without opening the JSON.
        import json as _json
        if not (model and name):
            return ""
        try:
            return _json.loads((model_configs_dir(model) / name).read_text(encoding="utf-8")).get("_comment", "")
        except Exception:
            return ""

    def _on_config_change(self, e):
        self.state["selected_config"] = e.value
        self.config_desc_label.set_text(self._config_desc(self.state["selected_model"], e.value))
        self.cfg_box.set_text("Config: %s" % (e.value or "none"))

    def _on_model_change(self, e):
        # Selecting a model RESETS the config selector to that model's configs/ folder.
        model = e.value
        self.state["selected_model"] = model
        made, said = ensure_config(model)
        if said:
            self.safe_notify(said, type="positive" if made else "warning")
        cfgs = list_model_configs(model)
        new_cfg = model_default_config(model) or (cfgs[0] if cfgs else None)
        self.config_select.set_options(cfgs, value=new_cfg)
        self.state["selected_config"] = new_cfg or ""
        self.config_desc_label.set_text(self._config_desc(model, new_cfg))
        self.cfg_box.set_text("Config: %s" % (new_cfg or "none"))

    async def show_tools(self, got=None, said=()):
        try:
            got = got or await self.client.tools_settings()
        except Exception as ex:                       # noqa: BLE001 — said in the section
            self.tools_lbl.set_text("the server does not answer (%s)" % type(ex).__name__)
            return
        if not got.get("tools_file"):
            self.tools_lbl.set_text("the server's model has no tools")
            return
        lines = list(said) + (["—"] if said else []) + ["the tools file: %s" % got["tools_file"]]
        for name, e in sorted(got.get("entries", {}).items()):
            now = ("switched off" if e["enabled"] == "no" else "not available — %s" % e["unavailable"]
                   if e.get("unavailable") else "on" if e["enabled"] == "yes" else "on (auto)")
            settings = ", ".join("%s %s" % (k, json.dumps(v)) for k, v in e.get("config", {}).items())
            lines.append("%-10s %s%s" % (name, now, "\n           %s" % settings if settings else ""))
        self.tools_lbl.set_text("\n".join(lines))

    async def on_rebuild_tools(self):
        try:
            got = await self.client.rebuild_tools()
        except Exception as ex:                       # noqa: BLE001 — said to the user
            self.safe_notify("the tools were not rebuilt: %s" % ex, type="negative")
            return
        await self.show_tools(got, got.get("said") or ["nothing changed"])
        self.safe_notify("the tools rebuilt", type="positive")

    async def refresh_status(self):
        # Probe the configured endpoint so externally-started servers (e.g. ones spawned by a
        # previous UI instance and outliving it) get surfaced too, not just subprocesses we own.
        external = None
        try:
            with unattended():
                h = await self.client.health()
            external = f"alive at {self.client.endpoint} (model={h.get('model_id', '?')})"
        except Exception:
            external = None
        if SERVER.failure and not SERVER.is_running():
            if self.state.get("failure_shown") != SERVER.failure:
                self.state["failure_shown"] = SERVER.failure
                self.safe_notify(("the server ended: " if SERVER.started else "not started: ") + SERVER.failure,
                                 type="negative")
            self.setup_lbl.text = ("⛔ the server ended: " if SERVER.started else "⛔ not started: ") + SERVER.failure
            self.setup_lbl.classes(replace="text-sm whitespace-pre-wrap text-red-700")
        elif SERVER.setup:
            self.setup_lbl.text = SERVER.setup
            self.setup_lbl.classes(replace="text-sm whitespace-pre-wrap text-slate-700")
        else:
            self.setup_lbl.text = ""
        if SERVER.is_running():
            self.status_lbl.text = f"status: {SERVER.status_text()}"
        elif SERVER.failure:
            self.status_lbl.text = "status: ended" if SERVER.started else "status: not started"
        elif external:
            self.status_lbl.text = (f"status: external server {external} — not under UI control "
                                    f"(kill it from the box that started it)")
        else:
            self.status_lbl.text = "status: stopped"
        self.start_btn.set_enabled(not SERVER.is_running())
        self.stop_btn.set_enabled(SERVER.is_running())

    async def do_start(self):
        if not self.model_select.value:
            self.safe_notify("Pick a model first", type="warning")
            return
        if not self.config_select.value:
            made, said = ensure_config(self.model_select.value)
            if not made:
                self.safe_notify(said or "Pick a config first", type="warning")
                return
            self.config_select.set_options(list_model_configs(self.model_select.value), value=made)
            self.state["selected_config"] = made
            self.safe_notify(said, type="positive")
        try:
            model_dir = str(MODELS_DIR / self.model_select.value)
            # a --local app starts it again when it is next started, once it has booted
            SERVER.start(model_dir, self.config_select.value, int(self.port_in.value),
                         remember={"model": self.model_select.value, "config": self.config_select.value,
                                   "port": int(self.port_in.value)} if LOCAL_APP else None)
            self.state["server_port"] = int(self.port_in.value)
            self.state["selected_model"] = self.model_select.value
            self.state["selected_config"] = self.config_select.value
            new_ep = f"http://localhost:{int(self.port_in.value)}"
            self.state["endpoint"] = new_ep
            self.endpoint_in.value = new_ep
            self.server_sel.set_value("local")
            self.client.set_endpoint(new_ep)
            self.safe_notify(f"started: {self.model_select.value} / {self.config_select.value} "
                             f"on port {int(self.port_in.value)}", type="positive")
        except Exception as e:
            self.safe_notify(f"start failed: {e}", type="negative")
        await self.refresh_status()

    async def do_stop(self):
        try:
            # stop() now saves conversations first (blocking, can be ~min for big windows) —
            # run it off the UI event loop so the page doesn't freeze during the save.
            await asyncio.to_thread(SERVER.stop)
            self.safe_notify("stopped", type="info")
        except Exception as e:
            self.safe_notify(f"stop failed: {e}", type="negative")
        await self.refresh_status()

    def pump_log(self):
        snap = SERVER.snapshot()
        cur = self.state["log_cursor"]
        if SERVER.rewrites != self.state.get("log_rewrites"):   # a line replaced in place: the pane again, whole
            self.state["log_rewrites"] = SERVER.rewrites
            cur = len(snap) + 1
        if cur > len(snap):
            self.log_pane.clear()
            cur = 0
        for line in snap[cur:]:
            self.log_pane.push(line)
        self.state["log_cursor"] = len(snap)

    def open_config_editor(self, filename: Optional[str]):
        """Edit the selected model's configs/<filename>.json (filename=None means New)."""
        model = self.state.get("selected_model") or ""
        if not model:
            self.safe_notify("Pick a model first", type="warning")
            return
        cfg_dir = model_configs_dir(model)
        is_new = not (filename and (cfg_dir / filename).exists())
        if not is_new:
            body = (cfg_dir / filename).read_text(encoding="utf-8")
        else:
            body = json.dumps(new_config(model), indent=2)
        with self.page_client, ui.dialog().props("maximized") as dlg, ui.card().classes("w-full h-full"):
            ui.label(("New config" if is_new else f"Edit {filename}")
                     ).classes("font-bold")
            name_in = ui.input(label="filename",
                               value=(filename or "new_config.json")
                               ).classes("w-80")
            body_in = ui.textarea(value=body
                                  ).classes("w-full font-mono text-xs flex-grow"
                                            ).props("outlined autogrow rows=30")
            err = ui.label("").classes("text-red-500 text-sm")

            async def save():
                try:
                    parsed = json.loads(body_in.value)
                except json.JSONDecodeError as e:
                    err.text = f"JSON parse error: {e}"
                    return
                target = cfg_dir / name_in.value
                try:
                    cfg_dir.mkdir(parents=True, exist_ok=True)      # a pack that came with no configs/ folder
                    target.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
                except OSError as e:
                    err.text = f"could not write {target}: {e}"
                    return
                self.safe_notify(f"saved {target.name}", type="positive")
                dlg.close()
                ui.navigate.reload()

            with ui.row().classes("justify-end w-full mt-2 gap-2"):
                ui.button("config guide", icon="help_outline",
                          on_click=lambda: (dlg.close(), self.help.open_config_guide())).props("flat")
                ui.button("cancel", on_click=dlg.close).props("flat")
                ui.button("save", on_click=save).props("color=primary")
        dlg.open()

    def forget_autotune(self):
        """The setup the boot chose for the selected model on this machine forgotten (its `autotune.json`): the next start
        tests the setups again, and what the config leaves on auto is chosen afresh. The configs are not touched."""
        model = self.state.get("selected_model") or ""
        if not model:
            self.safe_notify("Pick a model first", type="warning")
            return
        target = MODELS_DIR / model / "autotune.json"
        if not target.exists():
            self.safe_notify("Nothing to forget: the next start tests the setups anyway", type="info")
            return
        with self.page_client, ui.dialog() as dlg, ui.card().classes("gap-2"):
            ui.label("Autotune again?").classes("font-bold")
            ui.label("Forgets the setup chosen for this machine — where the experts sit, the tier on the card, the "
                     "context. The next start tests the setups again (a minute or so) and chooses afresh. The configs "
                     "are not changed." + (" A server running now keeps its setup until it is started again."
                                           if SERVER.is_running() else "")).classes("text-xs opacity-70 max-w-md")

            def do_forget():
                try:
                    target.unlink()
                except Exception as e:
                    self.safe_notify(f"could not forget it: {e}", type="negative")
                    dlg.close()
                    return
                self.safe_notify("forgotten: the next start tests the setups again", type="positive")
                dlg.close()

            with ui.row().classes("justify-end w-full gap-2"):
                ui.button("cancel", on_click=dlg.close).props("flat")
                ui.button("autotune again", on_click=do_forget).props("color=primary")
        dlg.open()

    def delete_config(self, filename: Optional[str]):
        """Delete the selected model's configs/<filename>.json (with a confirm step)."""
        model = self.state.get("selected_model") or ""
        if not model:
            self.safe_notify("Pick a model first", type="warning")
            return
        if not filename:
            self.safe_notify("No config selected", type="warning")
            return
        target = model_configs_dir(model) / filename
        if not target.exists():
            self.safe_notify(f"{filename} not found", type="warning")
            return
        with self.page_client, ui.dialog() as dlg, ui.card().classes("gap-2"):
            ui.label(f"Delete {filename}?").classes("font-bold")
            ui.label("Permanently removes this config file. The model's weights are "
                     "not affected.").classes("text-xs opacity-70 max-w-md")

            def do_delete():
                try:
                    target.unlink()
                except Exception as e:
                    self.safe_notify(f"delete failed: {e}", type="negative")
                    dlg.close()
                    return
                self.safe_notify(f"deleted {filename}", type="positive")
                dlg.close()
                ui.navigate.reload()

            with ui.row().classes("justify-end w-full gap-2"):
                ui.button("cancel", on_click=dlg.close).props("flat")
                ui.button("delete", on_click=do_delete).props("color=negative")
        dlg.open()
