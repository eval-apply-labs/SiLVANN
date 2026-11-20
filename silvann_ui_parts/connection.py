"""The server the page talks to: whether it answers, what it serves, and a change of either."""
from typing import Optional

from .part import Part


class Connection(Part):
    """The server's answer to a ping, and what follows a change of server or model."""

    def wire(self):
        """The connect button."""
        if self.admin:
            self.options.connect_btn.on("click", self.connect_clicked)

    async def ping(self):
        try:
            h = await self.client.health()
            self.shell.status_dot.classes(replace="text-green-500").style("font-size: 12px")
            self.shell.status_tip.text = (f"OK — {h['model_id']} · layers {h['pp_slice']} · "
                                          f"gpus {h['num_gpus']} · tenants {h['tenants_loaded']}"
                                          + (f" · active {h['active'][:8]}" if h['active'] else ""))
            self.chat.attach_btn.set_visibility(bool(h.get("images") or h.get("files")))
            # a picture model makes a picture of each message: it has no thinking and no tool to switch
            for switch in (self.chat.thinking_mode_toggle, self.chat.show_think_toggle, self.chat.tools_toggle, self.chat.tools_mode):
                switch.set_visibility(not h.get("pictures"))
            self.chat.show_context(context=0 if h.get("pictures") else h.get("max_context", 0))
        except Exception as e:
            self.shell.status_dot.classes(replace="text-red-500").style("font-size: 12px")
            self.shell.status_tip.text = f"down: {type(e).__name__}: {e}"
            self.chat.attach_btn.set_visibility(False)
            return
        if await self.model_changed(h.get("model_id")):
            self.close_chat()
            with self.page_client:                              # the timer's task has no slot of its own to notify in
                self.safe_notify(f"{h.get('model_id')} is serving now: its conversations are in the menu")
            await self.conversations.render_convs()
        await self.conversations.first_conversation(h.get("model_id"))

    async def model_changed(self, model: Optional[str]) -> bool:
        """Whether the server serves another model than the open conversation's. Not while an answer runs, here or on
        the server: a subagent lent the card shows its own model until the answer that called it is done."""
        if not self.state.get("active_id") or self.state["streaming"]:
            return False
        try:
            if self.state.get("active_model"):
                if model == self.state["active_model"]:
                    return False
            elif self.state["active_id"] in [c["id"] for c in await self.client.list_cartridges()]:
                self.state["active_model"] = model              # a conversation opened before its model was kept
                return False
            b = await self.client.busy()
            if b.get("busy") or b.get("cart"):
                return False
            if self.state.get("active_model"):
                return (await self.client.health()).get("model_id") != self.state["active_model"]
            return self.state["active_id"] not in [c["id"] for c in await self.client.list_cartridges()]
        except Exception:
            return False

    def close_chat(self):
        """The chat emptied of a conversation another model made: the menu lists the serving model's."""
        self.state["active_id"] = self.state["active_name"] = self.state["active_model"] = None
        self.state["history"] = []
        self.state["active_bound"] = False
        self.state["context_used"] = 0
        self.chat.speed_label.text = "—"                             # the last model's rate
        self.chat.render_chat()

    async def connect_clicked(self):
        self.state["endpoint"] = self.options.endpoint_in.value
        self.client.set_endpoint(self.options.endpoint_in.value)
        self.client.use_token(self.page.token_for(self.options.endpoint_in.value))
        await self.ping()
        await self.conversations.render_convs()

    async def on_server_mode(self, mode):
        """The server chosen at the bottom of Options: this machine's, at its port, or the address given."""
        self.state["server_mode"] = mode
        if mode == "local":
            self.options.endpoint_in.value = f"http://localhost:{int(self.state['server_port'])}"
        await self.connect_clicked()
