"""A turn: a message sent, its answer streamed, the model's asks answered, the answer stopped."""
import asyncio
import time

import httpx

from .part import Part
from .session import LIVE
from .sign_in import LOCAL_USERS


class Turn(Part):
    """A message sent and its answer streamed."""

    def wire(self):
        """The send button and the Enter key."""
        self.set_send_btn(self.state["streaming"])                     # a page opened while this browser's answer streams: its stop

        self.chat.send_btn.on("click", self.send_or_stop)

        # Plain Enter sends — `.exact.prevent` stops the browser from inserting a newline locally
        # before set_value("") round-trips back from the server. Shift+Enter has no listener attached
        # so the browser's default newline behavior runs.
        self.chat.input_box.on("keydown.enter.exact.prevent",
                               lambda e: asyncio.create_task(self.send_or_stop()))

    # ----- Chat input wiring -----
    def set_send_btn(self, streaming: bool):
        if streaming:
            self.chat.send_btn.props("icon=stop color=negative")
        else:
            self.chat.send_btn.props("icon=send color=primary")

    def drop_message(self, m: dict):
        """The message `m` out of the conversation, wherever it is in it."""
        for i, x in enumerate(self.state["history"]):
            if x is m:
                del self.state["history"][i]
                return

    def streams(self, before: list, turn: list) -> dict:
        """This page's answer recorded as the browser's (▶ `LIVE`): the conversation `before` it, its `turn`, the answer
        last — the stream writes the answer there."""
        record = {"task": asyncio.current_task(), "cart": self.state["active_id"], "before": before, "turn": turn,
                  "ask": None, "seen": 0}
        LIVE[self.browser_id] = record
        return record

    def page_here(self) -> bool:
        """Whether this page is still open: a page gone leaves the browser's state to the pages after it."""
        return not self.page_client.is_deleted

    def hold(self):
        """⭐ THIS MACHINE'S SIGN-IN HELD WHILE AN ANSWER STREAMS (▶ users.Users.hold): the model may work for an hour
        with nothing asked of it — its ten minutes begin as the answer ends (`release`). The server holds the sign-in
        its own request carries; this is the page's own, where the server is another machine's. Answers the token."""
        token = self.state.get("user_token") if self.me is not None else None
        if token:
            LOCAL_USERS.hold(token)
        return token

    def release(self, token):
        if token:
            LOCAL_USERS.release(token)

    async def send(self):
        if self.state["streaming"]:
            return
        txt = (self.chat.input_box.value or "").strip()
        attached = list(self.chat._cart_pending()) if self.state["active_id"] else []
        if not txt and not attached:
            return
        if self.state.get("archived_active"):
            self.safe_notify("This conversation is archived: it is shown, and not written to", type="warning")
            return
        # a message typed with no conversation open goes to a new one, and one not online is brought online: typing is
        # the user's say-so, as the Activate button is
        if self.state["active_id"] is None and not await self.conversations.start_conversation():
            return
        if not self.state.get("active_bound"):
            await self.conversations.bind_active()
            if not self.state.get("active_bound"):
                return
        if self.state.get("ask"):
            self.safe_notify("The model waits on your answer to its ask — the buttons under its answer", type="warning")
            return
        # Explicit set_value pushes the empty string back to the client. Plain assignment doesn't
        # always trigger the value-update event for q-input/textarea, hence the explicit call.
        self.chat.input_box.set_value("")
        self.state["streaming"] = True
        self.set_send_btn(True)
        image_ids = [a["image_id"] for a in attached]
        whole = [a["image_id"] for a in attached if a.get("whole")]
        display_user = txt
        if attached:
            # a document the server keeps as a file for the model to look into is said so beside its name
            display_user = "📎 " + ", ".join(
                a["filename"] + (" (kept as a file, not read whole)" if a.get("carried") == "named" and not a.get("whole")
                                 and a.get("kind") == "document" else " (a file, named to the model)" if a.get("kind") == "file" else "")
                for a in attached) + (f"\n\n{txt}" if txt else "")
        before = [dict(m) for m in self.state["history"]]
        self.state["history"].append({"role": "user", "content": display_user})
        self.state["history"].append({"role": "assistant", "content": ""})
        asked, msg = self.state["history"][-2], self.state["history"][-1]      # this page's copies of the turn
        answer = {"role": "assistant", "content": ""}                 # the stream's own (▶ `LIVE`)
        record = self.streams(before, [dict(asked), answer])
        self.chat.add_messages(len(self.state["history"]) - 2)
        self.chat.follow_chat(force=True)
        # The engine serializes turns, so if another conversation is mid-turn the server holds our
        # request until it frees up. Surface that as "waiting my turn" rather than a dead UI; the
        # first received event clears the placeholder. Cheap single poll — no busy-loop.
        waiting = False
        try:
            b = await self.client.busy()
            other = b.get("cart")
            if b.get("busy") and other and other != self.state["active_id"]:
                answer["content"] = "_⏳ waiting my turn…_"
                waiting = True
                self.chat.mark_live()
        except Exception:
            pass
        # Speed counter: walltime starts at the *first* received token to exclude server prefill
        # time (which can dominate on long conversations / offload mode) from the rate.
        first_token_at = None
        n_tokens = 0
        self.chat.speed_label.text = "queued…" if waiting else "…"
        held = self.hold()
        try:
            async for kind, payload in self.client.stream_message(self.state["active_id"], txt, images=image_ids,
                                                                    whole=whole):
                if kind == "waiting":                      # the card is serving another turn; asked again in 5 s
                    if not waiting:
                        waiting = True
                        answer["content"] = "_⏳ waiting my turn…_"
                        self.chat.speed_label.text = "queued…"
                        self.chat.mark_live()
                    continue
                if waiting:
                    waiting = False
                    answer["content"] = ""   # drop the "waiting my turn" placeholder
                if kind == "token":
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                    n_tokens += 1
                    answer["content"] += payload.get("text", "")
                    if "used" in payload:
                        self.chat.show_context(used=payload["used"])
                    elapsed = time.monotonic() - first_token_at
                    if elapsed > 0.3 and n_tokens > 1:
                        rate = (n_tokens - 1) / elapsed   # exclude the first token from the divisor
                        self.chat.speed_label.text = f"{rate:.1f} tok/s"
                    self.chat.mark_live()
                elif not self.on_tool_event(kind, payload, record):
                    break
        except httpx.HTTPStatusError as ex:
            if not self.page_here():
                pass                                   # the page that went: the pages after it read the conversation anew
            elif ex.response.status_code == 409 and (self.state.get("ask") or "waits on an answer" in ex.response.text):
                # the model's ask is still open — a page that was put away and back may not have known it: its buttons
                # again, and the typed message kept for after the answer
                self.drop_message(msg)          # the empty assistant message
                self.drop_message(asked)        # the user message they typed
                self.chat.input_box.set_value(txt)
                try:
                    self.state["ask"] = (await self.client.activate(self.state["active_id"])).get("ask") or self.state.get("ask")
                except Exception:
                    pass
                self.safe_notify("The model waits on your answer to its ask — the buttons under its answer", type="warning")
            elif ex.response.status_code == 409:
                # Server says cart isn't bound anymore — almost certainly evicted (we went idle
                # past the heartbeat window). Drop the assistant placeholder, flip back to
                # banner-mode, and let the user re-Activate.
                self.drop_message(msg)          # the empty assistant message
                self.drop_message(asked)        # the user message they typed
                self.state["active_bound"] = False
                self.chat.bind_banner_label.text = "Session evicted (idle). Send again, or click Activate, to bring it back online."
                self.chat.input_box.set_value(txt)  # restore the typed message so it's not lost
                self.safe_notify("Session was evicted while idle — send again, or click Activate.", type="warning")
            elif ex.response.status_code == 423:
                answer["content"] += "\n\n_[server: in use by another session]_"
            else:
                answer["content"] += f"\n\n_[stream error: HTTP {ex.response.status_code}]_"
        except Exception as e:
            answer["content"] += f"\n\n_[stream error: {e}]_"
        finally:
            self.release(held)
            if first_token_at is not None and n_tokens > 1:
                elapsed = time.monotonic() - first_token_at
                rate = (n_tokens - 1) / elapsed if elapsed > 0 else 0
                self.chat.speed_label.text = f"{n_tokens} tok · {elapsed:.1f}s · {rate:.1f} tok/s"
            else:
                self.chat.speed_label.text = "—"
            self.set_send_btn(False)
            if self.page_here():
                self.chat.follow_answer(record)
                self.state["streaming"] = False
                # Clear any consumed images from the chip row. Server has either consumed
                # them (success) or rejected the request (error); either way the IDs are
                # one-shot and re-using them would 404.
                if attached:
                    cart_id = self.state.get("active_id")
                    if cart_id:
                        self.state["pending_images"][cart_id] = []
                    self.chat.render_pending_images()
                self.chat.finish_live()

    def on_tool_event(self, kind, payload, record: dict):
        """An answer's event other than a token, into its `record`'s answer (▶ `LIVE`): a call to the model's tool and
        its value, the ask that stops it, or the server's error. False where the stream is over."""
        answer = record["turn"][-1]
        if kind == "tool":
            answer["content"] += payload.get("text", "")
            if payload.get("made"):                        # what the call made among the conversation's files: its chips
                answer["made"] = answer.get("made", []) + [m for m in payload["made"] if m not in answer.get("made", [])]
            self.chat.mark_live()
        elif kind == "progress":                          # a picture model's, as the picture is made: said beside the box
            self.chat.speed_label.text = payload.get("text", "")
        elif kind == "ask":
            record["ask"] = payload
            if self.page_here():
                self.state["ask"] = payload
        elif kind == "done" and "used" in payload:
            self.chat.show_context(used=payload["used"], context=payload.get("context"))
        elif kind == "error":
            answer["content"] += f"\n\n_[server error: {payload.get('error')}]_"
            self.chat.mark_live()
            return False
        return True

    async def answer_ask(self, choice: str):
        """The user's answer to the model's ask, then the rest of its answer, into the conversation's last message."""
        if self.state["streaming"] or not self.state.get("ask") or not self.state["active_id"]:
            return
        # a conversation put away while the ask waited is brought back first, as sending brings it back
        if not self.state.get("active_bound"):
            await self.conversations.bind_active()
            if not self.state.get("active_bound"):
                return
        asked = self.state["ask"]
        self.state["ask"] = None
        self.state["streaming"] = True
        self.set_send_btn(True)
        if not self.state["history"] or self.state["history"][-1]["role"] != "assistant":
            self.state["history"].append({"role": "assistant", "content": ""})
        answer = dict(self.state["history"][-1])        # the answer goes on from where the ask stopped it
        record = self.streams([dict(m) for m in self.state["history"][:-1]], [answer])
        self.chat.render_chat()
        held = self.hold()
        try:
            async for kind, payload in self.client.stream_answer(self.state["active_id"], choice):
                if kind == "token":
                    answer["content"] += payload.get("text", "")
                    if "used" in payload:
                        self.chat.show_context(used=payload["used"])
                    self.chat.mark_live()
                elif not self.on_tool_event(kind, payload, record):
                    break
        except httpx.HTTPStatusError as ex:
            answer["content"] += f"\n\n_[stream error: HTTP {ex.response.status_code}]_"
            if ex.response.status_code in (409, 423) and self.page_here():
                self.state["ask"] = asked                  # not answered: its buttons stay
        except Exception as e:
            answer["content"] += f"\n\n_[stream error: {e}]_"
        finally:
            self.release(held)
            self.set_send_btn(False)
            if self.page_here():
                self.chat.follow_answer(record)
                self.state["streaming"] = False
                self.chat.finish_live()

    async def stop(self):
        if not self.state["streaming"] or not self.state["active_id"]:
            return
        try:
            await self.client.stop_generation(self.state["active_id"])
            self.safe_notify("stop signal sent — finishing current token", type="info")
        except httpx.HTTPStatusError as ex:
            # Older server doesn't have /stop. The stream will continue server-side; client UI
            # can't cleanly interrupt without it. Surface the limitation.
            if ex.response.status_code == 404:
                self.safe_notify("server doesn't support stop (restart it to pick up the new endpoint)",
                          type="warning")
            else:
                self.safe_notify(f"stop failed: {ex}", type="negative")
        except Exception as e:
            self.safe_notify(f"stop failed: {e}", type="negative")

    async def send_or_stop(self):
        if self.state["streaming"]:
            await self.stop()
        else:
            await self.send()
