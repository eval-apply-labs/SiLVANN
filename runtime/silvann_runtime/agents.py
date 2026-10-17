"""A MODEL'S SUBAGENT — a conversation of a SiLVANN server, driven through its HTTP API as silvann_cli drives one
(▶ `src/docs/design/pending/agents.md`).

⚖ 2026-10-16: *"(agent 'model 'config_str(including the lora) 'task) and make it a repl, so it is a silvann_cli session
and the sender can exit itself once it is completed"*. A `Link` creates the conversation on a server — the caller's own
(loopback), a worker, or the caller's lent to another model — with its LoRA and its tool (chill, the commands its launch
allows), sends the task, and answers each reply; an ask the subagent makes comes back in the reply, and is answered with
`answer`. `close` deletes the conversation.
"""
import json

import httpx

# what a subagent's REPL has when its config names nothing: reading, and Lisp
READ_ONLY = ("help", "ls", "read", "pwd", "cd", "library", "import", "skills", "guide", "find")
ANSWER_TOKENS = 4096          # a reply's tokens at most, its thinking and its calls with it


class Reply:
    """What a subagent answered: its text (its thinking left out), the calls it made, and an ask it waits on, if any."""

    def __init__(self, text, calls, ask):
        self.text, self.calls, self.ask = text, calls, ask


class Link:
    """A subagent's session on the server at `endpoint`: `first`, the reply to the task; `say`, `answer`, `close`."""

    def __init__(self, endpoint, model, config, task, user="agent", progress=None):
        self.endpoint, self.model, self.config = endpoint.rstrip("/"), model, config
        self.name = model + (" (%s)" % config["lora"] if config.get("lora") else "")
        self.progress = progress                  # (text) said as the subagent works, or None
        self.http = httpx.Client(base_url=self.endpoint, timeout=httpx.Timeout(30.0, read=None),
                                 headers={"X-Web-User-Id": user})
        self.ask, self.cid = None, None
        made = self.http.post("/cartridges", json={"name": "agent: " + task[:60], "lora": config.get("lora"),
                                                   "system": config.get("system"),
                                                   "tools_allowed": list(config.get("tools_allowed") or READ_ONLY),
                                                   "agent": True})
        made.raise_for_status()
        self.cid = made.json()["id"]
        self.http.patch("/cartridges/%s" % self.cid, json={"tools": True, "tools_mode": "chill"}).raise_for_status()
        self.http.post("/cartridges/%s/bind" % self.cid).raise_for_status()
        self.first = self.say(task)

    def say(self, text):
        return self._stream("/cartridges/%s/messages" % self.cid, {"text": text, "max_new_tokens": ANSWER_TOKENS})

    def answer(self, choice):
        self.ask = None
        return self._stream("/cartridges/%s/tools/answer" % self.cid, {"choice": choice, "max_new_tokens": ANSWER_TOKENS})

    def _stream(self, path, body):
        out, calls, kind, ask = "", [], None, None
        with self.http.stream("POST", path, json=body) as r:
            if r.status_code != 200:
                r.read()
                raise RuntimeError("the agent's server answered %d: %s" % (r.status_code, r.text[:300]))
            for line in r.iter_lines():
                if line.startswith("event:"):
                    kind = line[6:].strip()
                elif line.startswith("data:"):
                    d = json.loads(line[5:])
                    if kind == "token":
                        out += d.get("text", "")
                    elif kind == "tool" and d.get("form"):
                        calls.append(d["form"])
                        if self.progress:
                            self.progress("the agent calls %s" % d["form"][:80])
                    elif kind == "ask":
                        ask = d
                    elif kind == "error":
                        raise RuntimeError("the agent failed: %s" % d.get("error"))
        self.ask = ask
        return Reply(out.split("</think>")[-1].strip(), calls, ask)

    def shown(self, reply):
        """A reply as the caller's REPL answers it: headed by the agent, its calls named, its ask and how to go on."""
        lines = ["[agent %s] %s" % (self.name, reply.text or "(no text)")]
        if reply.calls:
            lines.append("[it called: %s]" % " · ".join(c[:60] for c in reply.calls))
        if reply.ask:
            lines.append("[it asks: %s — (answer 'once) or (answer 'refuse)]" % reply.ask["question"])
        else:
            lines.append("[your next call is a message to it; (exit) ends the session]")
        return "\n".join(lines)

    def close(self):
        if self.cid is not None:
            try:
                self.http.delete("/cartridges/%s" % self.cid)
            except httpx.HTTPError:
                pass
            self.cid = None
        self.http.close()
