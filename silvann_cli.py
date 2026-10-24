#!/usr/bin/env python3
"""SiLVANN in a terminal — a conversation with a running server (start_silvann_server.py), as the UI has one: the answer
written as it comes, its thinking dimmed, the model's tool calls and their values shown, and the tool's asks answered at
the prompt.

    silvann_cli [--endpoint http://localhost:8765] [--notools] [--nothink] [--max-tokens 2048] [--system "…"]
    venv/bin/python silvann_cli.py …                 # the same, from the release's folder

A new conversation takes the server's defaults — its config's thinking, and the model's tool in the config's mode —
unless --notools or --nothink says otherwise. An ask stops the answer: y allows it once, f allows the folder (a read),
n or Enter refuses. In the conversation: /new · /sys <prompt> (a new conversation with it) · /mode ask|chill|full-auto
(what the tool may do without asking) · /quit or Ctrl-D. Ctrl-C stops an answer.
"""
import argparse
import json
import os
import re
import sys
import time
import uuid

try:
    import httpx
except ImportError:
    sys.exit("silvann_cli needs httpx:  pip install httpx")

DIM, CYAN, GREEN, YELLOW, RED, RESET = "\033[2m", "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"
MODES = {"ask": "ask first", "chill": "reads unasked", "full-auto": "full auto"}
TOOL = re.compile(r"<(tool_\w+)>(.*)</\1>", re.S)
SHOWN_LINES = 12                                   # a value's lines shown; the model reads all of them
PICTURE = re.compile(r"!\[[^\]]*\]\(picture:([A-Za-z0-9_.-]+)\)")    # a picture model's answer names its picture


class Chat:
    def __init__(self, a):
        self.a = a
        self.http = httpx.Client(base_url=a.endpoint, headers={"X-Web-User-Id": a.user},
                                 timeout=httpx.Timeout(30.0, read=None))
        self.cid = None
        self.inside = False                        # within the answer's think block
        self.fresh = True                          # the cursor at the start of a line
        self.gap = False                           # after a think block or a call: the blank lines it leaves dropped

    # ── the conversation ───────────────────────────────────────────────────────────────────────────────────────────
    def new(self, system):
        r = self.http.post("/cartridges", json={"name": "cli-" + time.strftime("%H%M%S"), "system": system})
        r.raise_for_status()
        self.cid = r.json()["id"]
        if self.a.notools:
            self.http.patch("/cartridges/%s" % self.cid, json={"tools": False})
        if self.a.nothink:
            self.http.patch("/cartridges/%s" % self.cid, json={"enable_thinking": False})
        d = self.bind()
        tool = ("the tool, %s" % MODES.get(d.get("tools_mode"), d.get("tools_mode"))) if d.get("tools") else "no tool"
        print("%s[a new conversation — thinking %s, %s]%s" % (DIM, "on" if d.get("enable_thinking") else "off", tool,
                                                             RESET))

    def bind(self):
        """The conversation on the server's hands again: one left idle two minutes is put away to its folder."""
        r = self.http.post("/cartridges/%s/bind" % self.cid)
        r.raise_for_status()
        return r.json()

    def mode(self, mode):
        r = self.http.patch("/cartridges/%s" % self.cid, json={"tools_mode": mode})
        if r.status_code != 200:
            print("%s[%s]%s" % (RED, _detail(r), RESET))
        else:
            print("%s[the tool: %s]%s" % (DIM, MODES[mode], RESET))

    def say(self, text):
        """A message, its answer, and the asks it stops on, each answered here."""
        self.bind()
        ask = self.stream("/cartridges/%s/messages" % self.cid, {"text": text, "max_new_tokens": self.a.max_tokens})
        while ask:
            choice = self.decide(ask)
            self.bind()
            ask = self.stream("/cartridges/%s/tools/answer" % self.cid,
                              {"choice": choice, "max_new_tokens": self.a.max_tokens})
        self.line()

    # ── the stream ─────────────────────────────────────────────────────────────────────────────────────────────────
    def stream(self, path, body):
        """One request's answer, written as it comes; the ask it ends on, or None."""
        self.inside, ask = False, None
        for _ in range(60):                        # a turn on the card for another conversation: 503, and again
            try:
                with self.http.stream("POST", path, json=body) as r:
                    if r.status_code == 503:
                        r.read()
                        self.line()
                        print("%s[the card is busy — waiting]%s" % (DIM, RESET), flush=True)
                        time.sleep(float(r.headers.get("Retry-After", 5)))
                        continue
                    if r.status_code != 200:
                        r.read()
                        self.line()
                        print("%s[%d: %s]%s" % (RED, r.status_code, _detail(r), RESET))
                        return None
                    event = None
                    for line in r.iter_lines():
                        if line.startswith("event:"):
                            event = line[6:].strip()
                        elif line.startswith("data:"):
                            d = json.loads(line[5:])
                            if event == "progress":            # a picture model's, as its picture is made
                                sys.stdout.write("\r%s[%s]%s\033[K" % (DIM, d.get("text", ""), RESET))
                                sys.stdout.flush()
                                self.fresh = False
                            elif event == "token":
                                if PICTURE.search(d.get("text", "")):
                                    sys.stdout.write("\r\033[K")
                                    self.fresh = True
                                self.write(PICTURE.sub(lambda m: self.picture(m.group(1)), d.get("text", "")))
                            elif event == "tool":
                                self.tool(d.get("text", ""))
                            elif event == "ask":
                                ask = d
                            elif event == "error":
                                self.line()
                                print("%s[the server: %s]%s" % (RED, d.get("error"), RESET))
                return ask
            except KeyboardInterrupt:
                self.http.post("/cartridges/%s/stop" % self.cid)
                self.line()
                print("%s[stopped]%s" % (DIM, RESET))
                return None
        return None

    def picture(self, name):
        """A picture a picture model made, saved in the working folder: the text that says where."""
        r = self.http.get("/cartridges/%s/pictures/%s" % (self.cid, name))
        if r.status_code != 200:
            return "[the picture %s could not be read: %s]" % (name, _detail(r))
        with open(name, "wb") as f:
            f.write(r.content)
        return "[the picture: %s]" % os.path.abspath(name)

    def write(self, text):
        """The answer's text: its thinking dimmed, without the tags that open and close it."""
        while text:
            tag = "</think>" if self.inside else "<think>"
            at = text.find(tag)
            part, text = (text, "") if at < 0 else (text[:at], text[at + len(tag):])
            if self.gap:
                part = part.lstrip("\n")
                self.gap = not part
            if part:
                sys.stdout.write((DIM + part + RESET) if self.inside else part)
                self.fresh = part.endswith("\n")
            if at >= 0:
                self.inside = not self.inside
                if not self.inside:
                    self.line()
                    self.gap = True
        sys.stdout.flush()

    def tool(self, text):
        """A call or its value, as the server shows it (`<tool_call>FORM</tool_call>`, `<tool_response>…`); an ask
        and its answer are said at the prompt."""
        m = TOOL.fullmatch(text.strip())
        if not m or m.group(1) not in ("tool_call", "tool_response"):
            return
        kind, body = m.groups()
        self.line()
        if kind == "tool_call":
            print("%s> %s%s" % (YELLOW, body, RESET))
        else:
            lines = body.split("\\n")              # a text value's lines, as the model reads them
            more = len(lines) - SHOWN_LINES
            print(DIM + "\n".join(lines[:SHOWN_LINES]) + ("\n… %d more lines" % more if more > 0 else "") + RESET)
        self.gap = True

    def decide(self, ask):
        """The user's answer to an ask: y once, f the folder (a read's), n or Enter refuse."""
        self.line()
        keys = {"y": "once", "n": "refuse"}
        if "folder" in ask.get("choices", []):
            keys["f"] = "folder"
        hint = "y once · f the folder · N refuse" if "f" in keys else "y once · N refuse"
        while True:
            try:
                got = input("%s? the model asks to %s — %s: %s" % (YELLOW, ask.get("question"), hint, RESET))
            except (EOFError, KeyboardInterrupt):
                print()
                got = "n"
            got = got.strip().lower()[:1] or "n"
            if got in keys:
                return keys[got]

    def line(self):
        if not self.fresh:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self.fresh = True


def _detail(r):
    try:
        return r.json().get("detail", r.text)
    except ValueError:
        return r.text[:300]


def main():
    ap = argparse.ArgumentParser(description="A conversation with a running SiLVANN server, in the terminal.")
    ap.add_argument("--endpoint", default="http://localhost:8765", help="the server (default http://localhost:8765)")
    ap.add_argument("--notools", action="store_true", help="no tool for the model, whatever the server's config says")
    ap.add_argument("--nothink", action="store_true", help="no thinking, whatever the server's config says")
    ap.add_argument("--max-tokens", type=int, default=2048, help="the most tokens an answer runs before a call or an end")
    ap.add_argument("--system", default=None, help="a system prompt for the conversation")
    ap.add_argument("--user", default="cli-" + uuid.uuid4().hex[:8], help="the session the conversation belongs to")
    a = ap.parse_args()

    chat = Chat(a)
    system = a.system
    try:
        h = chat.http.get("/health").json()
        print("%s[silvann] %s at %s, a context of %s — /new · /sys <prompt> · /mode ask|chill|full-auto · /quit%s"
              % (DIM, h.get("model_id"), a.endpoint, h.get("max_context"), RESET))
        if system and not h.get("user_system_prompt"):
            print("%s[silvann] the server keeps its own system prompt: --system is not used (its config's "
                  "user_system_prompt)%s" % (DIM, RESET))
        chat.new(system)
    except httpx.HTTPError as ex:
        sys.exit("[silvann] %s: %s" % (a.endpoint, ex))
    while True:
        try:
            line = input("%syou>%s " % (GREEN, RESET)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            break
        try:
            if line == "/new":
                chat.new(system)
            elif line.startswith("/sys "):
                system = line[5:].strip()
                if not h.get("user_system_prompt"):
                    print("%s[silvann] the server keeps its own system prompt: a new conversation with it%s" % (DIM, RESET))
                chat.new(system)
            elif line.split()[0] == "/mode":
                m = line.split()[1:2]
                if m and m[0] in MODES:
                    chat.mode(m[0])
                else:
                    print("%s[/mode %s]%s" % (DIM, "|".join(MODES), RESET))
            else:
                print("%ssilvann>%s " % (CYAN, RESET), end="", flush=True)
                chat.fresh = False
                chat.say(line)
        except httpx.HTTPError as ex:
            chat.line()
            print("%s[%s]%s" % (RED, _detail(ex.response) if isinstance(ex, httpx.HTTPStatusError) else ex, RESET))
    try:
        chat.http.post("/cartridges/%s/unbind" % chat.cid)       # saved, and the card free for another
    except httpx.HTTPError:
        pass
    print("%s[silvann] bye%s" % (DIM, RESET))


if __name__ == "__main__":
    main()
