# The UI

The chat in a browser: `silvann ui`, or `venv/bin/python silvann_ui.py --port 8090` from the release's folder, then
http://localhost:8090. The Docker image's app is the same UI. It only talks HTTP to a SiLVANN server, on this machine or
another one; it can also start and stop a server on this machine (**Options**).

## The left drawer

- **Chat**: the conversations, under the Chat row. Click one to open it.
- **Options**: the models on this machine, their configs, and a server to start and stop.
- **Help**: this guide and the release's documents.

The dot next to "SiLVANN" is the server's state: **gray** not asked yet, **green** answering, **red** unreachable. Its
tooltip names the model. **Endpoint**, at the bottom, is the server the UI talks to: `http://localhost:8765` when both
run here. **connect** asks it again.

## Conversations

Each conversation in the list has an icon:

- a **chip**: the server holds it now, so it answers at once;
- a **bubble**: it is saved on the server's disk, and is read back when it is opened and written to;
- a **lock**: it has a password;
- a **⚠** after its name: it belongs to another model than the one the server runs, and that server will not open it.

The arrow at its right opens its menu: **Rename**; **Export full session**, a `.silvann.tar` with its state that another
server running the same model can import; **Export chat history (.md)**, its messages as a Markdown file; **Delete**,
which removes it from the server's disk.

**+ new** starts one. The dialog shows the model's context window (its config's `max_context`), how much RAM it takes
saved, and on a model over several cards each card's share of its cache. **import** takes a `.silvann.tar` exported
earlier, and is refused when it was made with another model.

A conversation opened from the list waits off the card. A banner says so; typing a message, or **Activate**, brings it
on.

## The chat

The top bar, left to right:

- **The conversation's name.**
- **The speed**: tokens a second while an answer streams, then the answer's tokens and time. A picture model says how far
  its picture has got here.
- **The context meter**: the conversation's tokens of the most the model takes, as "4,210 / 32,768". The bar turns
  amber at three quarters and red at nine tenths. Its total is the context the server booted with. With
  `"max_context": "auto"` that is the model's whole length where the card holds it.
- **thinking mode**: the model thinks before it answers, where the model can. It applies from the next message.
- **show thinking**: each answer's thinking is shown, in a section headed "thinking". Off, it is hidden, and
  "thinking…" shows while the model thinks.
- **tools**: the model's tool, a Lisp REPL on the server's machine ([`tools.md`](tools.md)). A conversation that began
  without it cannot turn it on. Beside it, what the tool may do without asking you:
  - **ask first** reads in its workspace, and asks for anything else;
  - **reads unasked** reads anywhere, and asks to write, run a command or fetch a page;
  - **full auto** asks nothing.
- **The lock**: set, change or remove the conversation's password. Others using the same server then cannot open it.

A run of the model's calls to its tool is folded in a section headed "tool · N calls": each call after `>`, its value
under it.
When a call needs your answer, a card below the messages says what the model asks to do, with **allow once** and
**refuse**; for a read also **allow the folder**, after which every conversation reads there without asking.

Below the messages:

- **📎** attaches a file to the next message. A document (PDF, Word, text, Markdown, CSV, JSON) is read as text by any
  model. A picture or a video needs a model that reads them: the Qwen models and GLM 5.3 Flash. The server reads it when
  it is attached, a chip shows it, and the message carries it. It stays in the conversation and is saved with it.
- **The message box**: **Enter** sends, **Shift+Enter** is a new line.
- **max tokens** caps an answer, up to 4,096. The model stops sooner when it is done.
- **The send button** becomes a stop button while an answer streams, and stops it.

One conversation is answered at a time. A message sent while another is being answered waits ("⏳ waiting my turn…")
and then streams.

**A picture model** (Qwen-Image 2.1) makes a picture of each message, shown in the chat and saved with the conversation.
Its chat has no thinking and no tool switches, and no context meter.

## Options: a server on this machine

**Mode**:
- **Remote** talks to a server started elsewhere: set its address in the Endpoint field.
- **Local** starts one on this machine: the rest of this section.

**Models** lists the models this release can fetch, their sizes and which are here.
- **download** fetches one from Hugging Face into the data folder's `models/`, with its progress. One downloads at a
  time. A download is refused when the disk has less room than the model needs; stopped and started again, it keeps the
  files it finished.
- A **Hugging Face token** (read access, from huggingface.co/settings/tokens) makes the downloads faster. **save**
  checks it with Hugging Face and keeps it for every download after; **forget** removes it.

Pick a **model** and one of its **configs**, a **Port**, and **start**. The UI runs the server, points the Endpoint at
it, and shows its output in the **Subprocess log**. Under **status** the server says what it chose for this machine:
where the experts are, the tier on the card, the context. A server that cannot start says why there, and what to change.
**stop** saves the conversations and stops it.

A config is a file in the model's `configs/`. **new config** writes one with every key the server reads at its default,
and the model's own options listed under `_options`. **edit selected** opens one as JSON, and **delete selected**
removes it (not the model). [`config.md`](config.md) says what each key does. For example:

```json
{ "max_context": "auto", "thinking": true, "tools": "chill", "lora": null, "sampler": "generation", "options": { } }
```

## Where things are

- **The data folder** holds the models and the conversations: `models/<name>/` and `user_sessions/<model>/<id>/`, on
  the server's machine. It is the release's folder, or the one `silvann models <folder>` named (for `silvann ui` and
  `silvann serve`), or `SILVANN_DATA`. In the Docker image it is the volume, `/data`.
- **A model's folder**: its configs, and the programs it runs as written at its last start
  (`models/<name>/programs.lisp`). The setup the server chose is kept there too (`autotune.json`).
- **The UI's own settings** (the endpoint, the last conversation, the conversations' passwords you typed): kept by the
  UI, on its machine, for this browser — in `.nicegui/` beside `silvann_ui.py`, or the volume's `/data/.nicegui` in the
  Docker image. The passwords are there as you typed them.
- **The Hugging Face token** that Options saves: `.huggingface_token` in the data folder, readable by its owner alone.
  An export of the Docker image's volume carries it, and the UI's settings with the passwords: **forget** the token,
  and remove `.nicegui/`, before giving an export to someone else.
