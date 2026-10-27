---
name: search
description: the web searched through this machine's SearXNG, or the Brave Search API with a key
enabled: auto
check: search.py:available
config: {"searxng": "auto", "brave_key": null}
---
A skill the release ships (NN-121): search engines turn away a program that reads their results pages, so the model
searches through an engine of the machine's own — a SearXNG instance's JSON API — or Brave's search API with a key.

Its settings, the tools file's `search` entry (▶ docs/tools.md):
- `searxng`: `"auto"` — the first SearXNG that answers at `http://127.0.0.1:8888` (this machine's container) or
  `http://searxng:8080` (a container called `searxng` on the server's network) — an address `"http://host:port"`, or
  `null` for none. Its JSON format must be on (`search: formats: [html, json]` in its settings.yml).
- `brave_key`: a Brave Search API key, asked when there is no SearXNG or it does not answer; `null` for none.

The server checks it as it starts (`available`): with no SearXNG to reach and no key, `(help)` says so and the model
does not reach for it.

## (search "query" [count])
touches: net
run: search.py:search
The web searched, the first 8 results or `count` (at most 20), each a title, an address and a few words. A result's
few words are the page's own; read the page with (fetch "address"). It asks this machine's
SearXNG or a search API, never a search engine's own pages, which turn robots away.
