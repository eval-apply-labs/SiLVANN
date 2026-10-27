"""The search skill (▶ SKILL.md): the web searched through this machine's SearXNG, or the Brave Search API with a key.

Its settings come in `context["config"]`, the tools file's `search` entry — `searxng` ("auto", an address, or null) and
`brave_key` — and never reach the model.
"""
import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request

# ⭐ WHERE A SEARXNG IS LOOKED FOR, with `searxng` "auto": this machine's own container, as docs/tools.md starts it, and a
#   container called `searxng` on the server's container network
PLACES = ("http://127.0.0.1:8888", "http://searxng:8080")
AGENT = "SiLVANN-search/1 (+https://github.com/eval-apply-labs/SiLVANN)"
BRAVE = "https://api.search.brave.com/res/v1/web/search"
RESULTS = 8                      # the results answered when not told how many
RESULTS_MAX = 20
SECONDS = 20
PROBE_SECONDS = 3
MOST = 4 << 20                   # the most of an answer read
_found = {}                      # "auto" -> the SearXNG found there, for the process


class Failed(Exception):
    """An engine that did not answer, and what it said."""


def _get(url, headers, seconds, most=MOST):
    req = urllib.request.Request(url, headers=dict({"User-Agent": AGENT}, **headers))
    with urllib.request.urlopen(req, timeout=seconds) as r:
        return r.read(most)


def _places(config):
    said = config.get("searxng", "auto")
    if said is None or str(said).lower() in ("", "no", "none"):
        return []
    return list(PLACES) if str(said).lower() == "auto" else [str(said).rstrip("/")]


def _probe(at):
    """None where the SearXNG at `at` answers in JSON, else why not — asked an empty query, which searches nothing: 400
    where its JSON is on, 403 where it is off (`MEASURED` on node03's, measurements/2026-10-26_nn121_search_node03.md)."""
    try:
        _get(at + "/search?q=&format=json", {"Accept": "application/json"}, PROBE_SECONDS, 4096)
        return None
    except urllib.error.HTTPError as ex:
        if ex.code == 400:
            return None
        if ex.code == 403:
            return "the SearXNG at %s refuses JSON (its settings.yml's search: formats: needs json)" % at
        return "the SearXNG at %s: %s" % (at, ex)
    except OSError:
        return "no SearXNG answers at %s" % at


def _searxng_at(config):
    """The SearXNG to ask and, where there is none, why: the one named, as it is; with "auto" the first of PLACES that
    answers, kept once found."""
    places = _places(config)
    if len(places) == 1 and str(config.get("searxng")).lower() != "auto":
        return places[0], None
    if not places:
        return None, None
    if _found.get("auto"):
        return _found["auto"], None
    said = []
    for at in places:
        why = _probe(at)
        if why is None:
            _found["auto"] = at
            return at, None
        said.append(why)
    return None, "; ".join(said)


def available(config):
    """The startup check: None where a search can be answered — a SearXNG answers, or a Brave key is set — else why not."""
    _found.pop("auto", None)
    places, said = _places(config), []
    for at in places:
        why = _probe(at)
        if why is None:
            if str(config.get("searxng")).lower() == "auto":
                _found["auto"] = at
            return None
        said.append(why)
    if config.get("brave_key"):
        return None
    if not places:
        said.append("no SearXNG is set")
    return ("%s, and no Brave Search API key is set — the tools file's search entry names them (searxng, brave_key)"
            % "; ".join(said))


def _searxng(at, query, count):
    url = "%s/search?%s" % (at, urllib.parse.urlencode({"q": query, "format": "json"}))
    try:
        got = json.loads(_get(url, {"Accept": "application/json"}, SECONDS).decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as ex:
        if ex.code == 403:
            raise Failed("the SearXNG at %s refuses JSON: its settings.yml's search: formats: needs json" % at)
        raise Failed("the SearXNG at %s: %s" % (at, ex))
    except (OSError, ValueError) as ex:
        raise Failed("the SearXNG at %s: %s" % (at, ex))
    return [(r.get("title") or "", r.get("url") or "", r.get("content") or "")
            for r in got.get("results", []) if r.get("url")][:count]


def _brave(key, query, count):
    url = "%s?%s" % (BRAVE, urllib.parse.urlencode({"q": query, "count": count}))
    try:
        got = json.loads(_get(url, {"Accept": "application/json", "X-Subscription-Token": key}, SECONDS)
                         .decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as ex:
        try:                                          # its refusals say why in the body: {"error": {"detail": "…"}}
            why = json.loads(ex.read(MOST).decode("utf-8", errors="replace"))["error"]["detail"]
        except (OSError, ValueError, KeyError, TypeError):
            why = ""
        raise Failed("Brave Search: HTTP %d %s" % (ex.code, why or ex.reason or ""))
    except (OSError, ValueError) as ex:
        raise Failed("Brave Search: %s" % ex)
    plain = lambda s: html.unescape(re.sub(r"<[^>]+>", "", s or ""))   # noqa: E731 — its snippets mark the words found
    return [(plain(r.get("title")), r.get("url") or "", plain(r.get("description")))
            for r in got.get("web", {}).get("results", []) if r.get("url")][:count]


def search(query, count=RESULTS, context=None):
    """`query`'s first `count` results, numbered — title, address, snippet — from the SearXNG first, then Brave with the
    key; what each said when neither answers, raised."""
    config = (context or {}).get("config") or {}
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= RESULTS_MAX:
        raise ValueError("(search \"query\" count) answers 1 to %d results" % RESULTS_MAX)
    said = []
    at, why = _searxng_at(config)
    found = engine = None
    if at:
        try:
            found, engine = _searxng(at, query, count), "SearXNG at %s" % at
        except Failed as ex:
            said.append(str(ex))
    elif why:
        said.append(why)
    key = config.get("brave_key")
    if found is None and key:
        try:
            found, engine = _brave(key, query, count), "Brave Search"
        except Failed as ex:
            said.append(str(ex))
    if found is None:
        raise LookupError("; ".join(said) if said else "no search is set up on this machine: the tools file's search "
                          "entry names a SearXNG (searxng) or a Brave Search API key (brave_key) — docs/tools.md")
    if not found:
        return "no results for %s (%s)" % (json.dumps(query, ensure_ascii=False), engine)
    return ("%d result%s from %s — (fetch \"address\") reads one:\n" % (len(found), "" if len(found) == 1 else "s", engine)
            + "\n".join("%d. %s\n   %s%s" % (i + 1, " ".join(title.split()), address,
                                             "\n   " + " ".join(snippet.split()) if snippet.strip() else "")
                        for i, (title, address, snippet) in enumerate(found)))
