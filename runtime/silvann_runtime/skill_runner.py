"""A TOOL THE MODEL ADDED, RUN OUTSIDE THE SERVER (NN-128, ▶ tools' `add_tool`) — its Python never in the server's own
process, where the tools file's settings and the server's memory are, but in one of its own, confined as the model's
`(bash …)` is (▶ jail.py).

    python -I skill_runner.py PAYLOAD      PAYLOAD: {"folder", "file", "function", "args", "context"} as JSON

The function is called with the form's arguments and, where it takes one, `context` (cwd, workspace, its folder, its
settings — the tools file's entry's); what it prints goes to the command's output, and its answer, or what it raised,
comes last on a line of its own after MARK, as JSON.
"""
import contextlib
import importlib.util
import inspect
import json
import os
import sys

MARK = "\x1esilvann-tool-answer "


def plain(value):
    """An answer as JSON holds it: texts, numbers, truth, nothing, lists and dicts; anything else its text."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    return str(value)


def main():
    payload = json.loads(sys.argv[1])
    folder = payload["folder"]
    os.chdir(payload["context"]["cwd"])
    try:
        spec = importlib.util.spec_from_file_location("silvann_added_tool", os.path.join(folder, payload["file"]))
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, folder)
        with contextlib.redirect_stdout(sys.stderr):
            spec.loader.exec_module(module)
            fn = getattr(module, payload["function"], None)
            if not callable(fn):
                raise LookupError("%s has no function %s" % (payload["file"], payload["function"]))
            try:
                takes = "context" in inspect.signature(fn).parameters
            except (TypeError, ValueError):
                takes = False
            got = {"value": plain(fn(*payload["args"], **({"context": payload["context"]} if takes else {})))}
    except Exception as ex:                           # noqa: BLE001 — what it raised: the model's to read
        got = {"error": "%s: %s" % (type(ex).__name__, ex)}
    sys.stderr.flush()
    sys.stdout.write("\n" + MARK + json.dumps(got) + "\n")


if __name__ == "__main__":
    main()
