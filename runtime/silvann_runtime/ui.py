"""THE REPL'S HANDS ON A SCREEN (NN-139) — the pointer, a picture around it, clicks, typing, keys and scrolling on an X11
display, for the REPL's `(ui …)` (▶ tools.py).

⚖ 2026-10-28: *"a side project would be a lisp browser, where you can ask for a (browser mouse_move x y) (browser image
'percent) that grabs just the percentage of screen specified centered on the mouse pointer, (browser mouse_click left)
(browser type '"text") … maybe browser is not the right word, it is ui"*.

The input goes through X's XTest extension, `libX11` and `libXtst` by ctypes — every X11 desktop has both — and the
picture is Pillow's `ImageGrab` of the display, so nothing is added to what the server installs. A Wayland session's
screen is not reached: its compositor takes no input from another program and gives no picture of what is not X's.
Windows' and macOS' are not here yet.
```
  Screen(display)       the display opened: ":0", "host:N", or None for $DISPLAY — `Unreachable` where it is not
  .size()  .pointer()   (width, height) and the pointer's (x, y)
  .move(x, y)           the pointer moved there
  .click(button, n)     1 left, 2 middle, 3 right, n times
  .scroll(n)            n notches, down where n > 0
  .type(text)           the text typed a character at a time, whatever the keyboard's layout
  .key("ctrl+s")        a chord: its modifiers held, its key pressed and everything let go
  .grab()               the screen as a Pillow image
```
"""
import ctypes
import ctypes.util
import os
import time

KEY_DELAY = 0.008                 # between a key's press and release, and one key and the next: a toolkit keeps up
CLICK_DELAY = 0.04                # between a double click's two clicks
MODIFIERS = {"ctrl": "Control_L", "control": "Control_L", "shift": "Shift_L", "alt": "Alt_L", "meta": "Meta_L",
             "super": "Super_L", "win": "Super_L", "cmd": "Super_L", "altgr": "ISO_Level3_Shift"}
# the names a model reaches for, as X names the key
KEY_NAMES = {"enter": "Return", "return": "Return", "esc": "Escape", "escape": "Escape", "tab": "Tab",
             "backspace": "BackSpace", "delete": "Delete", "del": "Delete", "insert": "Insert", "ins": "Insert",
             "home": "Home", "end": "End", "pageup": "Prior", "pgup": "Prior", "pagedown": "Next", "pgdn": "Next",
             "up": "Up", "down": "Down", "left": "Left", "right": "Right", "space": "space", "menu": "Menu",
             "printscreen": "Print", "capslock": "Caps_Lock"}


class Unreachable(Exception):
    """The screen cannot be reached here: why."""


def _lib(name):
    path = ctypes.util.find_library(name)
    for candidate in ([path] if path else []) + ["lib%s.so.6" % name]:
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    raise Unreachable("lib%s is not on this machine (an X11 desktop has it: the package libxtst6 or libXtst)" % name)


def why_not(display=None):
    """Why no screen answers here — or None where one does."""
    try:
        Screen(display).close()
        return None
    except Unreachable as ex:
        return str(ex)


class Screen:
    """An X11 display, opened for the pointer, the keys and its picture."""

    def __init__(self, display=None):
        if os.name == "nt":
            raise Unreachable("(ui …) reaches an X11 screen; Windows' is not here yet")
        name = display or os.environ.get("DISPLAY")
        if not name:
            if os.environ.get("WAYLAND_DISPLAY"):
                raise Unreachable("this is a Wayland session: its compositor takes no input from another program — "
                                  "(ui …) reaches an X11 screen")
            raise Unreachable("no screen: the server has no DISPLAY (a desktop session's, or the tools file's ui display)")
        if not display and os.environ.get("XDG_SESSION_TYPE") == "wayland":
            raise Unreachable("this is a Wayland session: XWayland's screen holds only the X programs, and its "
                              "compositor takes no input from another program — (ui …) reaches an X11 screen")
        self.name = name
        x, t = self.x, self.t = _lib("X11"), _lib("Xtst")
        x.XOpenDisplay.restype = ctypes.c_void_p
        x.XOpenDisplay.argtypes = [ctypes.c_char_p]
        for f in ("XDefaultRootWindow",):
            getattr(x, f).restype = ctypes.c_ulong
            getattr(x, f).argtypes = [ctypes.c_void_p]
        for f in ("XDefaultScreen", "XFlush", "XSync", "XCloseDisplay"):
            getattr(x, f).argtypes = [ctypes.c_void_p] + ([ctypes.c_int] if f == "XSync" else [])
        x.XDisplayWidth.argtypes = x.XDisplayHeight.argtypes = [ctypes.c_void_p, ctypes.c_int]
        x.XQueryPointer.argtypes = [ctypes.c_void_p, ctypes.c_ulong] + [ctypes.POINTER(ctypes.c_ulong)] * 2 + \
            [ctypes.POINTER(ctypes.c_int)] * 4 + [ctypes.POINTER(ctypes.c_uint)]
        x.XStringToKeysym.restype = ctypes.c_ulong
        x.XStringToKeysym.argtypes = [ctypes.c_char_p]
        x.XKeysymToKeycode.restype = ctypes.c_ubyte
        x.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        x.XDisplayKeycodes.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        x.XGetKeyboardMapping.restype = ctypes.POINTER(ctypes.c_ulong)
        x.XGetKeyboardMapping.argtypes = [ctypes.c_void_p, ctypes.c_ubyte, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        x.XChangeKeyboardMapping.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                             ctypes.POINTER(ctypes.c_ulong), ctypes.c_int]
        x.XFree.argtypes = [ctypes.c_void_p]
        t.XTestFakeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
        t.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
        t.XTestFakeKeyEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
        self.d = x.XOpenDisplay(name.encode())
        if not self.d:
            raise Unreachable("the display %s does not answer (its X server, or the right to use it — XAUTHORITY)" % name)
        self.root = x.XDefaultRootWindow(self.d)
        self.screen = x.XDefaultScreen(self.d)

    def close(self):
        if getattr(self, "d", None):
            self.x.XCloseDisplay(self.d)
            self.d = None

    # ── where ──────────────────────────────────────────────────────────────────────────────────────────────────
    def size(self):
        return self.x.XDisplayWidth(self.d, self.screen), self.x.XDisplayHeight(self.d, self.screen)

    def pointer(self):
        r, c = ctypes.c_ulong(), ctypes.c_ulong()
        rx, ry, wx, wy, mask = ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_uint()
        self.x.XQueryPointer(self.d, self.root, ctypes.byref(r), ctypes.byref(c), ctypes.byref(rx), ctypes.byref(ry),
                             ctypes.byref(wx), ctypes.byref(wy), ctypes.byref(mask))
        return rx.value, ry.value

    def grab(self):
        from PIL import ImageGrab
        return ImageGrab.grab(xdisplay=self.name)

    # ── the pointer ────────────────────────────────────────────────────────────────────────────────────────────
    def move(self, x, y):
        w, h = self.size()
        x, y = max(0, min(int(x), w - 1)), max(0, min(int(y), h - 1))
        self.t.XTestFakeMotionEvent(self.d, -1, x, y, 0)
        self.x.XSync(self.d, 0)
        return x, y

    def click(self, button=1, n=1):
        for i in range(n):
            if i:
                time.sleep(CLICK_DELAY)
            self.t.XTestFakeButtonEvent(self.d, button, 1, 0)
            self.t.XTestFakeButtonEvent(self.d, button, 0, 0)
            self.x.XSync(self.d, 0)

    def scroll(self, n):
        for _ in range(abs(int(n))):
            self.click(5 if n > 0 else 4)
            time.sleep(KEY_DELAY)

    # ── the keys ───────────────────────────────────────────────────────────────────────────────────────────────
    def _keysym(self, name):
        s = self.x.XStringToKeysym(name.encode())
        if not s:
            raise ValueError("no key is named %r" % name)
        return s

    def _press(self, code, down):
        self.t.XTestFakeKeyEvent(self.d, code, 1 if down else 0, 0)
        self.x.XSync(self.d, 0)
        time.sleep(KEY_DELAY)

    def _mapping(self):
        lo, hi = ctypes.c_int(), ctypes.c_int()
        self.x.XDisplayKeycodes(self.d, ctypes.byref(lo), ctypes.byref(hi))
        per = ctypes.c_int()
        m = self.x.XGetKeyboardMapping(self.d, lo.value, hi.value - lo.value + 1, ctypes.byref(per))
        table = {lo.value + i: [m[i * per.value + j] for j in range(per.value)] for i in range(hi.value - lo.value + 1)}
        self.x.XFree(m)
        return table, per.value

    def _code_for(self, keysym, table):
        """The keycode and shift level that type `keysym` in the keyboard's own map, or None."""
        for code, syms in table.items():
            for level in (0, 1):
                if level < len(syms) and syms[level] == keysym:
                    return code, level
        return None

    def type(self, text):
        """`text` typed a character at a time: a character the keyboard's map has, as the keyboard types it (Shift for
        its second level); one it lacks, bound for the moment to a keycode the map leaves empty, and given back."""
        table, per = self._mapping()
        shift = self.x.XKeysymToKeycode(self.d, self._keysym("Shift_L"))
        spare = next((c for c, syms in sorted(table.items(), reverse=True) if not any(syms)), None)
        for ch in text:
            sym = {"\n": self._keysym("Return"), "\t": self._keysym("Tab")}.get(ch)
            if sym is None:
                o = ord(ch)
                sym = o if 0x20 <= o <= 0x7e or 0xa0 <= o <= 0xff else 0x01000000 + o
            found = self._code_for(sym, table)
            if found is not None:
                code, level = found
                if level:
                    self._press(shift, True)
                self._press(code, True)
                self._press(code, False)
                if level:
                    self._press(shift, False)
                continue
            if spare is None:
                raise ValueError("the keyboard's map has no room for %r" % ch)
            syms = (ctypes.c_ulong * per)(*([sym, sym] + [0] * (per - 2)))
            self.x.XChangeKeyboardMapping(self.d, spare, per, syms, 1)
            self.x.XSync(self.d, 0)
            time.sleep(KEY_DELAY)
            self._press(spare, True)
            self._press(spare, False)
            self.x.XChangeKeyboardMapping(self.d, spare, per, (ctypes.c_ulong * per)(*([0] * per)), 1)
            self.x.XSync(self.d, 0)

    def key(self, chord):
        """A chord, "ctrl+shift+t" or "Return": its modifiers held in order, its key pressed and released, the modifiers
        let go in reverse."""
        parts = [p.strip() for p in chord.replace(" ", "").split("+") if p.strip()] or ([chord] if chord == "+" else [])
        if not parts:
            raise ValueError("a key's name, or a chord as \"ctrl+s\"")
        *mods, last = parts
        codes = []
        for m in mods:
            name = MODIFIERS.get(m.lower())
            if name is None:
                raise ValueError("%r is not a modifier: ctrl, shift, alt, super" % m)
            codes.append(self.x.XKeysymToKeycode(self.d, self._keysym(name)))
        name = KEY_NAMES.get(last.lower(), last)
        if len(name) == 1:                            # a character: its keysym, as typing finds it
            o = ord(name)
            sym = o if 0x20 <= o <= 0x7e or 0xa0 <= o <= 0xff else 0x01000000 + o
        else:
            sym = self._keysym(name if not name[:1] == "f" or not name[1:].isdigit() else name.upper())
        code = self.x.XKeysymToKeycode(self.d, sym)
        if not code:
            raise ValueError("the keyboard has no key %r" % last)
        for c in codes:
            self._press(c, True)
        self._press(code, True)
        self._press(code, False)
        for c in reversed(codes):
            self._press(c, False)
