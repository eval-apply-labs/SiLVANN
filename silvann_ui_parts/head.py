"""The mark, and the page's head: its styles and its scripts."""
from nicegui import ui


# ⭐ THE MARK: silicon's square in the periodic table — its number, 14, and "Si" — the browser tab's icon (`favicon`)
# and the drawer's first letters, "LVANN" following it there. One drawing for both, white on the blue the drawer's
# active row uses, so it reads on a light and a dark page alike.
SILVANN_TILE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-label="Si">'
    '<rect x="1" y="1" width="62" height="62" rx="7" fill="#1d4ed8"/>'
    '<text x="7" y="17" fill="#fff" fill-opacity="0.85" font-size="13" font-weight="600"'
    ' font-family="system-ui,-apple-system,Segoe UI,Roboto,sans-serif">14</text>'
    '<text x="32" y="49" fill="#fff" font-size="33" font-weight="800" text-anchor="middle" letter-spacing="-1"'
    ' font-family="system-ui,-apple-system,Segoe UI,Roboto,sans-serif">Si</text>'
    '</svg>')


# The page's head: its layout's styles, the chat's looks (NN-102: the thinking and the tool's sections wrapped, the
# dark colours, the copy button on code) and the button's script.
PAGE_HEAD = """
<style>
    /* Keep Quasar's natural q-page-container left-padding so the (mini) drawer pushes the
       content rather than overlapping it. Only zero top-padding (no header here). */
    .q-page-container { padding-top: 0 !important; }
    .q-page {
        padding: 0 !important;
        /* dvh = dynamic viewport height — tracks the visible area on mobile as the address
           bar shows/hides. Falls back to vh on older browsers via the duplicate decl. */
        min-height: 100vh;
        min-height: 100dvh;
        /* ⛳ AND EXACTLY THAT TALL: a page only AT LEAST the viewport leaves its flex children an indefinite
           height, and the chat's scroll area — which needs a definite one — then sizes its track to one
           height and its content to another (the scrollbar spanning the view, the messages only the top). */
        height: 100vh;
        height: 100dvh;
        display: flex;
        flex-direction: column;
        overflow-x: hidden;          /* never scroll the page sideways — overflow goes vertical */
    }
    .nicegui-content {
        padding: 0 !important;
        flex: 1 1 0;
        min-height: 0;
        display: flex;
        flex-direction: column;
        overflow-x: hidden;          /* clamp horizontal; the views stack + scroll vertically */
        overflow-y: auto;
    }
    /* The main views: full width, never wider than the area; vertical scroll inside. min-width:0
       lets flex content shrink instead of forcing a sideways scrollbar. */
    .silvann-main {
        flex: 1 1 0; min-height: 0; min-width: 0; max-width: 100%;
        display: flex; flex-direction: column;
        overflow-x: hidden;
    }
    /* The conversation is as wide as its view: a line longer than a bubble — a tool's value, a model's code —
       scrolls sideways inside its own code block. A bubble's text is a flex item, which would otherwise grow to
       its longest line and carry the whole conversation, the user's messages with it, off to the right. */
    .silvann-chat .q-scrollarea__content { width: 100%; }
    .silvann-chat .q-message-container > div { min-width: 0; }
    .silvann-chat .q-message-text pre { overflow-x: auto; }

    /* Menu rows in the drawer. In mini mode (60px wide drawer) only the leading icon shows;
       when the drawer expands on hover (overflow-x: visible), labels/chevrons reappear. */
    .silvann-menu-item {
        cursor: pointer; padding: 0.5rem 0.75rem; border-radius: 0.25rem;
        display: flex; align-items: center; gap: 0.5rem;
        user-select: none; white-space: nowrap;
    }
    .silvann-menu-item:hover { background: rgba(0,0,0,0.05); }
    .silvann-menu-active { background: rgba(59, 130, 246, 0.12); color: rgb(37, 99, 235); font-weight: 600; }

    /* The collapsed mini-drawer is 60px wide. Hide everything past the leading icon so
       labels don't smear into the next item — using `display: none` (not opacity) so the
       hidden elements don't reserve width and the icon really centers. */
    .q-drawer--mini .silvann-menu-item > :not(.q-icon:first-child) {
        display: none;
    }
    .q-drawer--mini .silvann-menu-item { justify-content: center; padding: 0.5rem 0; }
    .q-drawer--mini .silvann-conv-item,
    .q-drawer--mini .silvann-brand-label,
    .q-drawer--mini .silvann-convs-col { display: none; }
    .q-drawer { transition: width 0.18s ease; }
    /* The drawer is a fixed-height (~100vh) flex column. Without flex-wrap:nowrap its content
       WRAPS into side-by-side columns (logo header | Chat/Options/Help | endpoint footer) once
       it's taller than the drawer — that's the "3 columns / horizontal scroll" bug. Force a
       single non-wrapping vertical stack and let it scroll vertically instead. */
    .nicegui-drawer { flex-wrap: nowrap !important; }
    .nicegui-drawer .q-drawer__content {
        flex-wrap: nowrap !important;
        overflow-y: auto;
        overflow-x: hidden !important;
    }

    .silvann-conv-item {
        cursor: pointer; padding: 0.35rem 0.5rem; border-radius: 0.25rem;
        display: flex; align-items: center; gap: 0.4rem;
        user-select: none;
    }
    .silvann-conv-item:hover { background: rgba(0,0,0,0.05); }
    .silvann-conv-active { background: rgba(59, 130, 246, 0.15); }

    /* The mark: silicon's square (SILVANN_TILE) and, with the drawer open, "LVANN" beside it in the wordmark's bold. */
    .silvann-logo {
        font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
        font-weight: 800;
        font-size: 1.25rem;
        letter-spacing: -0.02em;
        display: inline-flex;
        align-items: center;
        gap: 0.2rem;
    }
    .silvann-tile { width: 2rem; height: 2rem; flex: none; display: block; }
    /* The closed drawer is 60px wide: the square and the server's dot side by side in it. */
    .q-drawer--mini .silvann-brand-row { padding-left: 4px; padding-right: 4px; justify-content: center; }

    /* ⭐ THE THINKING AND THE TOOL'S SECTIONS WRAP: their code — the tool's calls and answers, a thought the markdown
       reads as code — goes on to the next line inside its section, where a block that kept its lines widened the
       section past its bubble and was cut off; an answer's own code keeps its lines and scrolls sideways. Prose wraps
       already (Quasar's bubble breaks long words). */
    .silvann-chat .silvann-part pre { white-space: pre-wrap; overflow-wrap: anywhere; overflow-x: visible; }

    /* A code block's copy button (the script below), in the block's top corner, faint until the block is pointed at. */
    .silvann-copy-host { position: relative; }
    .silvann-copy {
        position: absolute; top: 4px; right: 4px; z-index: 1;
        font: 11px ui-sans-serif, system-ui, sans-serif; padding: 1px 6px; border-radius: 4px; cursor: pointer;
        border: 1px solid rgba(0,0,0,0.2); background: rgba(255,255,255,0.85); color: #333;
        opacity: 0.3; transition: opacity 0.15s;
    }
    .silvann-copy-host:hover .silvann-copy, .silvann-copy:focus { opacity: 1; }

    /* ⭐ DARK: Quasar darkens its own components; what this page colours itself follows here — and the chat's bubbles,
       light by Quasar's default, turn dark under the dark colours NiceGUI gives code. */
    .body--dark .silvann-menu-item:hover, .body--dark .silvann-conv-item:hover { background: rgba(255,255,255,0.08); }
    .body--dark .silvann-menu-active { background: rgba(96,165,250,0.18); color: rgb(147,197,253); }
    .body--dark .silvann-conv-active { background: rgba(96,165,250,0.2); }
    .body--dark .q-message-text--received { color: #24302a; }
    .body--dark .q-message-text--sent { color: #30353f; }
    .body--dark .q-message-text-content--received, .body--dark .q-message-text-content--sent { color: #e8e8e8; }
    .body--dark .bg-amber-50 { background: rgba(245,158,11,0.14) !important; }
    .body--dark .bg-slate-100 { background: #1e293b !important; }
    .body--dark .text-slate-800 { color: #e2e8f0 !important; }
    .body--dark .text-amber-700 { color: #fbbf24 !important; }
    .body--dark .text-green-700 { color: #4ade80 !important; }
    .body--dark .text-red-700 { color: #f87171 !important; }
    .body--dark .silvann-copy { background: rgba(30,30,30,0.85); color: #ddd; border-color: rgba(255,255,255,0.25); }
</style>
<script>
/* ⭐ A COPY BUTTON ON EVERY CODE BLOCK OF THE CONVERSATION — put on each block as it is drawn, and drawn again as a
   streaming answer's are; it copies the block's text. The clipboard's own call needs a secure page (https or localhost),
   so a UI reached over the LAN by http copies through a hidden text area. */
(() => {
  const copy = (text) => {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    const area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.cssText = 'position:fixed;top:0;left:0;opacity:0';
    document.body.appendChild(area);
    area.select();
    try { return document.execCommand('copy') ? Promise.resolve() : Promise.reject(); } finally { area.remove(); }
  };
  const equip = (pre) => {
    const host = pre.parentElement && pre.parentElement.classList.contains('codehilite') ? pre.parentElement : pre;
    if (host.querySelector(':scope > .silvann-copy')) return;
    host.classList.add('silvann-copy-host');
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'silvann-copy';
    button.textContent = 'copy';
    button.addEventListener('click', (e) => {
      e.stopPropagation();
      const code = pre.querySelector('code');
      copy((code || pre).textContent).then(() => { button.textContent = 'copied'; }, () => { button.textContent = 'not copied'; });
      setTimeout(() => { button.textContent = 'copy'; }, 1500);
    });
    host.appendChild(button);
  };
  new MutationObserver(() => document.querySelectorAll('.silvann-chat pre').forEach(equip))
    .observe(document.documentElement, { childList: true, subtree: true });
})();
/* ⭐ THE CHAT FOLLOWS A STREAMING ANSWER ONLY WHILE ITS VIEW IS AT THE BOTTOM (⚖ 2026-11-13 "if you are not seeing the
   bottom it should stay where the cursor is"): a scroll up leaves the bottom, and a scroll that reaches it follows
   again. Only a scroll up stops the following: a scroll's event comes after the answer has grown below it, so the
   distance to the bottom it finds says nothing of where the reader is — the follow's own scroll down found the answer
   grown past it and stopped following. While it follows, the view goes to the bottom each time the messages grow,
   after the browser has drawn them. `force`: to the bottom whatever the view, as a message sent or a conversation
   opened goes, and following from there. */
/* The person at the page — a key, the pointer, a wheel, a touch — said to the UI at most every 20 seconds: a use of
   their sign-in, which is over ten minutes after the last (▶ page.py's `_active`). A page left alone says nothing. */
(() => {
  let last = 0;
  const here = () => {
    const now = Date.now();
    if (now - last < 20000 || typeof emitEvent !== 'function') return;
    last = now;
    emitEvent('silvann_active');
  };
  for (const kind of ['keydown', 'pointerdown', 'pointermove', 'wheel', 'touchstart'])
    window.addEventListener(kind, here, { passive: true, capture: true });
})();
window.silvannChatFollow = (id, force) => {
  const area = getHtmlElement(id);
  const view = area && area.querySelector('.q-scrollarea__container');
  if (!view) return;
  if (!view.silvannWatched) {
    view.silvannWatched = true;
    view.silvannAtBottom = true;
    let last = view.scrollTop;
    view.addEventListener('scroll', () => {
      const top = view.scrollTop, gap = view.scrollHeight - view.clientHeight - top;
      if (top < last - 1) view.silvannAtBottom = gap <= 48;      // up: following only if still at the bottom
      else if (gap <= 48) view.silvannAtBottom = true;           // down to the bottom: following again
      last = top;
    }, { passive: true });
    // a wheel turned up leaves the bottom at once, before a growth's follow can take the view back down
    view.addEventListener('wheel', (e) => { if (e.deltaY < 0 && view.scrollTop > 0) view.silvannAtBottom = false; },
                          { passive: true });
    const content = view.querySelector('.q-scrollarea__content') || view.firstElementChild;
    if (content && window.ResizeObserver) {
      new ResizeObserver(() => { if (view.silvannAtBottom) view.scrollTop = view.scrollHeight; }).observe(content);
    }
  }
  if (force) view.silvannAtBottom = true;
  if (view.silvannAtBottom) view.scrollTop = view.scrollHeight;
};
</script>
"""
# ⛳ IN EVERY PAGE'S FIRST HTML: a head a page adds once it has awaited its client's connection is inserted into the page
#   already drawn, where its styles apply but its scripts do not run — the chat's follow, the code blocks' copy button
ui.add_head_html(PAGE_HEAD, shared=True)
