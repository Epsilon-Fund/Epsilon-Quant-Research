"""
Dashboards page — live signal + trade-logging view, one tab per strategy.
"""
import os
import sys

import streamlit as st

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# ── Path setup ────────────────────────────────────────────────────────────────
_PAGES_DIR = os.path.dirname(os.path.abspath(__file__))   # live_trading/pages/
_LT_DIR    = os.path.dirname(_PAGES_DIR)                   # live_trading/
if _LT_DIR not in sys.path:
    sys.path.insert(0, _LT_DIR)

from shared.styles import apply_styles
apply_styles()

# ── Page-level header (above strategy tabs) ───────────────────────────────────
st.markdown("""
<h1 style="font-size:35px;font-weight:700;letter-spacing:-0.01em;margin-bottom:10px">
  Epsilon Fund — Live Trading Dashboard
</h1>
""", unsafe_allow_html=True)

# ── Paths to strategy entry points ────────────────────────────────────────────
_MOMENTUM_APP = os.path.join(_LT_DIR, 'dashboards', 'momentum', 'streamlit_app.py')


def _exec_page(path: str, extra: dict = None) -> None:
    """
    Execute a standalone Streamlit page file inside the current render context.

    Patches applied for the duration of the exec:
      1. st.set_page_config → no-op  (already called by app.py)
      2. __file__ → target path  (so path setup inside exec'd file is correct)
    extra: optional dict merged into the exec namespace (e.g. _SUPPRESS_H1=True).
    """
    import streamlit as _st
    _orig_spc = _st.set_page_config
    _st.set_page_config = lambda *a, **kw: None
    try:
        source = open(path, encoding='utf-8').read()
        ns = {'__file__': path, '__builtins__': __builtins__}
        if extra:
            ns.update(extra)
        exec(source, ns)
    finally:
        _st.set_page_config = _orig_spc


# ── Strategy tabs (below the header) ─────────────────────────────────────────

tab_momentum, tab_statarb, tab_bb = st.tabs(
    ["Momentum", "Stat Arb", "BB Breakout"]
)

with tab_momentum:
    _exec_page(_MOMENTUM_APP, extra={'_SUPPRESS_H1': True})

with tab_statarb:
    st.info("Stat Arb — not yet configured")

with tab_bb:
    st.info("BB Breakout — not yet configured")
