"""
Retro CRT terminal styling for the DishaNifty dashboard.

Everything visual that is not a chart lives here: the phosphor-green stylesheet,
the ASCII art, the box-drawing helpers, and Disha — the mascot who keeps the user
company while a model trains.

Two different font stacks are used on purpose. The UI chrome gets a retro webfont
for character; ASCII art gets a plain system monospace stack, because the webfont
may not ship box-drawing glyphs and a fallback glyph of a different width would
tear the art apart.
"""

from __future__ import annotations

import html
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

BG = "#050505"
PANEL = "#0c0c0c"
# Panel fills are translucent so the halftone dots stay faintly visible through
# every card instead of stopping at the gutters.
PANEL_RGBA = "rgba(12,12,12,.78)"
GREEN = "#39ff6a"
GREEN_DIM = "#1f8a3f"
AMBER = "#ffb627"
RED = "#ff5555"
CYAN = "#3ae0d8"
MUTED = "#5f8f6c"
GRID = "#163a22"

UP_C, DOWN_C, FLAT_C = GREEN, RED, AMBER
DIRECTION_COLORS = {"UP": UP_C, "NEUTRAL": FLAT_C, "DOWN": DOWN_C}
ACTION_COLORS = {"BUY": UP_C, "HOLD": FLAT_C, "SELL": DOWN_C}
MODEL_COLORS = {"Random Forest": GREEN, "XGBoost": AMBER, "LSTM": CYAN}
PALETTE = [GREEN, AMBER, CYAN, "#b388ff", MUTED, RED]

FONT_ART = "'Cascadia Mono','Consolas','Menlo','DejaVu Sans Mono','Courier New',monospace"
FONT_UI = f"'Share Tech Mono',{FONT_ART}"


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

:root {{
  --bg: {BG}; --panel: {PANEL_RGBA}; --green: {GREEN}; --dim: {GREEN_DIM};
  --amber: {AMBER}; --red: {RED}; --cyan: {CYAN}; --muted: {MUTED}; --grid: {GRID};
}}

#MainMenu, footer, [data-testid="stStatusWidget"] {{visibility: hidden;}}
[data-testid="stHeader"] {{background: transparent; height: 0;}}
[data-testid="stDecoration"] {{display: none;}}

html, body, [data-testid="stAppViewContainer"] {{
  background-color: var(--bg);
  color: var(--green);
  font-family: {FONT_UI};
}}

/* Black-and-white halftone. Two identical dot grids offset by half a tile give
   the 45-degree lattice of a print screen; the coarse layer sits under the fine
   one so the texture does not read as a plain square grid. Fixed attachment
   keeps the dots still while the page scrolls. */
[data-testid="stAppViewContainer"] {{
  background-image:
    radial-gradient(circle at center, rgba(255,255,255,.26) 1.1px, transparent 1.6px),
    radial-gradient(circle at center, rgba(255,255,255,.26) 1.1px, transparent 1.6px),
    radial-gradient(circle at center, rgba(255,255,255,.10) 2.2px, transparent 2.8px),
    radial-gradient(circle at center, rgba(255,255,255,.10) 2.2px, transparent 2.8px);
  background-size: 9px 9px, 9px 9px, 27px 27px, 27px 27px;
  background-position: 0 0, 4.5px 4.5px, 0 0, 13.5px 13.5px;
  background-attachment: fixed;
}}

/* Scanlines + vignette. pointer-events:none so it never eats a click. */
[data-testid="stAppViewContainer"]::before {{
  content: ""; position: fixed; inset: 0; z-index: 9999; pointer-events: none;
  background:
    repeating-linear-gradient(0deg, rgba(0,0,0,.14) 0 1px, transparent 1px 3px),
    radial-gradient(ellipse at center, transparent 68%, rgba(0,0,0,.45) 100%);
}}

.block-container {{padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px;}}

h1, h2, h3, h4 {{
  font-family: {FONT_UI} !important;
  color: var(--green) !important;
  text-transform: uppercase; letter-spacing: .13em; font-weight: 400 !important;
  text-shadow: 0 0 9px rgba(57,255,106,.45);
}}
h1 {{font-size: 24px !important;}}
h2 {{font-size: 19px !important;}}
h3 {{font-size: 16px !important;}}

p, span, label, li, div, .stMarkdown {{font-family: {FONT_UI};}}

/* --- ASCII art blocks -----------------------------------------------------
   Rendered as <div>, not <pre>: Streamlit rewrites a raw <pre> into its own
   element and drops the class, which would collapse every block to one line. */
div.crt {{
  font-family: {FONT_ART} !important;
  background: var(--panel);
  color: var(--green);
  border: 1px solid var(--grid);
  border-radius: 2px;
  padding: 12px 14px;
  margin: 0 0 12px 0;
  font-size: 13px; line-height: 1.32;
  white-space: pre; overflow-x: auto;
  text-shadow: 0 0 7px rgba(57,255,106,.35);
}}
div.crt.bare {{background: transparent; border: none; padding: 0; overflow-x: auto;}}
div.crt.amber {{color: var(--amber); text-shadow: 0 0 7px rgba(255,182,39,.35);}}
div.crt.red   {{color: var(--red);   text-shadow: 0 0 7px rgba(255,85,85,.35);}}
div.crt.cyan  {{color: var(--cyan);  text-shadow: 0 0 7px rgba(58,224,216,.35);}}
div.crt.dim   {{color: var(--muted); text-shadow: none;}}

.logo {{
  font-family: {FONT_ART};
  color: var(--green); font-size: 13px; line-height: 1.12;
  white-space: pre; margin: 0;
  text-shadow: 0 0 12px rgba(57,255,106,.55);
}}
.tag {{color: var(--muted); font-size: 12px; letter-spacing: .26em; text-transform: uppercase;}}
.eyebrow {{
  color: var(--muted); font-size: 11px; letter-spacing: .2em;
  text-transform: uppercase;
}}
.blink {{animation: blink 1.1s step-end infinite;}}
@keyframes blink {{50% {{opacity: 0;}}}}

/* --- metrics -------------------------------------------------------------- */
[data-testid="stMetric"] {{
  background: var(--panel); border: 1px solid var(--grid);
  padding: 10px 14px; border-radius: 2px;
}}
[data-testid="stMetricLabel"] p {{
  font-size: 10px !important; letter-spacing: .17em; text-transform: uppercase;
  color: var(--muted) !important;
}}
[data-testid="stMetricValue"] {{
  font-family: {FONT_ART} !important; font-size: 22px !important;
  color: var(--green) !important; text-shadow: 0 0 8px rgba(57,255,106,.4);
}}
[data-testid="stMetricDelta"] {{font-family: {FONT_ART} !important; font-size: 12px !important;}}

/* Last close carries the session's direction in the number itself, not just in
   the delta badge, which is all Streamlit colours on its own. Scoped by the
   container key app.py sets. Up days need no rule -- green is already the
   default above. The glow is overridden too, or a red number keeps a green halo. */
.st-key-last-close-down [data-testid="stMetricValue"] {{
  color: var(--red) !important; text-shadow: 0 0 8px rgba(255,85,85,.4);
}}
.st-key-last-close-flat [data-testid="stMetricValue"] {{
  color: var(--amber) !important; text-shadow: 0 0 8px rgba(255,182,39,.4);
}}

/* --- sidebar -------------------------------------------------------------- */
[data-testid="stSidebar"] {{
  background: rgba(3,3,3,.72); border-right: 1px solid var(--grid);
}}
[data-testid="stSidebar"] * {{color: var(--green);}}
[data-testid="stSidebar"] label {{
  font-size: 11px !important; letter-spacing: .15em; text-transform: uppercase;
  color: var(--muted) !important;
}}

/* --- inputs --------------------------------------------------------------- */
[data-baseweb="select"] > div, [data-baseweb="input"] > div,
[data-baseweb="base-input"], .stTextInput input, .stNumberInput input {{
  background: var(--panel) !important;
  border: 1px solid var(--dim) !important;
  border-radius: 2px !important;
  color: var(--green) !important;
  font-family: {FONT_ART} !important;
}}
[data-baseweb="select"] svg {{color: var(--green) !important; fill: var(--green) !important;}}
[data-baseweb="popover"] li, [data-baseweb="menu"] li {{
  background: var(--panel) !important; color: var(--green) !important;
  font-family: {FONT_ART} !important; font-size: 13px !important;
}}
[data-baseweb="popover"] li:hover, [data-baseweb="menu"] li:hover {{
  background: #16291a !important; color: #c9ffd8 !important;
}}
[data-baseweb="menu"] {{border: 1px solid var(--dim) !important;}}

.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
  background: var(--panel); color: var(--green);
  border: 1px solid var(--dim); border-radius: 2px;
  font-family: {FONT_UI}; letter-spacing: .13em; text-transform: uppercase;
  font-size: 12px; transition: all .12s;
}}
.stButton > button:hover, .stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {{
  background: var(--green); color: #04140a; border-color: var(--green);
  box-shadow: 0 0 14px rgba(57,255,106,.5);
}}

/* --- top nav button -------------------------------------------------------
   Sits beside the logo and swaps the whole view, so it must not read as just
   another control. Amber against the green chrome, with a steady glow. */
.st-key-nav_screener .stButton > button {{
  width: 100%; padding: 11px 6px;
  background: var(--panel); color: var(--amber);
  border: 1px solid var(--amber);
  font-size: 12px; letter-spacing: .12em; white-space: nowrap;
  box-shadow: 0 0 14px rgba(255,182,39,.45);
}}
.st-key-nav_screener .stButton > button:hover {{
  background: var(--amber); color: #140d02; border-color: var(--amber);
  box-shadow: 0 0 24px rgba(255,182,39,.8);
}}

.st-key-nav_back .stButton > button {{
  width: 100%; padding: 11px 6px;
  background: var(--panel); color: var(--green);
  border: 1px solid var(--green);
  font-size: 12px; letter-spacing: .12em; white-space: nowrap;
  box-shadow: 0 0 10px rgba(57,255,106,.30);
}}
.st-key-nav_back .stButton > button:hover {{
  background: var(--green); color: #04140a;
  box-shadow: 0 0 24px rgba(57,255,106,.8);
}}

/* --- radio pills ---------------------------------------------------------- */
.stRadio [role="radiogroup"] {{gap: 6px;}}
.stRadio label {{
  background: var(--panel); border: 1px solid var(--grid);
  padding: 2px 10px; border-radius: 2px; font-size: 12px !important;
  letter-spacing: .08em; color: var(--green) !important;
}}
.stRadio input:checked + div {{color: var(--green);}}

/* --- tabs ----------------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {{
  gap: 4px; background: transparent; border-bottom: 1px solid var(--grid);
}}
.stTabs [data-baseweb="tab"] {{
  background: var(--panel); border: 1px solid var(--grid); border-bottom: none;
  border-radius: 2px 2px 0 0; padding: 6px 18px;
  font-family: {FONT_UI}; letter-spacing: .14em; text-transform: uppercase;
  font-size: 12px; color: var(--muted);
}}
.stTabs [aria-selected="true"] {{
  background: var(--green) !important; color: #04140a !important;
  border-color: var(--green) !important;
}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{display: none;}}

/* --- dataframe ------------------------------------------------------------ */
[data-testid="stDataFrame"], [data-testid="stTable"] {{
  border: 1px solid var(--grid); border-radius: 2px;
}}
[data-testid="stDataFrame"] * {{font-family: {FONT_ART} !important; font-size: 12px !important;}}

/* --- expander / alerts ---------------------------------------------------- */
[data-testid="stExpander"] {{
  background: var(--panel); border: 1px solid var(--grid); border-radius: 2px;
}}
[data-testid="stExpander"] summary {{
  font-family: {FONT_UI}; letter-spacing: .12em; text-transform: uppercase;
  font-size: 12px; color: var(--green);
}}
[data-testid="stAlert"] {{
  background: var(--panel); border: 1px solid var(--dim); border-radius: 2px;
  color: var(--green); font-family: {FONT_ART};
}}
[data-testid="stAlert"] * {{color: var(--green) !important;}}

hr {{border-color: var(--grid) !important;}}
[data-testid="stCaptionContainer"] p, .stCaption, small {{
  color: var(--muted) !important; font-size: 11.5px !important; letter-spacing: .03em;
}}
::-webkit-scrollbar {{width: 9px; height: 9px;}}
::-webkit-scrollbar-track {{background: var(--bg);}}
::-webkit-scrollbar-thumb {{background: var(--dim); border-radius: 0;}}

/* --- panels --------------------------------------------------------------- */
.panel {{
  background: var(--panel); border: 1px solid var(--grid);
  border-radius: 2px; padding: 16px 18px;
}}
.panel.glow {{box-shadow: inset 0 0 40px rgba(57,255,106,.06);}}
.big-call {{
  font-family: {FONT_ART}; font-size: 54px; line-height: 1.05; font-weight: 700;
  letter-spacing: .06em;
}}
.loading-wrap {{
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  min-height: 62vh; gap: 4px;
}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# ASCII art
# ---------------------------------------------------------------------------

LOGO = r"""
 ____  ___ ____  _   _    _      _   _ ___ _____ _______   __
|  _ \|_ _/ ___|| | | |  / \    | \ | |_ _|  ___|_   _\ \ / /
| | | || |\___ \| |_| | / _ \   |  \| || || |_    | |  \ V /
| |_| || | ___) |  _  |/ ___ \  | |\  || ||  _|   | |   | |
|____/|___|____/|_| |_/_/   \_\ |_| \_|___|_|     |_|   |_|
"""


def _rule(width: int, left: str, fill: str, right: str, title: str = "") -> str:
    if title:
        label = f" {title} "
        pad = max(width - 2 - len(label), 0)
        lead = 2
        return left + fill * lead + label + fill * (pad - lead) + right
    return left + fill * (width - 2) + right


def box(lines: list[str], title: str = "", width: int = 0, pad: int = 1) -> str:
    """Wrap text in a single-line box-drawing frame. Lines are clipped to width."""
    body = [str(line) for line in lines]
    inner = max([len(line) for line in body] + [len(title) + 6, 10])
    inner = max(inner, width - 2 - pad * 2) if width else inner
    total = inner + 2 + pad * 2
    spacer = " " * pad

    out = [_rule(total, "┌", "─", "┐", title)]
    for line in body:
        out.append("│" + spacer + line[:inner].ljust(inner) + spacer + "│")
    out.append(_rule(total, "└", "─", "┘"))
    return "\n".join(out)


# No space as the first rung: the series minimum should still draw something.
_BLOCKS = "▁▂▃▄▅▆▇█"


def ascii_bar(value: float, width: int = 22, fill: str = "█", empty: str = "·") -> str:
    """A proportional bar. value is 0..1 and is clamped."""
    value = max(0.0, min(1.0, float(value)))
    n = int(round(value * width))
    return fill * n + empty * (width - n)


def labelled_bars(rows: list[tuple[str, float]], width: int = 22,
                  marker: float | None = None) -> str:
    """
    One `LABEL [██████····] 42.1%` line per row, plus an optional threshold ruler
    underneath showing where `marker` falls on the same scale.
    """
    pad = max(len(label) for label, _ in rows)
    out = [f"{label.ljust(pad)} │{ascii_bar(v, width)}│ {v:6.1%}" for label, v in rows]
    if marker is not None:
        at = int(round(max(0.0, min(1.0, marker)) * width))
        ruler = (" " * max(at - 1, 0)) + "▲"
        out.append(" " * pad + " │" + ruler.ljust(width) + "│")
        out.append(" " * pad + "  " + f"act threshold {marker:.0%}")
    return "\n".join(out)


def sparkline(values, width: int = 60) -> str:
    """Compress a numeric series into one line of block characters."""
    vals = [float(v) for v in values if v == v]
    if not vals:
        return ""
    if len(vals) > width:
        step = len(vals) / width
        vals = [vals[min(int(i * step), len(vals) - 1)] for i in range(width)]
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 0:
        return _BLOCKS[4] * len(vals)
    return "".join(_BLOCKS[int((v - lo) / span * (len(_BLOCKS) - 1))] for v in vals)


def crt(text: str, tone: str = "", bare: bool = False) -> None:
    """Render pre-formatted ASCII inside a CRT-styled block."""
    classes = " ".join(filter(None, ["crt", tone, "bare" if bare else ""]))
    st.markdown(f'<div class="{classes}">{html.escape(text)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Disha — the loading mascot
# ---------------------------------------------------------------------------

# Interior width of the head is fixed at 15 characters and the body at 19, so
# every frame substitutes equal-length tokens and the art can never shear.
_DISHA_BODY = r"""            \  |  /
             \ | /
              \|/
       .---------------.
       |  {eye}     {eye}  |
       |               |
       |    {mouth}    |
       '---------------'
         |           |
     .---'-----------'---.
     |   D  I  S  H  A   |
     '-------------------'
         |  |     |  |
         '--'     '--'"""

_FACES = [
    ("(o)", r"\_____/"),
    ("(^)", r" \___/ "),
    ("(-)", r"\__O__/"),
    ("(o)", r"\__o__/"),
    ("(*)", r"\_____/"),
    ("(^)", r"\_____/"),
]

_SPINNER = ["|", "/", "-", "\\"]

_CHATTER = [
    "crunching candles...",
    "teaching the trees...",
    "folding the walk-forward windows...",
    "arguing with the LSTM...",
    "counting ups and downs...",
    "bribing the gradient...",
    "tasting the volatility...",
    "sweeping up overfitting...",
]


def disha(frame: int = 0) -> str:
    eye, mouth = _FACES[frame % len(_FACES)]
    return _DISHA_BODY.format(eye=eye, mouth=mouth)


def _progress_bar(done: int, total: int, width: int = 34) -> str:
    frac = 0.0 if total <= 0 else min(done / total, 1.0)
    return f"[{ascii_bar(frac, width, '█', '░')}] {frac:5.1%}"


def loading_screen(title: str, frame: int, done: int = 0, total: int = 0,
                   stage: str = "", log_tail: list[str] | None = None) -> str:
    """The HTML for one animation frame of the Disha loading screen."""
    spin = _SPINNER[frame % len(_SPINNER)]
    chatter = _CHATTER[(frame // 6) % len(_CHATTER)]
    bar = _progress_bar(done, total) if total else f"[{'░' * 34}]  working"

    panel = [
        f"{spin} {title}",
        "",
        bar,
        "",
        f"disha: {stage or chatter}",
    ]
    art = disha(frame) + "\n\n" + box(panel, title="PLEASE HOLD", width=52)
    tail = "\n".join((log_tail or [])[-6:])

    out = [
        '<div class="loading-wrap">',
        f'<div class="crt bare" style="font-size:14px;line-height:1.3">{html.escape(art)}</div>',
    ]
    if tail:
        out.append(
            '<div class="crt dim bare" style="font-size:11.5px;max-width:640px">'
            f'{html.escape(tail)}</div>'
        )
    out.append('<div class="tag blink">do not close this tab</div>')
    out.append("</div>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Running a background job behind the loading screen
# ---------------------------------------------------------------------------

_NOISE = re.compile(
    r"oneDNN|absl::InitializeLog|TensorFlow GPU support|port\.cc|"
    r"computation_placer|cuda|cuFFT|cuDNN|^\s*$|WARNING:tensorflow",
    re.IGNORECASE,
)

# Progress markers printed by src.evaluate, one per completed stage.
_STAGE = re.compile(r"(Fold \d+:|FINAL TEST:|live model saved|^--- )")
_STAGES_PER_MODEL = 6  # 4 folds + final test + live refit


def log_lines(text: str) -> list[str]:
    """Subprocess output with TensorFlow's startup banner filtered out."""
    return [ln.rstrip() for ln in text.splitlines() if ln.strip() and not _NOISE.search(ln)]


def run_job(title: str, argv: list[str], log_path: Path, cwd: Path,
            total_stages: int = 0, poll: float = 0.4) -> tuple[int, str]:
    """
    Run `argv` as a subprocess while animating Disha, and return (rc, log text).

    Output is piped to a file rather than a pipe so progress can be polled
    without risking a deadlock on a full pipe buffer. `-u` on the child keeps
    its stdout unbuffered, so the file reflects progress as it happens.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
           "TF_CPP_MIN_LOG_LEVEL": "3"}

    slot = st.empty()
    slot.markdown(loading_screen(title, 0, 0, total_stages, "warming up..."),
                  unsafe_allow_html=True)

    with open(log_path, "w", encoding="utf-8") as sink:
        proc = subprocess.Popen(argv, cwd=str(cwd), stdout=sink,
                                stderr=subprocess.STDOUT, env=env, text=True)
        frame = 0
        while proc.poll() is None:
            try:
                raw = log_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                raw = ""
            lines = log_lines(raw)
            done = sum(1 for ln in lines if _STAGE.search(ln) and not ln.startswith("---"))
            stage = next((ln.strip() for ln in reversed(lines) if _STAGE.search(ln)), "")
            slot.markdown(
                loading_screen(title, frame, done, total_stages, stage, lines),
                unsafe_allow_html=True,
            )
            frame += 1
            time.sleep(poll)

    text = log_path.read_text(encoding="utf-8", errors="replace")
    slot.empty()
    return proc.returncode, text


def train_stages(model_keys: list[str]) -> int:
    return len(model_keys) * _STAGES_PER_MODEL


def python_cmd(module: str, *args: str) -> list[str]:
    return [sys.executable, "-u", "-m", module, *args]


# ---------------------------------------------------------------------------
# Plotly: make charts look like they came off the same CRT
# ---------------------------------------------------------------------------

def terminal_chart(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, showlegend=legend, colorway=PALETTE,
        margin=dict(l=6, r=6, t=10, b=6),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_ART, size=11.5, color=MUTED),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=11)),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=GREEN_DIM,
                        font=dict(family=FONT_ART, color=GREEN, size=11.5)),
    )
    axis = dict(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                linecolor=GREEN_DIM, tickcolor=GREEN_DIM, ticks="outside",
                tickfont=dict(color=MUTED, size=10.5), title_font=dict(color=MUTED, size=11))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

TEAM = [
    ("Arindam Pal", "BCA/40002/24"),
    ("Sujal Kumar", "BCA/40005/24"),
    ("Khushi Kumari", "BCA/40016/24"),
    ("Sneha Tiwari", "BCA/40020/24"),
]


def footer(disclaimer: str) -> None:
    width = 62
    rows = [
        "DISHA  ·  next-day direction for the NSE".center(width - 4),
        "",
        "developed as part of our semester project by".center(width - 4),
        "",
    ]
    pad = max(len(name) for name, _ in TEAM)
    for name, roll in TEAM:
        rows.append(f"{name.ljust(pad)}   {roll}".center(width - 4))
    rows += ["", ("-" * (width - 20)).center(width - 4), ""]
    rows += [line.center(width - 4) for line in _wrap(disclaimer, width - 6)]

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    crt(box(rows, title="CREDITS", width=width), tone="dim")


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines
