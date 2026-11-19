#!/usr/bin/env python3
"""
paper4_figures_all_in_one.py -- every figure of Paper 4 from one file.

Figure 1 (study design), Figures 2 to 8 and the graphical abstract, drawn
from the result files the analysis scripts wrote. Nothing in this folder is
imported: the paths, the style and the code of every figure are below, so the
look of all figures can be changed here without touching the analysis. The
figure code is the code that drew the published figures; with the settings
as shipped, this file reproduces them.

WHAT TO EDIT is section 1, SETTINGS. Each setting says what it does, and the
most common ones can also be given on the command line.

Every run ends with a layout check. If a change makes two pieces of text
overlap, pushes a label onto another panel or off the canvas, the run names
the figure and the texts and exits with an error, so a broken figure cannot
pass unnoticed. The figures are still written, so you can look at them.

By default the figures go to ../Figures_styled/, so the paper's own Figures/
folder is never overwritten by an experiment. When a style is final, write it
to the paper with  --outdir ../Figures .

    python3 paper4_figures_all_in_one.py                         # everything
    python3 paper4_figures_all_in_one.py --only 1,5,8,ga         # some of them
    python3 paper4_figures_all_in_one.py --palette vibrant --labels "(A)"
    python3 paper4_figures_all_in_one.py --font-scale 1.1 --dpi 1200
    python3 paper4_figures_all_in_one.py --title-bold --check-only

READS   results_studyA_*/ and paper_final/derived/ (found by walking up from
        this file, or set SATSDOH_ROOT), ../constants_base.tex and
        ../constants_results.tex, conus_counties_5070.gpkg beside this file
NEEDS   numpy, pandas, matplotlib, geopandas, pyarrow
WRITES  Figure01_methodology ... Figure08_failure_geography, GraphicalAbstract
        as PNG (raster, DPI below) and PDF (vector)
"""
from __future__ import annotations

import argparse
import json
import re
import os
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.lines import Line2D                            # noqa: E402
from matplotlib.patches import Patch                           # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.text import Text                               # noqa: E402

# ===========================================================================
#  1. SETTINGS  (the part to edit)
# ===========================================================================
DPI = 600                  # PNG resolution in dots per inch. 600 is the
                           # minimum this file accepts; 1200 for a poster.
FORMATS = ("png", "pdf")   # PDF is vector and does not depend on DPI.

PALETTE = "paper"          # "paper"   = the published colours (Okabe-Ito)
                           # "vibrant" = the same meanings, more saturated
                           # or add your own entry to PALETTES below.

PANEL_LABEL = "{}"         # how the panel letter is printed; {} is the letter
                           #   "{}"   ->  A        "({})" ->  (A)
                           #   "{}."  ->  A.       "{})"  ->  A)
PANEL_LABEL_BOLD = True
SHOW_PANEL_TITLES = True   # the short title after each panel letter
PANEL_TITLE_BOLD = False
SHOW_FIGURE_TITLES = True  # the one-line title above Figures 2 to 8
FIGURE_TITLE_BOLD = False

FONT_SCALE = 1.0           # multiplies every font size: 1.1 is 10% larger.
                           # Every value from 0.9 to 1.1 was checked and
                           # passes; above 1.1 the layout check tells you
                           # which label needs shortening.
FONT_FAMILY = "DejaVu Sans"

OUTDIR = None              # None -> ../Figures_styled ; or any folder
LAYOUT_CHECK = True        # stop with an error when text collides

# Colours carry meaning, and the meaning is the same in every figure:
#   central   mean or median, the model itself            (blue)
#   noncentral quantile, spread, duration                 (orange)
#   good      passes, stable                              (green)
#   bad       fails, not located, negative                (red)
#   accent    the three night-time variables              (pink)
#   second    a secondary series, held-out states         (sky blue)
#   stages    the four stages of Figure 1 and the abstract (white text on them)
#   diverging the residual map of Figure 8, red to blue through light grey
# Both palettes keep blue against orange and green against red apart under
# the common forms of colour-vision deficiency.
PALETTES = {
    "paper": dict(
        central="#0072B2", noncentral="#E69F00", good="#009E73",
        bad="#D55E00", accent="#CC79A7", second="#56B4E9",
        stages=["#0072B2", "#6A3D9A", "#D55E00", "#009E73"],
        diverging=["#8A3A00", "#D55E00", "#F0B27A", "#F2F2F2", "#8FC7E8",
                   "#0072B2", "#00355A"]),
    "vibrant": dict(
        central="#0057E0", noncentral="#FF9500", good="#00B377",
        bad="#F23D0C", accent="#E3349F", second="#12B5F5",
        stages=["#0057E0", "#7B2FBE", "#E0360A", "#00905F"],
        diverging=["#7A2400", "#F23D0C", "#F7AE8A", "#F2F2F2", "#8CCBF5",
                   "#0057E0", "#002C70"]),
}

# neutral inks, shared by every palette
GREY, LIGHT, INK = "#8C8C8C", "#D9D9D9", "#222222"
FILL = "#F4F4F4"

# sizes in points at the authored width; FONT_SCALE multiplies them all
W = 5.5                    # inches, the MDPI text width; do not change
FS, FS_TICK, FS_LET = 8.5, 7.5, 9.5
FS_TITLE = FS + 1.0


# ===========================================================================
#  2. WHERE THE DATA ARE  (from _config.py, unchanged)
# ===========================================================================
HERE = Path(__file__).resolve().parent


@@PATHS@@

def _named(env_name, default):
    v = os.environ.get(env_name)
    return Path(v).expanduser().resolve() if v else default


_RUN = (os.environ.get("SATSDOH_RUN") or "").strip().strip("_")
DERIVED = _named("SATSDOH_DERIVED", PAPER / ("derived" + (f"_{_RUN}" if _RUN else "")))
FRAME = STUDY_A / "studyA_reference_frame.parquet"
RANKING_A = STUDY_A / "studyA_shap_ranking.csv"
GROUP = "fips"
NCE_TAIL = "eng_night_cooling_eff"
NCE_MEAN = "eng2_night_cooling_efficiency"
COUNTY_FILE = next((p for p in (HERE / "conus_counties_5070.gpkg",
                                PAPER / "analysis" / "conus_counties_5070.gpkg")
                    if p.exists()), HERE / "conus_counties_5070.gpkg")


@@TABLES@@

# ===========================================================================
#  3. HOUSE STYLE  (applied to every figure by save(), so none can drift)
# ===========================================================================
PAL = PALETTES[PALETTE]
BLUE = ORANGE = GREEN = RED = PINK = SKY = None
OUT = None
WRITE = True


def apply_settings():
    """Turn section 1 into colours, fonts and the output folder. Called once
    by main() after the command line has been read."""
    global PAL, BLUE, ORANGE, GREEN, RED, PINK, SKY, OUT
    if DPI < 600:
        raise SystemExit(f"DPI is {DPI}; 600 is the minimum for these figures.")
    if PALETTE not in PALETTES:
        raise SystemExit(f"PALETTE {PALETTE!r} is not one of {sorted(PALETTES)}")
    if "{}" not in PANEL_LABEL:
        raise SystemExit('PANEL_LABEL must contain {} where the letter goes, e.g. "({})"')
    PAL = PALETTES[PALETTE]
    BLUE, ORANGE = PAL["central"], PAL["noncentral"]
    GREEN, RED = PAL["good"], PAL["bad"]
    PINK, SKY = PAL["accent"], PAL["second"]
    OUT = Path(OUTDIR).expanduser().resolve() if OUTDIR else PAPER / "Figures_styled"
    plt.rcParams.update({
        "font.family": FONT_FAMILY, "font.size": FS, "font.weight": "normal",
        "axes.titlesize": FS, "axes.titleweight": "normal",
        "axes.labelsize": FS, "axes.labelweight": "normal",
        "axes.labelcolor": INK, "axes.edgecolor": "#555555",
        "axes.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": False,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "xtick.color": "#444444", "ytick.color": "#444444",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "legend.fontsize": FS_TICK, "legend.frameon": False,
        "lines.linewidth": 1.2, "patch.linewidth": 0.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "text.color": INK, "mathtext.default": "regular",
    })


def letter(ax, s):
    """Register a panel letter. It is placed at save time, at the outer left
    edge of the panel including its tick labels, so a letter can never sit on
    a label or fall off the canvas whatever the panel's margins are."""
    fig = ax.figure
    if not hasattr(fig, "_letters"):
        fig._letters = []
    fig._letters.append((ax, s))


# Figure titles and panel titles, one place for all of them.
@@TITLES@@

def _scale_fonts(fig):
    """Multiply every font size in the figure by FONT_SCALE. Tick labels are
    scaled through their axis, so ticks drawn later get the same size."""
    if FONT_SCALE == 1.0:
        return
    fig.canvas.draw()
    ticks = set()
    for ax in fig.axes:
        for which, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            tk = axis.get_major_ticks() + axis.get_minor_ticks()
            for t in tk:
                ticks.update((id(t.label1), id(t.label2)))
            if tk:
                ax.tick_params(axis=which, which="both",
                               labelsize=tk[0].label1.get_fontsize() * FONT_SCALE)
    for t in fig.findobj(Text):
        if id(t) not in ticks:
            t.set_fontsize(t.get_fontsize() * FONT_SCALE)


def _make_room(fig):
    """Larger text needs more room. With FONT_SCALE above 1 the canvas grows
    taller by the same factor, the left margin and the vertical gaps widen
    with it, and _resolve() widens the horizontal gaps only where text
    collides. The width stays the MDPI text width."""
    if FONT_SCALE <= 1.0:
        return
    from matplotlib.gridspec import GridSpec
    w, h = fig.get_size_inches()
    fig.set_size_inches(w, h * FONT_SCALE)
    seen = []
    for ax in fig.axes:
        ss = ax.get_subplotspec()
        if ss is None:
            continue
        gs = ss.get_topmost_subplotspec().get_gridspec()
        if isinstance(gs, GridSpec) and gs not in seen:
            seen.append(gs)
            p = gs.get_subplot_params(fig)
            gs.update(left=min(p.left * FONT_SCALE, 0.45),
                      hspace=p.hspace * FONT_SCALE)


def _tick_owner(fig):
    """id of every tick label -> (axes, 'x' or 'y')."""
    own = {}
    for ax in fig.axes:
        for which, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            for t in axis.get_major_ticks() + axis.get_minor_ticks():
                own[id(t.label1)] = own[id(t.label2)] = (ax, which)
    return own


def _wrap_once(label):
    """Break the longest line of a label at the space nearest its middle."""
    lines = label.split("\n")
    spaced = [i for i in range(len(lines)) if " " in lines[i]]
    if not spaced:
        return label
    k = max(spaced, key=lambda i: len(lines[i]))
    ln = lines[k]
    spaces = [i for i, c in enumerate(ln) if c == " "]
    if not spaces:
        return label
    cut = min(spaces, key=lambda i: abs(i - len(ln) / 2))
    lines[k:k + 1] = [ln[:cut], ln[cut + 1:]]
    return "\n".join(lines)


def _resolve(fig):
    """Only when FONT_SCALE is above 1: find the texts that the larger size
    makes collide and give them room, in up to six rounds. Crowded tick
    labels of one axis are eased back towards their original size; any other
    collision widens the gaps between panels. A figure with no collision is
    left exactly as it is."""
    if FONT_SCALE <= 1.0:
        return
    from matplotlib.gridspec import GridSpec
    base = {}
    for _ in range(6):
        clashes = _find_problems(fig, canvas=False)
        if not clashes:
            return
        own, widen = _tick_owner(fig), False
        for _msg, ti, tj in clashes:
            oi = own.get(id(ti))
            oj = own.get(id(tj)) if tj is not None else None
            if oi and oj and oi[0] is oj[0] and oi[1] != oj[1]:
                # an x and a y tick label meeting at a panel's corner
                for ax, which in (oi, oj):
                    axis = ax.xaxis if which == "x" else ax.yaxis
                    size = axis.get_major_ticks()[0].label1.get_fontsize()
                    floor = base.setdefault((id(ax), which), size / FONT_SCALE)
                    ax.tick_params(axis=which, labelsize=max(floor, size * 0.95))
            elif oi and oj and oi == oj:
                ax, which = oi
                axis = ax.xaxis if which == "x" else ax.yaxis
                labels = [t.get_text() for t in axis.get_ticklabels()]
                size = axis.get_major_ticks()[0].label1.get_fontsize()
                # first wrap a crowded multi-word label onto one more line
                wrapped = [_wrap_once(x) if x in (ti.get_text(), tj.get_text())
                           else x for x in labels]
                if wrapped != labels and which == "x":
                    axis.set_ticklabels(wrapped)
                    ax.tick_params(axis=which, labelsize=size)
                else:
                    # then ease the size back, never below the published size
                    floor = base.setdefault((id(ax), which), size / FONT_SCALE)
                    ax.tick_params(axis=which, labelsize=max(floor, size * 0.95))
            else:
                widen = True
        if widen:
            seen = []
            for ax in fig.axes:
                ss = ax.get_subplotspec()
                if ss is None:
                    continue
                gs = ss.get_topmost_subplotspec().get_gridspec()
                if isinstance(gs, GridSpec) and gs not in seen:
                    seen.append(gs)
                    p = gs.get_subplot_params(fig)
                    gs.update(wspace=p.wspace * 1.12)


def _place_letters(fig, panel_titles=None):
    """Panel letter at the outer left edge of each panel (tick labels
    included), followed on the same line by the panel title, both above
    anything the panel already carries on top."""
    panel_titles = panel_titles or {}
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, s in getattr(fig, "_letters", []):
        bb = ax.get_tightbbox(r).transformed(inv)
        pos = ax.get_position()
        # a long axis label can reach above the panel; the letter goes above it
        labs = [a.label.get_window_extent(r).transformed(inv).y1
                for a in (ax.xaxis, ax.yaxis)
                if a.label.get_visible() and a.label.get_text()]
        y = max([pos.y1, bb.y1] + labs) + 0.006
        t = fig.text(bb.x0, y, PANEL_LABEL.format(s), fontsize=FS_LET * FONT_SCALE,
                     fontweight="bold" if PANEL_LABEL_BOLD else "normal",
                     ha="left", va="bottom", color=INK)
        t._is_letter = True
        title = panel_titles.get(s) if SHOW_PANEL_TITLES else None
        if title:
            w = t.get_window_extent(r).transformed(inv).width
            fig.text(bb.x0 + w + 0.012, y, title, fontsize=FS * FONT_SCALE,
                     fontweight="bold" if PANEL_TITLE_BOLD else "normal",
                     ha="left", va="bottom", color=INK)


def _place_title(fig, text):
    """The figure title, centred above everything else on the canvas."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    top = fig.get_tightbbox(r).y1 / fig.get_figheight()   # inches to figure
    fig.text(0.5, top + 0.02, text, fontsize=FS_TITLE * FONT_SCALE, ha="center",
             va="bottom", color=INK,
             fontweight="bold" if FIGURE_TITLE_BOLD else "normal")


def save(fig, stem):
    """Figures 2 to 8: fonts, letters and title, the layout check, then the
    files. The canvas grows to hold every text (bbox_inches='tight')."""
    fig_title, panel_titles = TITLES.get(stem, ("", {}))
    # every text regular weight first, then the panel letters alone in bold
    for t in fig.findobj(Text):
        t.set_fontweight("normal")
    _make_room(fig)
    _scale_fonts(fig)
    _resolve(fig)
    _place_letters(fig, panel_titles)
    if fig_title and SHOW_FIGURE_TITLES:
        _place_title(fig, fig_title)
    check_layout(fig, stem, canvas=False)
    if WRITE:
        OUT.mkdir(parents=True, exist_ok=True)
        # Matplotlib clips an axis label to the canvas when it computes the
        # tight box, so a long label can lose its last characters. Listing
        # every text explicitly makes the saved area grow to hold it instead.
        extra = fig.get_default_bbox_extra_artists() + [
            t for ax in fig.axes for t in (ax.xaxis.label, ax.yaxis.label)
            if t.get_visible() and t.get_text()] + list(fig.texts)
        for ext in FORMATS:
            fig.savefig(OUT / f"{stem}.{ext}", dpi=DPI, facecolor="white",
                        bbox_inches="tight", pad_inches=0.04,
                        bbox_extra_artists=extra)
        print(f"  [saved] {OUT.name}/{stem} ({', '.join(FORMATS)}) at {DPI} dpi")
    plt.close(fig)


def _save(fig, stem: Path, dpi=None):
    """Figure 1 and the graphical abstract: fixed canvas, same check."""
    check_layout(fig, stem.name, canvas=True)
    if WRITE:
        stem.parent.mkdir(parents=True, exist_ok=True)
        for ext in FORMATS:
            fig.savefig(f"{stem}.{ext}", dpi=DPI, facecolor="white")
        print(f"  [saved] {stem.parent.name}/{stem.name} ({', '.join(FORMATS)}) at {DPI} dpi")
    plt.close(fig)


# ===========================================================================
#  4. LAYOUT CHECK
# ===========================================================================
#  Draws the figure once and records the box of every text actually drawn.
#  Three faults are reported:
#    * two texts whose boxes overlap,
#    * a text lying on the plotting area of a panel it does not belong to,
#    * (Figure 1 and the abstract, whose canvas is fixed) a text leaving the
#      canvas.
#  Text boxes are measured tight around the letters, so two texts must stay
#  at least GAP_PT points apart; a text may touch a panel's plotting area by
#  up to 1.5 points (its own blank margin).
PROBLEMS = []
TOL_PT = 1.5
GAP_PT = 0.8


def _drawn_texts(fig):
    drawn, orig, orig_extent = [], Text.draw, Text.get_window_extent

    def rec(self, renderer):
        orig(self, renderer)
        if self.get_visible() and self.get_text().strip() and self.get_alpha() != 0:
            # the text alone: an annotation's own extent includes its leader line
            drawn.append((self, orig_extent(self, renderer)))
    Text.draw = rec
    try:
        fig.canvas.draw()
    finally:
        Text.draw = orig
    return drawn


def _overlap(a, b, tol):
    return (min(a.x1, b.x1) - max(a.x0, b.x0) > tol
            and min(a.y1, b.y1) - max(a.y0, b.y0) > tol)


def _crowd(a, b, gap):
    """Side by side closer than `gap`, or one on top of the other."""
    return (min(a.x1, b.x1) - max(a.x0, b.x0) > -gap
            and min(a.y1, b.y1) - max(a.y0, b.y0) > gap / 2)


def _owner(t):
    """The panel a text belongs to; None for the letters and titles, which
    belong to the figure."""
    return t.axes


def _find_problems(fig, canvas):
    """[(message, text, other text or None)] for every fault on the canvas."""
    tol = TOL_PT * fig.dpi / 72
    drawn = _drawn_texts(fig)
    found = []
    for i in range(len(drawn)):
        ti, bi = drawn[i]
        for j in range(i + 1, len(drawn)):
            tj, bj = drawn[j]
            if _crowd(bi, bj, GAP_PT * fig.dpi / 72):
                found.append((f"texts overlap or touch: {ti.get_text()!r} and "
                              f"{tj.get_text()!r}", ti, tj))
        own = _owner(ti)
        for ax in fig.axes:
            if ax is own or not ax.get_visible():
                continue
            if own is not None and (ax in getattr(own, "child_axes", [])
                                    or own in getattr(ax, "child_axes", [])):
                continue
            if _overlap(bi, ax.bbox, tol):
                found.append((f"text {ti.get_text()!r} lies on another panel",
                              ti, None))
                break
        if canvas:
            fb = fig.bbox
            if (bi.x0 < fb.x0 - tol or bi.x1 > fb.x1 + tol
                    or bi.y0 < fb.y0 - tol or bi.y1 > fb.y1 + tol):
                found.append((f"text {ti.get_text()!r} runs off the canvas",
                              ti, None))
    return found


def check_layout(fig, name, canvas):
    if not LAYOUT_CHECK:
        return
    msgs = [m for m, _a, _b in _find_problems(fig, canvas)]
    for f in dict.fromkeys(msgs):
        PROBLEMS.append((name, f))
        print(f"  [LAYOUT] {name}: {f}")


# ===========================================================================
#  5. FIGURE 1 AND THE GRAPHICAL ABSTRACT
# ===========================================================================
@@SCHEMATIC@@

# ===========================================================================
#  6. FIGURES 2 TO 8
# ===========================================================================
@@FIGURES@@

# ===========================================================================
#  7. RUN
# ===========================================================================
def _schematic(which):
    K = read_constants(PAPER)
    (figure01 if which == "1" else graphical_abstract)(K, OUT)


FIGS = {"1": lambda: _schematic("1"), "2": fig02, "3": fig03, "4": fig04,
        "5": fig05, "6": fig06, "7": fig07, "8": fig08,
        "ga": lambda: _schematic("ga")}


def main():
    global DPI, PALETTE, PANEL_LABEL, FONT_SCALE, OUTDIR, WRITE
    global FIGURE_TITLE_BOLD, PANEL_LABEL_BOLD
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", default="", help="e.g. 1,5,8,ga")
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--palette", default=PALETTE, choices=sorted(PALETTES))
    ap.add_argument("--labels", default=PANEL_LABEL, help='e.g. "({})" or "(A)"')
    ap.add_argument("--font-scale", type=float, default=FONT_SCALE)
    ap.add_argument("--title-bold", action="store_true")
    ap.add_argument("--labels-regular", action="store_true",
                    help="panel letters in regular weight")
    ap.add_argument("--outdir", default=OUTDIR or "")
    ap.add_argument("--check-only", action="store_true",
                    help="draw and check every figure, write nothing")
    a = ap.parse_args()
    DPI, PALETTE, FONT_SCALE = a.dpi, a.palette, a.font_scale
    # accept "(A)" as well as "({})"
    PANEL_LABEL = a.labels if "{}" in a.labels else a.labels.replace("A", "{}", 1)
    OUTDIR = a.outdir or None
    WRITE = not a.check_only
    FIGURE_TITLE_BOLD = FIGURE_TITLE_BOLD or a.title_bold
    PANEL_LABEL_BOLD = PANEL_LABEL_BOLD and not a.labels_regular
    apply_settings()
    want = ([x.strip().lower() for x in a.only.split(",") if x.strip()]
            if a.only else list(FIGS))
    print(f"palette {PALETTE}, panel label {PANEL_LABEL.format('A')}, "
          f"font scale {FONT_SCALE}, {DPI} dpi -> {OUT}")
    failed = []
    for k in want:
        if k not in FIGS:
            raise SystemExit(f"unknown figure {k!r}; choose from {list(FIGS)}")
        print(f"Figure {k}")
        try:
            FIGS[k]()
        except Exception:                              # report, keep going
            import traceback
            traceback.print_exc()
            failed.append(k)
    if failed:
        raise SystemExit(f"figures that failed: {failed}")
    if PROBLEMS:
        raise SystemExit(f"{len(PROBLEMS)} layout problem(s), listed above as "
                         "[LAYOUT]. The files were written so you can see "
                         "them; a smaller FONT_SCALE or a shorter label "
                         "usually clears them.")
    print("layout check passed: no overlapping or misplaced text")


if __name__ == "__main__":
    main()
