#!/usr/bin/env python3
"""
figure01_methodology_and_graphical_abstract.py -- the two schematic figures.

READS   ../constants_base.tex and ../constants_results.tex, the same generated
        constants the manuscript prints, so every number in these figures is
        the number in the text
WRITES  ../Figures/Figure01_methodology.png and .pdf     (Figure 1)
        ../Figures/GraphicalAbstract.png and .pdf       (uploaded separately)

Both are placeholders in the sense that a hand-drawn version can replace
either: save it under the same file name and the manuscript picks it up.
They are drawn here so that the paper compiles complete without one.

House style as in paper_figures.py: DejaVu Sans, regular weight only, the
Okabe-Ito colours with the same meaning as in Figures 2 to 10.

    python3 figure01_methodology_and_graphical_abstract.py
    python3 figure01_methodology_and_graphical_abstract.py --paper ../../paper_final
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# ---------------------------------------------------------------- settings
HERE = Path(__file__).resolve().parent
PAPER_DEFAULT = HERE.parent           # the folder holding main.tex
DPI = 600
W = 5.5                               # inches, the MDPI text width

BLUE, ORANGE, GREEN, RED = "#0072B2", "#E69F00", "#009E73", "#D55E00"
PINK, SKY, GREY, INK = "#CC79A7", "#56B4E9", "#8C8C8C", "#222222"
FILL = "#F4F4F4"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.weight": "normal",
    "text.color": INK, "mathtext.default": "regular",
    "pdf.fonttype": 42, "figure.facecolor": "white",
})


# ---------------------------------------------------------------- constants
def read_constants(paper: Path) -> dict:
    """Every macro of the two constants files, resolved the way TeX resolves
    them: files in the order main.tex loads them, \\providecommand only when
    the name is still undefined, \\newcommand and \\renewcommand always.
    Values come back as printable text ({,} becomes a comma, $-$ a minus)."""
    vals = {}
    head = re.compile(r"\\(newcommand|renewcommand|providecommand)\{\\([A-Za-z]+)\}\{")
    for name in ("constants_base.tex", "constants_results.tex"):
        text = "\n".join(line.split("%")[0] if not line.lstrip().startswith("%")
                         else "" for line in (paper / name).read_text().splitlines())
        pos = 0
        while True:
            m = head.search(text, pos)
            if not m:
                break
            depth, i = 1, m.end()
            while depth and i < len(text):
                depth += {"{": 1, "}": -1}.get(text[i], 0)
                i += 1
            value, pos = text[m.end():i - 1], i
            kind, key = m.group(1), m.group(2)
            if kind == "providecommand" and key in vals:
                continue
            vals[key] = (value.replace("{,}", ",").replace("$-$", "\u2212")
                         .replace("\\,", " ").replace("$", ""))
    return vals


def box(ax, x, y, w, h, fc=FILL, ec="none"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                fc=fc, ec=ec, lw=0.8))


def arrow(ax, x0, y0, x1, y1, scale=9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=scale, lw=0.9, color=GREY))


# ---------------------------------------------------------------- Figure 1
def figure01(K: dict, out: Path):
    fig = plt.figure(figsize=(W, 3.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")

    stages = [
        ("1  Reduce", BLUE,
         [f"{K['Nmodalities']} satellite and",
          "gridded modalities",
          f"{K['Ncounties']} counties",
          f"{K['YearStart']} to {K['YearEnd']}",
          "several summaries",
          "per field, one pass"]),
        ("2  Model", BLUE,
         [f"{K['Nfeatpruned']} admitted columns",
          "no socioeconomic",
          "or clinical input",
          "XGBoost, 8 county-",
          "grouped folds",
          f"R\u00b2 = {K['Rtwo']}",
          "interventional SHAP"]),
        ("3  Compare", ORANGE,
         [f"census of {K['StatFieldsComparable']} fields",
          "central vs non-central",
          "SHAP vs mutual",
          "information",
          "matched refit",
          "8 instruments, nulls"]),
        ("4  What survives", GREEN,
         ["atmospheric lifetime",
          f"({K['LtSpecies']} species)",
          "turning points: noise,",
          "100th meridian",
          "model error vs",
          "socioeconomic data"]),
    ]
    x0, wbox, gap = 0.25, 2.45, 0.27
    for i, (head, col, lines) in enumerate(stages):
        x = x0 + i * (wbox + gap)
        box(ax, x, 0.35, wbox, 4.75)
        ax.add_patch(FancyBboxPatch((x, 4.6), wbox, 0.5,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=col, ec="none"))
        ax.text(x + 0.14, 4.85, head, fontsize=8, color="white",
                va="center", ha="left")
        for j, line in enumerate(lines):
            ax.text(x + 0.14, 4.1 - j * 0.55, line, fontsize=6.9,
                    va="center", ha="left", color=INK)
        if i < len(stages) - 1:
            arrow(ax, x + wbox + 0.02, 2.7, x + wbox + gap - 0.02, 2.7)
    ax.text(5.5, 5.6, "Study design"
            , fontsize=9.5, ha="center", va="center")
    _save(fig, out / "Figure01_methodology")


# ---------------------------------------------------------------- abstract
def graphical_abstract(K: dict, out: Path):
    """The visual abstract, in the style of Figure 1: four stages left to
    right, each with its key numbers. Landscape, 3300 x 1410 px at 300 dpi,
    above MDPI's minimum of 1100 x 560 px."""
    fig = plt.figure(figsize=(11.0, 4.7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 22)
    ax.set_ylim(1.0, 10.4)
    ax.axis("off")
    fs, fsn = 11, 15

    ax.text(11, 9.75, "Predictive multiplicity in environmental health",
            fontsize=17, ha="center", va="center")

    N = K["StatFieldsComparable"]
    stages = [
        ("1  Data and model", BLUE, [
            ("text", f"{K['Nmodalities']} satellite and gridded modalities"),
            ("text", f"{K['Ncounties']} US counties"),
            ("text", f"{K['YearStart']} to {K['YearEnd']}"),
            ("text", "no socioeconomic input"),
            ("num", f"R\u00b2 = {K['Rtwo']}"),
            ("text", "out of fold, county-grouped"),
        ]),
        ("2  Two descriptions", BLUE, [
            ("text", "formaldehyde, time above P75"),
            ("num", f"rank {K['HchoDurRank']}"),
            ("text", "same field, annual mean"),
            ("num", f"rank {K['HchoConRank']}"),
            ("text", "night cooling, two forms"),
            ("num", f"{K['NCEratio']}\u00d7 apart"),
        ]),
        ("3  Multiplicity", ORANGE, [
            ("text", "SHAP favours a non-central summary"),
            ("num", f"{K['StatNonCentralWins']} of {N}"),
            ("text", f"{K['ExchExpected']} expected at random"),
            ("text", "mutual information favours the mean"),
            ("num", f"{K['MIcentralWins']} of {N}"),
            ("text", "R\u00b2, central-only vs non-central-only"),
            ("num", f"{K['StatRefitCentralRtwo']} vs {K['StatRefitNonCentralRtwo']}"),
        ]),
        ("4  What survives", GREEN, [
            ("text", "duration advantage vs lifetime"),
            ("num", f"\u03c1 = {K['LtRho']}"),
            ("text", "nocturnal turning points"),
            ("num", f"dew point {K['TpDew']} \u00b0C"),
            ("num", f"wet-bulb {K['TpWet']} \u00b0C"),
            ("text", "shift across the 100th meridian"),
            ("num", f"\u2264 {K['RegMerTripletMax']}% of range"),
        ]),
    ]
    x0, wbox, gap = 0.3, 5.05, 0.36
    for i, (head, col, rows) in enumerate(stages):
        x = x0 + i * (wbox + gap)
        box(ax, x, 2.15, wbox, 6.65)
        ax.add_patch(FancyBboxPatch((x, 8.0), wbox, 0.8,
                                    boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=col, ec="none"))
        ax.text(x + 0.22, 8.4, head, fontsize=fs + 1, color="white",
                va="center", ha="left")
        y = 7.35
        for kind, txt in rows:
            if kind == "text":
                ax.text(x + 0.22, y, txt, fontsize=fs - 1.5, va="center",
                        color="#444444")
                y -= 0.58
            else:
                ax.text(x + 0.22, y, txt, fontsize=fsn if len(txt) < 22 else fs + 1,
                        va="center", color=INK)
                y -= 0.85
        if i < len(stages) - 1:
            arrow(ax, x + wbox + 0.03, 5.2, x + wbox + gap - 0.03, 5.2, scale=16)

    ax.text(11, 1.55, "Attribution records how a model digests a field, not how much "
            "information the field holds; physical structure survives the multiplicity.",
            fontsize=fs, ha="center", va="center", color=INK)
    _save(fig, out / "GraphicalAbstract", dpi=300)


def _save(fig, stem: Path, dpi: int = DPI):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"  [saved] {stem.name}.png and .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=str(PAPER_DEFAULT),
                    help="folder holding main.tex and the two constants files")
    ap.add_argument("--outdir", default="",
                    help="where to write; default <paper>/Figures")
    a = ap.parse_args()
    paper = Path(a.paper).resolve()
    out = Path(a.outdir).resolve() if a.outdir else paper / "Figures"
    K = read_constants(paper)
    figure01(K, out)
    graphical_abstract(K, out)


if __name__ == "__main__":
    main()
