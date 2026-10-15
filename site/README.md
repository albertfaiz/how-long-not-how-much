# Project website

`index.html` at the repository root is the project page served by GitHub Pages
(workflow `.github/workflows/pages.yml`). It presents the paper with its
equations, figures and tables for readers and new members of the lab.

How it is made:

- `tex2html.py` reads the manuscript source and turns paragraphs, captions and
  tables into HTML. Every number comes from `paper_final/constants_base.tex` and
  `paper_final/constants_results.tex`, the same files that write the numbers
  into the paper, so the page and the paper cannot disagree.
- `build_site.py` assembles the page: summary tiles, abstract, methods with
  equations, results, the turning-point and lifetime tables, limitations,
  reproduction steps, glossary and citation.
- `figures/` holds web copies (2000 pixels wide) of the paper figures in
  `paper_final/Figures/`.

The manuscript source `main.tex` is not part of this repository while the paper
is under review. To rebuild the page, place it in `paper_final/` (or point
`SITE_PAPER_DIR` at the folder that holds it) and run

```
python site/build_site.py
```

Equations are typeset in the browser by MathJax 3.2.2, loaded from
cdn.jsdelivr.net.

Terminology: atmospheric (tropospheric) lifetime is used in its standard sense,
the burden divided by the removal rate (the e-folding time); a half-life is that
lifetime times ln 2. Terms introduced by this study are marked "(this study)" in
the glossary.
