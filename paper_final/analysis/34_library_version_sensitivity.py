#!/usr/bin/env python3
"""
34_library_version_sensitivity.py -- how much do the numbers move between
library versions?

READS   ../derived/{spatial_holdout,residual_spatial_structure,
                    permutation_across_splits}.json        (final run)
        ../derived/reference_run_2026-10-05/...            (earlier run)
        the manifests of both runs, for the library versions
WRITES  ../derived/library_version_sensitivity.json

WHY THIS EXISTS
---------------
Every script is seeded and gives identical answers when rerun on one machine
(scripts 01 and 10 prove it). Across library releases that is not guaranteed:
a new xgboost release may build a slightly different tree from the same data
and seed. The battery was run twice with identical settings, once with
xgboost 3.2.0 and once with xgboost 3.0.2. This script records which numbers
moved and by how much, so the paper can say so in one sentence. It computes
nothing new; it compares two sets of result files.

    python3 34_library_version_sensitivity.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _config as C                                            # noqa: E402

REF = C.DERIVED / "reference_run_2026-10-05"
OUT = C.DERIVED / "library_version_sensitivity.json"


def load(folder, name):
    return json.load(open(folder / name))


def versions(folder):
    m = folder / "manifests" / "27_permutation_importance_across_splits.manifest.json"
    if not m.exists():
        return {}
    j = json.load(open(m))
    return dict(j.get("versions", {}), platform=j.get("versions", {}).get("platform"))


def main():
    C.banner("34  library-version sensitivity")
    if not REF.exists():
        print("  no reference run present; nothing to compare")
        return
    new_v, old_v = versions(C.DERIVED), versions(REF)
    hn, ho = load(C.DERIVED, "spatial_holdout.json"), load(REF, "spatial_holdout.json")
    mn, mo = (load(C.DERIVED, "residual_spatial_structure.json"),
              load(REF, "residual_spatial_structure.json"))
    pn, po = (load(C.DERIVED, "permutation_across_splits.json"),
              load(REF, "permutation_across_splits.json"))
    old = {s["seed"]: s for s in po["splits"]}
    shifts = {who: [abs(s[f"{who}_wins"] - old[s["seed"]][f"{who}_wins"])
                    for s in pn["splits"] if s["seed"] in old]
              for who in ("production", "second")}
    same_split = sum(s["production_wins"] == old[s["seed"]]["production_wins"]
                     for s in pn["splits"] if s["seed"] in old)
    out = dict(
        versions_final=new_v, versions_reference=old_v,
        spatial_holdout_r2=dict(final=hn["r2"], reference=ho["r2"]),
        residual_morans_i=dict(final=mn["morans_i"], reference=mo["morans_i"],
                               p_final=mn["p_value"], p_reference=mo["p_value"]),
        permutation_count_shift=dict(
            production_max=max(shifts["production"]),
            second_max=max(shifts["second"]),
            production_splits_identical=int(same_split),
            n_splits=len(pn["splits"])),
        note="Same seeds and settings in both runs; only library versions differ.")
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"  xgboost {old_v.get('xgboost')} -> {new_v.get('xgboost')}")
    print(f"  state hold-out R2      {ho['r2']:.3f} -> {hn['r2']:.3f}")
    print(f"  residual Moran's I     {mo['morans_i']:.3f} -> {mn['morans_i']:.3f}")
    print(f"  permutation count shift, production up to "
          f"{max(shifts['production'])} fields, second learner up to "
          f"{max(shifts['second'])}")
    print(f"  [saved] {OUT}")
    C.write_manifest("34_library_version_sensitivity.py",
                     inputs=[C.DERIVED / "spatial_holdout.json"], outputs=[OUT],
                     settings={})


if __name__ == "__main__":
    main()
