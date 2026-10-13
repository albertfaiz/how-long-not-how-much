"""
10_verify_reproducibility_by_rerunning.py
Run a script twice and prove its answers did not move.

WHY THIS EXISTS
---------------
A number in a manuscript has to survive being recomputed. The way to know that
is not to reason about seeds but to run the thing again and compare the bytes.
This does exactly that: it records the hash of every output a script wrote,
reruns the script, and compares. Any difference is printed with the file that
differs.

It is also the script to reach for when a number in the paper has changed and
you want to know why. The manifests hold the settings each run used, so a
changed number is traced to a changed setting rather than to chance.

WHAT IT CANNOT PROVE
--------------------
That the answer is right. Only that it is stable. A reproducible mistake is
still a mistake.

WHAT IT COSTS
-------------
It reruns the script, so verifying script 04 costs a second full refit. Verify
the fast ones routinely and the slow ones once, before submission.

    python3 10_verify_reproducibility_by_rerunning.py 02
    python3 10_verify_reproducibility_by_rerunning.py 02 06
    python3 10_verify_reproducibility_by_rerunning.py --all-fast
    python3 10_verify_reproducibility_by_rerunning.py      # same as --all-fast
"""
from __future__ import annotations

# ===========================================================================
#  SETTINGS
# ===========================================================================
# Scripts that finish quickly enough to verify as a matter of routine.
FAST = ["02", "07"]
# Scripts that cost a full refit; verify these deliberately.
SLOW = ["03", "04", "05", "06"]
# Extra arguments passed to a script when verifying it, to keep the cost down
# while still exercising the same code path.
VERIFY_ARGS = {
    "06": ["--boot", "200"],
}
# ===========================================================================

import argparse
import json
import subprocess
import sys
from pathlib import Path

import _config as C


def find_script(prefix: str) -> Path:
    hits = sorted(C.ANALYSIS.glob(f"{prefix}_*.py"))
    if not hits:
        raise SystemExit(f"no script in this folder starts with {prefix!r}")
    return hits[0]


def manifest_for(script: Path) -> Path:
    return C.MANIFESTS / (script.stem + ".manifest.json")


def snapshot(script: Path):
    m = manifest_for(script)
    if not m.exists():
        return None
    return json.load(open(m))


def run(script: Path, extra):
    print(f"\n  running {script.name} "
          f"{' '.join(extra) if extra else ''}".rstrip())
    r = subprocess.run([sys.executable, str(script), *extra],
                       cwd=str(C.ANALYSIS), capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"  {script.name} exited {r.returncode}")
    return r.stdout


def compare(before, after, script):
    print(f"\n  {script.name}")
    if before is None:
        print("    no earlier manifest; this run recorded the baseline.")
        print("    run this command again to compare against it.")
        return None
    # The thread count is recorded but cannot move an answer: script 01
    # proves the fitted model identical at 1 and 4 threads. A run on a
    # machine with a different core count is therefore still a rerun.
    ignore = {"threads"}
    strip = lambda d: {k: v for k, v in d.items() if k not in ignore}
    same_settings = strip(before["settings"]) == strip(after["settings"])
    if before["settings"].get("threads") != after["settings"].get("threads"):
        print(f"    threads {before['settings'].get('threads')} -> "
              f"{after['settings'].get('threads')} (does not affect results; see 01)")
    if not same_settings:
        print("    SETTINGS DIFFER between the two runs:")
        for k in sorted((set(before["settings"]) | set(after["settings"])) - ignore):
            b, a = before["settings"].get(k), after["settings"].get(k)
            if b != a:
                print(f"      {k}: {b!r}  ->  {a!r}")
        print("    A different answer is expected. Fix the settings, not the")
        print("    number.")
    ok = True
    keys = sorted(set(before["outputs"]) | set(after["outputs"]))
    if not keys:
        print("    this script writes no data files; nothing to compare")
        return same_settings
    for k in keys:
        b = before["outputs"].get(k, "absent before")
        a = after["outputs"].get(k, "absent after")
        if b == a:
            print(f"    identical  {k}")
        else:
            ok = False
            print(f"    DIFFERS    {k}   {b} -> {a}")
    return ok and same_settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scripts", nargs="*", help="number prefixes, e.g. 02 06")
    ap.add_argument("--all-fast", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    picks = a.scripts or (FAST if a.all_fast else
                          (FAST + SLOW if a.all else []))
    if not picks:
        # Run with no arguments (for example from a loop over every numbered
        # script), verify the fast scripts rather than stopping the loop.
        print("  no script named; verifying the fast ones "
              f"({' '.join(FAST)}). Slow ones: {' '.join(SLOW)}, "
              "verify those by naming them.")
        picks = FAST

    C.banner("10  reproducibility check")
    print("  Each script is rerun and its outputs compared, by hash, against")
    print("  the manifest of the previous run.\n")

    verdicts = {}
    for pref in picks:
        s = find_script(pref)
        before = snapshot(s)
        run(s, VERIFY_ARGS.get(pref, []))
        after = snapshot(s)
        if after is None:
            print(f"\n  {s.name} wrote no manifest; cannot verify")
            verdicts[s.name] = None
            continue
        verdicts[s.name] = compare(before, after, s)

    print("\n" + "=" * 74)
    bad = [k for k, v in verdicts.items() if v is False]
    new = [k for k, v in verdicts.items() if v is None]
    good = [k for k, v in verdicts.items() if v is True]
    for k in good:
        print(f"  REPRODUCES   {k}")
    for k in new:
        print(f"  baseline set {k}  (run again to compare)")
    for k in bad:
        print(f"  DOES NOT     {k}")
    print("=" * 74)
    if bad:
        print("  Do not publish a number from a script that does not")
        print("  reproduce until you know which setting moved.")
        sys.exit(1)


if __name__ == "__main__":
    main()
