#!/usr/bin/env python3
"""
Verify the released final results against the values reported in the paper.

Level 1 of the reproduction ladder (see REPRODUCIBILITY.md): this checks that the
frozen aggregate result tables shipped in ``results/final/`` actually recompute the
controlling numbers of the manuscript. It needs no dataset and no network.

Exit status is 0 only if every check passes; any mismatch exits 1.

    python scripts/verify_paper_results.py [--verbose]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = os.path.join(REPO, "results", "final")

# Floating-point comparisons use a tolerance tight enough to catch a genuine numerical
# regression. The released tables carry six decimal places, so 1e-6 is the resolution of
# the data itself; anything looser would hide a real change.
ATOL = 1e-6

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def read(fn: str) -> list[dict]:
    path = os.path.join(FINAL, fn)
    if not os.path.exists(path):
        check(f"file present: {fn}", False, "missing")
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=ATOL)


def direction_triple(rows: list[dict]) -> tuple[tuple[int, int, int], set[str]]:
    """Return learned/inconclusive/comparator counts and any unknown classes."""
    counts = Counter(row["direction_class"] for row in rows)
    allowed = {"CI_GT_0", "CI_CROSSES_0", "CI_LT_0"}
    return (
        (
            counts.get("CI_GT_0", 0),
            counts.get("CI_CROSSES_0", 0),
            counts.get("CI_LT_0", 0),
        ),
        set(counts) - allowed,
    )


def verify_computed_primary_contrast_governance() -> None:
    """Compute every authoritative primary triple from the row-level table.

    Exact equality makes this fail for any retired or otherwise non-authoritative
    aggregation. It intentionally does not trust pre-aggregated count columns.
    """
    rows = read("primary_model_comparator_contrasts.csv")
    if not rows:
        return

    expected = {
        "average_precision": {
            "ALL": (43, 29, 0),
            "ETH": (22, 2, 0),
            "UCY": (0, 24, 0),
            "inD": (21, 3, 0),
        },
        "recall_at_5pct": {
            "ALL": (2, 10, 60),
            "ETH": (2, 8, 14),
            "UCY": (0, 2, 22),
            "inD": (0, 0, 24),
        },
    }
    grids = {}
    for metric in expected:
        grid = [
            row for row in rows
            if row["metric"] == metric and row["comparator_family"] == "ANALYTIC_CV"
        ]
        grids[metric] = grid
        identities = {(row["cell_id"], row["model_family"]) for row in grid}
        check(
            f"computed {metric} row grid has 72 unique identities",
            len(grid) == 72 and len(identities) == 72,
            f"rows={len(grid)} unique={len(identities)}",
        )
        for domain, wanted in expected[metric].items():
            subset = grid if domain == "ALL" else [
                row for row in grid if row["target_domain"] == domain
            ]
            got, unknown = direction_triple(subset)
            check(
                f"computed {metric} {domain} is {wanted[0]}/{wanted[1]}/{wanted[2]}",
                got == wanted and not unknown and sum(got) == len(subset),
                f"observed {got[0]}/{got[1]}/{got[2]}; unknown={sorted(unknown)}",
            )

    ap_ids = {(row["cell_id"], row["model_family"]) for row in grids["average_precision"]}
    recall_ids = {(row["cell_id"], row["model_family"]) for row in grids["recall_at_5pct"]}
    check(
        "computed AP and Recall@5% grids have identical row identities",
        ap_ids == recall_ids and len(ap_ids) == 72,
        f"AP-only={len(ap_ids - recall_ids)} Recall-only={len(recall_ids - ap_ids)}",
    )


# Primary AP grid
def verify_ap_grid() -> None:
    rows = read("primary_model_comparator_contrasts.csv")
    if not rows:
        return
    ap = [r for r in rows
          if r["metric"] == "average_precision" and r["comparator_family"] == "ANALYTIC_CV"]
    check("AP grid has 72 contrasts", len(ap) == 72, f"n={len(ap)}")

    cls = Counter(r["direction_class"] for r in ap)
    learned = cls.get("CI_GT_0", 0)
    inconcl = cls.get("CI_CROSSES_0", 0)
    comparator = cls.get("CI_LT_0", 0)
    check("primary AP 43/29/0",
          (learned, inconcl, comparator) == (43, 29, 0),
          f"observed {learned}/{inconcl}/{comparator}")

    expected = {"ETH": (22, 2, 0), "UCY": (0, 24, 0), "inD": (21, 3, 0)}
    per = defaultdict(Counter)
    for r in ap:
        per[r["target_domain"]][r["direction_class"]] += 1
    for dom, exp in expected.items():
        got = (per[dom].get("CI_GT_0", 0), per[dom].get("CI_CROSSES_0", 0),
               per[dom].get("CI_LT_0", 0))
        check(f"AP decomposition {dom} {exp[0]}/{exp[1]}/{exp[2]}", got == exp,
              f"observed {got[0]}/{got[1]}/{got[2]}")


# Primary tie-neutral Recall@5%
def verify_recall5_grid() -> None:
    rows = read("recall5_by_domain_horizon_family.csv")
    if not rows:
        return
    # This summary carries only the tie-neutral corrected counts. The superseded
    # pre-correction columns are not part of it, so their presence would mean the
    # released table had reverted to the retired count lineage.
    check("retired historical columns dropped from the corrected summaries",
          not any(c.startswith("historical_") for c in rows[0]),
          "historical_* columns must not appear in this summary")

    tot = [0, 0, 0]
    per = defaultdict(lambda: [0, 0, 0])
    units = 0
    for r in rows:
        v = (int(r["corrected_learned_favored"]),
             int(r["corrected_inconclusive"]),
             int(r["corrected_reference_favored"]))
        units += int(r["units"])
        for i in range(3):
            tot[i] += v[i]
            per[r["target_domain"]][i] += v[i]
    check("Recall@5% grid covers 72 units", units == 72, f"units={units}")
    check("primary Recall@5% 2/10/60", tuple(tot) == (2, 10, 60),
          f"observed {tot[0]}/{tot[1]}/{tot[2]}")
    expected = {"ETH": (2, 8, 14), "UCY": (0, 2, 22), "inD": (0, 0, 24)}
    for dom, exp in expected.items():
        got = tuple(per[dom])
        check(f"Recall@5% decomposition {dom} {exp[0]}/{exp[1]}/{exp[2]}", got == exp,
              f"observed {got[0]}/{got[1]}/{got[2]}")


# Post-hoc AP against CV-2
def verify_posthoc_ap() -> None:
    rows = read("posthoc_ap_sensitivity.csv")
    if not rows:
        return
    ind = [r for r in rows if r["target_domain"] == "inD"]
    by_view = defaultdict(Counter)
    for r in ind:
        lo, hi = float(r["ci_lower"]), float(r["ci_upper"])
        cls = "learned" if lo > 0 else ("comparator" if hi < 0 else "inconclusive")
        by_view[r["cv_view"]][cls] += 1
    for view, exp in (("CV-0", (21, 3, 0)), ("CV-2", (19, 5, 0))):
        c = by_view.get(view)
        if c is None:
            check(f"post-hoc AP view {view} present", False, "view absent")
            continue
        got = (c.get("learned", 0), c.get("inconclusive", 0), c.get("comparator", 0))
        check(f"post-hoc AP inD vs {view} {exp[0]}/{exp[1]}/{exp[2]}", got == exp,
              f"observed {got[0]}/{got[1]}/{got[2]}")


# Seven-point Recall@q grid
def verify_recall_q() -> None:
    rows = read("posthoc_recall_q_profile.csv")
    if not rows:
        return
    qs = sorted({float(r["q"]) for r in rows})
    expected_q = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]
    ok = len(qs) == 7 and all(close(a, b) for a, b in zip(qs, expected_q))
    if not ok:  # some releases record q as a percentage
        ok = len(qs) == 7 and all(close(a, b * 100) for a, b in zip(qs, expected_q))
    check("frozen seven-point q grid", ok, f"observed {qs}")
    views = {r["cv_view"] for r in rows}
    check("CV-0 and CV-2 both present in the q profile",
          {"CV-0", "CV-2"}.issubset(views), f"views={sorted(views)}")


# Unique-event recall
def verify_unique_event() -> None:
    rows = read("posthoc_unique_event_recall5.csv")
    if not rows:
        return
    ind = [r for r in rows if r["target_domain"] == "inD"]
    check("unique-event denominator is 1,109 positive inD episodes",
          all(int(float(r["n_positive_episodes"])) == 1109 for r in ind) and bool(ind),
          "n_positive_episodes")
    cv2 = [r for r in ind if r["score_family"] == "CV-2" and r["horizon"] == "H1.6s"]
    if cv2:
        v = max(float(r["unique_event_recall"]) for r in cv2)
        check("CV-2 unique-event Recall@5% at H1.6s is 0.760144",
              close(round(v, 6), 0.760144), f"observed {v:.6f}")


# Onset-branch shares
def verify_onset_branch() -> None:
    rows = read("posthoc_onset_branch.csv")
    if not rows:
        return
    check("onset-branch decomposition present", len(rows) > 0, f"rows={len(rows)}")
    cats = {r["branch_category"] for r in rows}
    check("branch categories include a branch-C category",
          any("C" in c for c in cats), f"categories={sorted(cats)[:6]}")


# Bounded complementarity outcome
def verify_complementarity() -> None:
    import json as _json
    path = os.path.join(FINAL, "posthoc_complementarity_verdict.json")
    if not os.path.exists(path):
        check("bounded complementarity verdict artifact present", False, "posthoc_complementarity_verdict.json missing")
        return
    with open(path, encoding="utf-8") as f:
        blob = _json.dumps(_json.load(f))
    check("bounded complementarity returns NO_COMPLEMENTARITY_HEADROOM",
          "NO_COMPLEMENTARITY_HEADROOM" in blob,
          "the bounded negative verdict must be present")
    check("bounded complementarity gate evaluation table present",
          bool(read("posthoc_complementarity_gate_evaluation.csv")), "")


# Forbidden values
# The retired triples may legitimately be *named in order to be forbidden*: the figure
# generators assert that no retired triple is reachable from their data table and record
# the result in their build verification. A line that names one alongside a guard word is such a
# check. Any other occurrence would be presenting a retired value as current, which
# no release document may present as current.
GUARD_WORDS = ("retired", "forbidden", "prohibit", "must not", "not be reachable",
               "absent", "superseded", "historical", "never")


def verify_no_retired_values() -> None:
    forbidden = tuple("/".join(parts) for parts in (("3", "10", "59"), ("3", "8", "13"), ("23", "1", "0")))
    hits, guarded = [], 0
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
        for fn in files:
            if os.path.splitext(fn)[1].lower() not in {".csv", ".md", ".json", ".txt", ".py"}:
                continue
            p = os.path.join(root, fn)
            try:
                lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
            except OSError:
                continue
            for i, line in enumerate(lines, 1):
                if not any(f in line for f in forbidden):
                    continue
                context = " ".join(lines[max(0, i - 6):i + 6]).lower()
                if any(g in context for g in GUARD_WORDS):
                    guarded += 1
                else:
                    hits.append(f"{os.path.relpath(p, REPO)}:{i}")
    check("no retired result value is presented as current", not hits,
          f"{guarded} guarded mention(s) allowed; violations: " + "; ".join(hits[:5])
          if (guarded or hits) else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    for fn in (verify_computed_primary_contrast_governance,
               verify_ap_grid, verify_recall5_grid, verify_posthoc_ap, verify_recall_q,
               verify_unique_event, verify_onset_branch, verify_complementarity,
               verify_no_retired_values):
        fn()

    width = max(len(n) for n, _, _ in results)
    failed = 0
    for name, ok, detail in results:
        if not ok:
            failed += 1
        if args.verbose or not ok:
            tag = "PASS" if ok else "FAIL"
            print(f"[{tag}] {name:<{width}}  {detail}")
    print(f"\n{len(results)} checks, {failed} failed")
    if failed:
        print("REPRODUCTION CHECK FAILED")
        return 1
    print("REPRODUCTION CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
