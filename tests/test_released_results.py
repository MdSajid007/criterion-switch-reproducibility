"""Integrity tests for the released final result tables.

These assert the same facts as scripts/verify_paper_results.py, so a plain ``pytest``
run also fails if a released result table drifts from the paper.
"""
import csv
import os
import subprocess
import sys
from collections import Counter

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = os.path.join(REPO, "results", "final")


def test_final_results_directory_exists():
    assert os.path.isdir(FINAL)


@pytest.mark.parametrize("name", [
    "primary_model_comparator_contrasts.csv",
    "recall5_by_domain_horizon_family.csv",
    "posthoc_ap_sensitivity.csv",
    "posthoc_recall_q_profile.csv",
    "posthoc_unique_event_recall5.csv",
    "posthoc_onset_branch.csv",
    "posthoc_complementarity_verdict.json",
])
def test_controlling_result_file_present(name):
    assert os.path.exists(os.path.join(FINAL, name)), name


def test_verify_paper_results_exits_zero():
    r = subprocess.run([sys.executable,
                        os.path.join(REPO, "scripts", "verify_paper_results.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_primary_contrast_rows_compute_to_authoritative_grids():
    path = os.path.join(FINAL, "primary_model_comparator_contrasts.csv")
    with open(path, encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

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

    identities = {}
    for metric, decompositions in expected.items():
        grid = [
            row for row in rows
            if row["metric"] == metric and row["comparator_family"] == "ANALYTIC_CV"
        ]
        identities[metric] = {(row["cell_id"], row["model_family"]) for row in grid}
        assert len(grid) == len(identities[metric]) == 72
        for domain, wanted in decompositions.items():
            subset = grid if domain == "ALL" else [
                row for row in grid if row["target_domain"] == domain
            ]
            counts = Counter(row["direction_class"] for row in subset)
            observed = (
                counts.get("CI_GT_0", 0),
                counts.get("CI_CROSSES_0", 0),
                counts.get("CI_LT_0", 0),
            )
            assert observed == wanted
            assert sum(observed) == len(subset)

    assert identities["average_precision"] == identities["recall_at_5pct"]
