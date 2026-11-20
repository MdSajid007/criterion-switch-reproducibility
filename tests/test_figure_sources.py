"""Every advertised machine-readable figure source parses as one table.

This module exists because a figure source that does not parse is not
machine-readable, whatever the file extension says. It uses only the standard
library, so it runs in the Level-1 environment.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FIGURE_DATA = os.path.join(REPO, "figure_data")

EXPECTED_ROWS = {
    "figure_03_domain_classifications_data.csv": 6,
    "figure_04_contrast_grid_data.csv": 72,
    "figure_05_panel_ab_retrieval_depth.csv": 14,
    "figure_05_panel_c_unique_event.csv": 14,
    "S08_model_selection_data.csv": 18,
}


def _sources():
    return sorted(
        name for name in os.listdir(FIGURE_DATA) if name.lower().endswith(".csv")
    )


def test_every_figure_source_is_accounted_for():
    assert set(_sources()) == set(EXPECTED_ROWS)


def test_every_figure_source_is_one_rectangular_table():
    for name in _sources():
        path = os.path.join(FIGURE_DATA, name)
        with open(path, encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        assert rows, f"{name} is empty"
        header = rows[0]
        assert header, f"{name} has an empty header"
        assert len(set(header)) == len(header), f"{name} has duplicate column names"
        body = [r for r in rows[1:] if r]
        assert len(body) == len(rows) - 1, f"{name} contains a blank interior row"
        widths = {len(r) for r in body}
        assert widths == {len(header)}, (
            f"{name} is not rectangular: header width {len(header)}, row widths {widths}"
        )
        # A second header line inside the body is the specific defect this
        # module guards against.
        assert not any(r == header for r in body), f"{name} repeats its header"


def test_every_figure_source_has_the_expected_row_count():
    for name, expected in EXPECTED_ROWS.items():
        path = os.path.join(FIGURE_DATA, name)
        with open(path, encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == expected, f"{name}: {len(rows)} rows, expected {expected}"


def test_figure_five_sources_carry_the_frozen_retrieval_grid():
    path = os.path.join(FIGURE_DATA, "figure_05_panel_ab_retrieval_depth.csv")
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grid = sorted({float(r["q_pct"]) for r in rows})
    assert grid == [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    assert sorted({r["horizon"] for r in rows}) == ["H1.6s", "H3.2s"]
    counts = [int(r["n_CV_favoured_CI_LT_0"]) for r in rows if r["horizon"] == "H1.6s"]
    assert counts == [10, 12, 12, 0, 0, 0, 0]
    counts = [int(r["n_CV_favoured_CI_LT_0"]) for r in rows if r["horizon"] == "H3.2s"]
    assert counts == [11, 12, 12, 12, 0, 0, 0]


def test_figure_five_panel_c_carries_every_score_family():
    path = os.path.join(FIGURE_DATA, "figure_05_panel_c_unique_event.csv")
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    families = sorted({r["score_family"] for r in rows})
    assert families == ["CV-0", "CV-1", "CV-2", "HGB_P2", "LR_P2", "RESMLP_P2", "RF_P2"]
    assert {int(r["n_positive_episodes"]) for r in rows} == {1109}


def test_figure_three_source_matches_the_published_decomposition():
    path = os.path.join(FIGURE_DATA, "figure_03_domain_classifications_data.csv")
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    got = {
        (r["target_domain"], r["criterion"]): (
            int(r["n_learned_favored"]),
            int(r["n_inconclusive"]),
            int(r["n_comparator_favored"]),
        )
        for r in rows
    }
    assert got == {
        ("ETH", "average_precision"): (22, 2, 0),
        ("ETH", "recall_at_5pct"): (2, 8, 14),
        ("UCY", "average_precision"): (0, 24, 0),
        ("UCY", "recall_at_5pct"): (0, 2, 22),
        ("inD", "average_precision"): (21, 3, 0),
        ("inD", "recall_at_5pct"): (0, 0, 24),
    }
    assert all(int(r["n_contrasts"]) == 24 for r in rows)
