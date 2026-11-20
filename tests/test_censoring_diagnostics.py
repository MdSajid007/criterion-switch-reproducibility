"""Integrity test for the released censoring state diagnostics.

Appendix I of the paper cites `results/final/censoring_state_diagnostics.csv` as the
evidence that censoring is conditioned on episode state. This test asserts the shape
of that table and the two directional facts the appendix quotes, so a plain ``pytest``
run fails if the released table drifts from the printed claim.
"""
import csv
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(REPO, "results", "final", "censoring_state_diagnostics.csv")

DOMAINS = {"ETH", "UCY", "inD"}
HORIZONS = {"1.6", "3.2"}


def rows():
    with open(PATH, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def test_file_exists():
    assert os.path.isfile(PATH)


def test_twelve_rows_covering_six_populations_by_status():
    r = rows()
    assert len(r) == 12
    keys = {(x["source_dataset"], x["horizon_seconds"], x["status"]) for x in r}
    assert len(keys) == 12
    assert {k[0] for k in keys} == DOMAINS
    assert {k[1] for k in keys} == HORIZONS
    assert {k[2] for k in keys} == {"eligible", "right_censored"}


def test_only_confirmed_negatives_are_censored():
    """The implemented rule censors NO_HIGH rows only, so every censored
    stratum has an incident share of exactly zero."""
    for x in rows():
        if x["status"] == "right_censored":
            assert float(x["incident_share"]) == 0.0


@pytest.mark.parametrize("domain", sorted(DOMAINS))
@pytest.mark.parametrize("horizon", sorted(HORIZONS))
def test_state_contrast_direction_is_the_same_in_all_six_populations(domain, horizon):
    """Appendix I quotes an approaching share of 0.600-0.693 for eligible rows
    against 0.188-0.444 for censored rows, and a smaller mean current separation
    for eligible rows, in the same direction in all six populations."""
    sel = {x["status"]: x for x in rows()
           if x["source_dataset"] == domain and x["horizon_seconds"] == horizon}
    elig, cens = sel["eligible"], sel["right_censored"]

    a_e, a_c = float(elig["approaching_share"]), float(cens["approaching_share"])
    assert a_e > a_c
    # the appendix prints these ranges rounded, so compare the rounded values
    assert 0.600 <= round(a_e, 3) <= 0.693
    assert 0.188 <= round(a_c, 3) <= 0.444

    d_e, d_c = float(elig["mean_distance_m"]), float(cens["mean_distance_m"])
    assert d_e < d_c
    assert 4.16 <= round(d_e, 2) <= 4.81
    assert 4.49 <= round(d_c, 2) <= 6.02
    assert float(elig["median_distance_m"]) < float(cens["median_distance_m"])
