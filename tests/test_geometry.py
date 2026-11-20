"""Tests for the HIGH proxy geometry and the analytic constant-velocity comparator.

These run with no dataset and exercise the frozen constant-velocity reference implementation.
"""
import numpy as np
import pytest

from criterion_switch import geometry as G


def arr(*v):
    return np.asarray(v, dtype=np.float64)


# Frozen constants
def test_frozen_constants_match_the_published_contract():
    assert G.CADENCE_SECONDS == 0.4
    assert G.DISTANCE_HIGH_METRES == 1.0
    assert G.DISTANCE_CONFLICT_METRES == 1.0
    assert G.TCA_HIGH_SECONDS == 1.5
    assert G.CPA_HIGH_METRES == 1.5
    assert G.PROJECTION_HORIZON_SECONDS == 1.0


def test_high_and_conflict_distances_are_equal_but_distinct_roles():
    # Section 3.4 states the two thresholds are numerically equal yet keep separate
    # current- and projected-separation roles. A change to either must be deliberate.
    assert G.DISTANCE_HIGH_METRES == G.DISTANCE_CONFLICT_METRES


# HIGH branch A
def test_branch_a_fires_when_current_separation_is_below_threshold():
    g = G.independent_high_geometry(arr(0.5), arr(0.0), arr(0.0), arr(0.0), arr(True))
    assert g.distance_m[0] == pytest.approx(0.5)
    assert bool(g.high_current_distance[0])
    assert bool(g.is_high[0])


def test_branch_a_does_not_fire_just_above_threshold():
    g = G.independent_high_geometry(arr(1.001), arr(0.0), arr(0.0), arr(0.0), arr(True))
    assert not bool(g.high_current_distance[0])


# TCA sign convention
def test_tca_is_positive_for_an_approaching_pair():
    # Closing head-on along x: separation 10 m, closing at 1 m/s.
    g = G.independent_high_geometry(arr(10.0), arr(0.0), arr(-1.0), arr(0.0), arr(True))
    assert bool(g.tca_valid[0])
    assert g.tca_seconds[0] == pytest.approx(10.0)


def test_tca_is_negative_when_closest_approach_is_already_past():
    # Separating: the closest approach lies in the past, so TCA is negative. This is the
    # sign convention Section 3.3 of the paper states explicitly.
    g = G.independent_high_geometry(arr(10.0), arr(0.0), arr(1.0), arr(0.0), arr(True))
    assert bool(g.tca_valid[0])
    assert g.tca_seconds[0] < 0.0


def test_cpa_equals_current_distance_for_purely_radial_motion():
    g = G.independent_high_geometry(arr(5.0), arr(0.0), arr(-1.0), arr(0.0), arr(True))
    assert g.cpa_m[0] == pytest.approx(0.0, abs=1e-9)


def test_cpa_equals_perpendicular_offset_for_passing_motion():
    # Moving in -x at unit speed with a fixed 2 m lateral offset: CPA is that offset.
    g = G.independent_high_geometry(arr(10.0), arr(2.0), arr(-1.0), arr(0.0), arr(True))
    assert g.cpa_m[0] == pytest.approx(2.0, abs=1e-9)


# Invalid rows
def test_invalid_kinematics_disable_the_projection_and_approach_branches():
    # Section 3.4: without a valid finite-difference velocity the projection and
    # closest-approach branches are inactive; only current separation can fire.
    g = G.independent_high_geometry(arr(0.5), arr(0.0), arr(np.nan), arr(np.nan),
                                    arr(False))
    assert not bool(g.tca_valid[0])
    assert not bool(g.high_projected_distance[0])
    assert not bool(g.high_tca_cpa[0])
    assert bool(g.high_current_distance[0])


def test_degenerate_relative_speed_is_treated_as_invalid():
    g = G.independent_high_geometry(arr(5.0), arr(0.0), arr(0.0), arr(0.0), arr(True))
    assert not bool(g.tca_valid[0])


# ACV / ANALYTIC_CV scoring
def test_acv_score_is_the_negated_projected_onset_time():
    s = G.cv_high_score(arr(10.0), arr(0.0), arr(-1.0), arr(0.0), arr(True), 3.2)
    assert s.score[0] == pytest.approx(-s.time_to_first_high_seconds[0])


def test_acv_ranks_an_earlier_projected_onset_higher():
    # Two approaching pairs, the second closing faster, so it reaches HIGH sooner and
    # must receive the larger score.
    slow = G.cv_high_score(arr(3.0), arr(0.0), arr(-1.0), arr(0.0), arr(True), 3.2)
    fast = G.cv_high_score(arr(3.0), arr(0.0), arr(-4.0), arr(0.0), arr(True), 3.2)
    assert fast.score[0] > slow.score[0]


def test_acv_uses_the_fail_closed_sentinel_when_no_onset_is_projected():
    # A separating pair never reaches the geometry within the horizon.
    H = 1.6
    s = G.cv_high_score(arr(20.0), arr(0.0), arr(5.0), arr(0.0), arr(True), H)
    assert not bool(s.predicted_high_at_any_step[0])
    assert s.time_to_first_high_seconds[0] == pytest.approx(H + G.CADENCE_SECONDS)


def test_acv_fails_closed_on_invalid_geometry():
    H = 1.6
    s = G.cv_high_score(arr(0.2), arr(0.0), arr(np.nan), arr(np.nan), arr(False), H)
    assert not bool(s.predicted_high_at_any_step[0])
    assert s.time_to_first_high_seconds[0] == pytest.approx(H + G.CADENCE_SECONDS)


def test_acv_projects_on_the_frozen_cadence_grid():
    s = G.cv_high_score(arr(10.0), arr(0.0), arr(-1.0), arr(0.0), arr(True), 3.2)
    t = s.time_to_first_high_seconds[0]
    assert t == pytest.approx(round(t / G.CADENCE_SECONDS) * G.CADENCE_SECONDS, abs=1e-12)


def test_acv_rejects_a_horizon_off_the_cadence_grid():
    with pytest.raises(Exception):
        G.cv_high_score(arr(1.0), arr(0.0), arr(-1.0), arr(0.0), arr(True), 1.5)


def test_acv_score_is_vectorised_and_order_independent():
    rx = arr(10.0, 3.0, 20.0)
    ry = arr(0.0, 0.0, 0.0)
    vx = arr(-1.0, -4.0, 5.0)
    vy = arr(0.0, 0.0, 0.0)
    ok = np.asarray([True, True, True])
    s = G.cv_high_score(rx, ry, vx, vy, ok, 3.2)
    idx = [2, 0, 1]
    s2 = G.cv_high_score(rx[idx], ry[idx], vx[idx], vy[idx], ok[idx], 3.2)
    assert np.allclose(s.score[idx], s2.score)


def test_both_horizons_of_the_paper_are_accepted():
    for H in (1.6, 3.2):
        s = G.cv_high_score(arr(2.0), arr(0.0), arr(-1.0), arr(0.0), arr(True), H)
        assert np.isfinite(s.score[0])
