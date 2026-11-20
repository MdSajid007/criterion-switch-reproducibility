"""Standalone frozen HIGH geometry and analytic constant-velocity comparator.

This module contains no fitted parameter, outcome table, or source trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CADENCE_SECONDS = 0.4
PROJECTION_HORIZON_SECONDS = 1.0
DISTANCE_HIGH_METRES = 1.0
DISTANCE_CONFLICT_METRES = 1.0
TCA_HIGH_SECONDS = 1.5
CPA_HIGH_METRES = 1.5
RELATIVE_SPEED_SQUARED_EPSILON = 1e-9

CV_SCORE_INPUT_COLUMNS = (
    "relative_x_m",
    "relative_y_m",
    "relative_vx_mps",
    "relative_vy_mps",
    "kinematic_pair_valid",
)


@dataclass(frozen=True)
class Geometry:
    distance_m: np.ndarray
    tca_seconds: np.ndarray
    tca_valid: np.ndarray
    cpa_m: np.ndarray
    projected_min_distance_m: np.ndarray
    high_current_distance: np.ndarray
    high_projected_distance: np.ndarray
    high_tca_cpa: np.ndarray
    is_high: np.ndarray


@dataclass(frozen=True)
class CVScore:
    score: np.ndarray
    time_to_first_high_seconds: np.ndarray
    first_high_step: np.ndarray
    predicted_high_at_any_step: np.ndarray


def independent_high_geometry(
    relative_x_m: np.ndarray,
    relative_y_m: np.ndarray,
    relative_vx_mps: np.ndarray,
    relative_vy_mps: np.ndarray,
    kinematic_pair_valid: np.ndarray,
) -> Geometry:
    """Reimplement the frozen HIGH equations without calling production code."""
    rx = np.asarray(relative_x_m, dtype=np.float64)
    ry = np.asarray(relative_y_m, dtype=np.float64)
    vx = np.asarray(relative_vx_mps, dtype=np.float64)
    vy = np.asarray(relative_vy_mps, dtype=np.float64)
    valid = np.asarray(kinematic_pair_valid, dtype=bool)

    distance = np.hypot(rx, ry)
    speed_squared = vx * vx + vy * vy
    dot = rx * vx + ry * vy
    finite_velocity = np.isfinite(vx) & np.isfinite(vy)
    tca_valid = (
        valid
        & finite_velocity
        & np.isfinite(speed_squared)
        & (speed_squared > RELATIVE_SPEED_SQUARED_EPSILON)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        tca = np.where(tca_valid, -dot / speed_squared, np.nan)
        clipped = np.where(
            tca_valid,
            np.clip(tca, 0.0, PROJECTION_HORIZON_SECONDS),
            np.nan,
        )
        projected = np.where(
            tca_valid,
            np.hypot(rx + clipped * vx, ry + clipped * vy),
            np.nan,
        )
        cpa = np.where(
            tca_valid,
            np.hypot(rx + tca * vx, ry + tca * vy),
            np.nan,
        )

    branch_1 = distance <= DISTANCE_HIGH_METRES
    branch_2 = tca_valid & (projected <= DISTANCE_CONFLICT_METRES)
    branch_3 = (
        tca_valid
        & (tca >= 0.0)
        & (tca <= TCA_HIGH_SECONDS)
        & (cpa <= CPA_HIGH_METRES)
    )
    high = branch_1 | branch_2 | branch_3
    return Geometry(
        distance_m=distance,
        tca_seconds=tca,
        tca_valid=tca_valid,
        cpa_m=cpa,
        projected_min_distance_m=projected,
        high_current_distance=branch_1,
        high_projected_distance=branch_2,
        high_tca_cpa=branch_3,
        is_high=high,
    )


def cv_high_score(
    relative_x_m: np.ndarray,
    relative_y_m: np.ndarray,
    relative_vx_mps: np.ndarray,
    relative_vy_mps: np.ndarray,
    kinematic_pair_valid: np.ndarray,
    screening_horizon_seconds: float,
) -> CVScore:
    """Return the pre-specified score: negative time to first predicted HIGH.

    States are analytically extrapolated at 0.4-second grid instants.  If
    current velocity geometry is unavailable, no extrapolated state is treated
    as available and the row receives the no-predicted-onset score.
    """
    rx = np.asarray(relative_x_m, dtype=np.float64)
    ry = np.asarray(relative_y_m, dtype=np.float64)
    vx = np.asarray(relative_vx_mps, dtype=np.float64)
    vy = np.asarray(relative_vy_mps, dtype=np.float64)
    valid = np.asarray(kinematic_pair_valid, dtype=bool)

    ratio = float(screening_horizon_seconds) / CADENCE_SECONDS
    number_of_steps = int(round(ratio))
    if not np.isclose(ratio, number_of_steps, atol=1e-12, rtol=0.0):
        raise ValueError("Screening horizon is not on the frozen 0.4-second grid")
    if number_of_steps < 1:
        raise ValueError("Screening horizon must include at least one grid step")

    sentinel_time = float(screening_horizon_seconds) + CADENCE_SECONDS
    first_time = np.full(rx.shape, sentinel_time, dtype=np.float64)
    first_step = np.full(rx.shape, number_of_steps + 1, dtype=np.int64)
    found = np.zeros(rx.shape, dtype=bool)

    for step in range(1, number_of_steps + 1):
        elapsed = step * CADENCE_SECONDS
        predicted_rx = rx + elapsed * vx
        predicted_ry = ry + elapsed * vy
        geometry = independent_high_geometry(
            predicted_rx,
            predicted_ry,
            vx,
            vy,
            valid,
        )
        # A relative state cannot be extrapolated when its velocity is invalid.
        # This fail-closed guard affects only extrapolated states; branch 1 of
        # independent_high_geometry remains velocity-independent for validation
        # against observed current states.
        hit = valid & geometry.is_high & ~found
        first_time[hit] = elapsed
        first_step[hit] = step
        found |= hit

    return CVScore(
        score=-first_time,
        time_to_first_high_seconds=first_time,
        first_high_step=first_step,
        predicted_high_at_any_step=found,
    )
