"""Tests for the tie-neutral fixed-fraction retrieval rule.

The frozen retrieval size is K = max(1, floor(q*N)) -- the FLOOR convention, matching
the manuscript and every row of the released q-profile.  The rule is the frozen
tie-boundary correction: where the top-q boundary falls inside a block of equal scores,
the whole block is included fractionally so the selected mass is exactly K and no
arbitrary row ordering can change the metric.
"""
import numpy as np
import pytest

from criterion_switch.retrieval import TieNeutralView


def k_five_percent(n):
    return max(1.0, np.floor(0.05 * n))


def view(y, score, cluster=None):
    y = np.asarray(y, dtype=np.float64)
    if cluster is None:
        cluster = np.zeros(len(y), dtype=np.int64)
    return TieNeutralView(y, np.asarray(score, dtype=np.float64), cluster,
                          int(np.max(cluster)) + 1)


# Basic correctness
def test_selected_mass_is_exactly_k():
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.1).astype(float)
    s = rng.random(200)
    r = view(y, s).full_sample(k_five_percent)
    assert r["inclusion_mass"] == pytest.approx(r["k"], abs=1e-9)


def test_recall_is_one_when_every_positive_sits_at_the_head():
    n = 100
    y = np.zeros(n); y[:5] = 1.0
    s = np.linspace(1.0, 0.0, n)          # strictly decreasing, no ties
    r = view(y, s).full_sample(k_five_percent)
    assert r["k"] == pytest.approx(5.0)
    assert r["recall_at_5pct"] == pytest.approx(1.0)


def test_recall_is_zero_when_every_positive_sits_at_the_tail():
    n = 100
    y = np.zeros(n); y[-5:] = 1.0
    s = np.linspace(1.0, 0.0, n)
    r = view(y, s).full_sample(k_five_percent)
    assert r["recall_at_5pct"] == pytest.approx(0.0)


# The tie-neutral core
def test_a_single_tied_block_gives_the_exact_expected_share():
    # Every score identical: the whole population is one tie block, so the selected
    # fraction of positives is exactly K/n regardless of ordering.
    n, pos = 100, 20
    y = np.zeros(n); y[:pos] = 1.0
    s = np.full(n, 0.7)
    r = view(y, s).full_sample(k_five_percent)
    assert r["n_tie"] == pytest.approx(float(n))
    assert r["alpha"] == pytest.approx(r["k"] / n)
    assert r["recall_at_5pct"] == pytest.approx(r["k"] / n)


def test_result_is_invariant_to_ordering_within_a_tied_block():
    rng = np.random.default_rng(7)
    n = 240
    y = (rng.random(n) < 0.15).astype(float)
    s = np.repeat(np.linspace(1.0, 0.0, 12), n // 12)   # 12 large tie blocks
    base = view(y, s).full_sample(k_five_percent)["recall_at_5pct"]
    for seed in range(5):
        p = np.random.default_rng(seed).permutation(n)
        got = view(y[p], s[p])._fixed_fraction(np.ones(n), k_five_percent)["recall_at_5pct"]
        assert got == pytest.approx(base, abs=1e-12)


def test_tie_bounds_bracket_the_tie_neutral_value():
    # The attainable min and max under adversarial within-block ordering must bracket
    # the tie-neutral answer. This is the tie-boundary diagnostic that motivated the rule.
    rng = np.random.default_rng(3)
    n = 300
    y = (rng.random(n) < 0.2).astype(float)
    s = np.repeat(np.linspace(1.0, 0.0, 6), n // 6)
    v = view(y, s)
    lo, hi = v.tie_bounds(k_five_percent)
    val = v.full_sample(k_five_percent)["recall_at_5pct"]
    assert lo - 1e-12 <= val <= hi + 1e-12


def test_tie_bounds_collapse_when_there_are_no_ties():
    n = 100
    y = np.zeros(n); y[:10] = 1.0
    s = np.linspace(1.0, 0.0, n)
    v = view(y, s)
    lo, hi = v.tie_bounds(k_five_percent)
    assert lo == pytest.approx(hi, abs=1e-12)


# K definition
def test_k_never_falls_below_one():
    y = np.array([1.0, 0.0, 0.0])
    s = np.array([0.9, 0.5, 0.1])
    r = view(y, s).full_sample(k_five_percent)
    assert r["k"] >= 1.0


def test_precision_and_prevalence_are_consistent_with_recall():
    rng = np.random.default_rng(11)
    n = 400
    y = (rng.random(n) < 0.25).astype(float)
    s = rng.random(n)
    r = view(y, s).full_sample(k_five_percent)
    assert r["prevalence"] == pytest.approx(y.sum() / n)
    assert r["precision_at_5pct"] == pytest.approx(r["effective_tp_at_K"] / r["k"])
    assert r["recall_at_5pct"] == pytest.approx(r["effective_tp_at_K"] / y.sum())
    assert r["alert_lift_at_5pct"] == pytest.approx(
        r["precision_at_5pct"] / r["prevalence"])


# Cluster resampling path
def test_replicate_weights_reproduce_the_full_sample_when_all_weights_are_one():
    rng = np.random.default_rng(5)
    n, g = 300, 10
    y = (rng.random(n) < 0.2).astype(float)
    s = rng.random(n)
    cl = rng.integers(0, g, n)
    v = view(y, s, cl)
    full = v.full_sample(k_five_percent)
    rep = v.replicate(np.ones(g), k_five_percent)
    assert rep["recall_at_5pct"] == pytest.approx(full["recall_at_5pct"], abs=1e-12)


def test_replicate_preserves_exact_selected_mass():
    rng = np.random.default_rng(13)
    n, g = 300, 10
    y = (rng.random(n) < 0.2).astype(float)
    s = np.repeat(np.linspace(1, 0, 15), 20)
    cl = rng.integers(0, g, n)
    v = view(y, s, cl)
    for seed in range(4):
        mult = np.random.default_rng(seed).multinomial(g, np.full(g, 1.0 / g)).astype(float)
        r = v.replicate(mult, k_five_percent)
        assert r["inclusion_mass"] == pytest.approx(r["k"], abs=1e-9)


def test_seven_point_q_grid_is_monotone_in_recall():
    # Recall at a fixed fraction cannot decrease as the retrieval fraction grows.
    rng = np.random.default_rng(17)
    n = 2000
    y = (rng.random(n) < 0.1).astype(float)
    s = rng.random(n)
    v = view(y, s)
    prev = -1.0
    for q in (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25):
        r = v.full_sample(lambda nn, q=q: max(1.0, np.floor(q * nn)))
        assert r["recall_at_5pct"] >= prev - 1e-12
        prev = r["recall_at_5pct"]


# Floor-versus-ceiling boundary convention
# The frozen convention is K = max(1, floor(q*N)).  These regression tests pin it at
# population sizes where floor(q*N) and ceil(q*N) disagree, so a silent reversion to the
# ceil convention cannot pass the suite.  Values are chosen so q*N is non-integer.
FLOOR_CEIL_CASES = [
    # (q,     N,    floor(q*N), ceil(q*N))
    (0.05,   101,    5,   6),
    (0.05,   119,    5,   6),
    (0.05,   250,   12,  13),
    (0.01,   150,    1,   2),
    (0.02,   175,    3,   4),
    (0.10,   333,   33,  34),
    (0.15,   111,   16,  17),
    (0.20,   137,   27,  28),
    (0.25,   201,   50,  51),
]


@pytest.mark.parametrize("q,n,k_floor,k_ceil", FLOOR_CEIL_CASES)
def test_frozen_k_uses_floor_not_ceil(q, n, k_floor, k_ceil):
    """K must equal max(1, floor(q*N)) at every size where the two conventions differ."""
    assert k_floor != k_ceil, "case does not discriminate the two conventions"
    rng = np.random.default_rng(101)
    y = (rng.random(n) < 0.2).astype(float)
    s = np.linspace(1.0, 0.0, n)          # strictly decreasing: no ties
    r = view(y, s).full_sample(lambda nn, qq=q: max(1.0, np.floor(qq * nn)))
    assert r["k"] == pytest.approx(float(k_floor))
    assert r["k"] != pytest.approx(float(k_ceil))


def test_five_percent_k_fn_is_the_floor_convention():
    """The suite's shared Recall@5% K function is floor, not ceil, at a discriminating N."""
    for n, expected in ((101, 5.0), (119, 5.0), (250, 12.0), (39, 1.0)):
        assert k_five_percent(n) == pytest.approx(expected)


def test_floor_convention_never_returns_zero_below_one_over_q():
    """floor(q*N) is 0 for N < 1/q; the max(1, .) clamp is what keeps K >= 1."""
    for n in (1, 5, 19):
        assert np.floor(0.05 * n) < 1.0
        assert k_five_percent(n) == pytest.approx(1.0)


def test_recall_at_the_floor_boundary_is_strictly_below_the_ceil_boundary():
    """With one extra positive sitting exactly at rank ceil(q*N), floor and ceil disagree
    on the reported recall -- so the convention is observable, not cosmetic."""
    n, q = 101, 0.05                       # floor -> 5, ceil -> 6
    y = np.zeros(n)
    y[:6] = 1.0                            # six positives at the very head
    s = np.linspace(1.0, 0.0, n)
    v = view(y, s)
    r_floor = v.full_sample(lambda nn: max(1.0, np.floor(q * nn)))
    r_ceil = v.full_sample(lambda nn: max(1.0, np.ceil(q * nn)))
    assert r_floor["recall_at_5pct"] == pytest.approx(5.0 / 6.0)
    assert r_ceil["recall_at_5pct"] == pytest.approx(1.0)
    assert r_floor["recall_at_5pct"] < r_ceil["recall_at_5pct"]
