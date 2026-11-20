# Frozen method contract

The paper evaluates six ordered domain-transfer routes plus three pooled-source routes at
two horizons, giving 18 transfer-horizon cells. Four learned families share one ten-feature
allowlist and are compared with two restricted controls and the analytic constant-velocity
comparator.

The ten features are current pair distance, relative speed, radial closing speed,
tangential relative speed, time to closest approach, closest-approach distance, minimum
projected separation over 1.0 s, episode age, an approaching indicator, and a
geometry-valid indicator. Provenance/grouping identifiers, future or post-event
quantities, and the current HIGH indicator with its branch flags are excluded.

The analytic score is negative projected onset time on the 0.4 s cadence. Tie-neutral
Recall@q uses K=max(1, floor(q*N)); a score-equivalence class intersecting the boundary is
included fractionally. The seven-point q grid is 1%, 2%, 5%, 10%, 15%, 20%, and 25%.

Recording-level bootstrap intervals use 2,000 shared replicates and seed 20260813.
Intervals strictly above zero are learned-favored, intervals strictly below zero are
comparator-favored, and intervals touching zero are inconclusive.
