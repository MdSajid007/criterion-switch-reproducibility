# Leakage assurance report

Five categories are covered by the frozen design checks: future-information exclusion,
label exclusion, identifier and provenance exclusion, target-label-free normalization,
and train/target keyset disjointness. The public feature contract is a fail-closed
ten-feature allowlist; no target-domain labels are used for held-out normalization or
selection. Transfer partitions are defined at whole-domain level.

The prepared feature rows needed to repeat these source-data-dependent checks are not
redistributed. This report therefore records the frozen outcome and contract but does not
present those checks as runnable Level 1 tests.
