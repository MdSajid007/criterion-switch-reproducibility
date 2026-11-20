# S01 - Dataset admission and harmonization

Admitted corpora are ETH (2 recordings), UCY (4 recordings), and inD (31 contributing
recordings nested in four locations). DUT was not admitted because provenance was
unresolved; SDD was not admitted because image-plane coordinates require calibration to
metric space. All admitted trajectories are metric. Harmonization uses per-recording
decimation to 0.400 s without interpolation; velocity is a backward difference and is
undefined at the first observation and after a gap.

Use `results/final/target_cluster_registry.csv` for the recording roster and
`results/final/four_model_primary_metrics.csv` for cell-level counts and prevalence.
Raw and prepared trajectory rows are not included.
