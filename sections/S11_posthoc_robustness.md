# S11 - Robustness and mechanism tables

The seven-point retrieval-depth profile is `results/final/posthoc_recall_q_profile.csv`.
It covers all three target domains (ETH, UCY and inD); the paper's mechanistic
interpretation is developed on the inD rows only, because only inD has adequate
recording-level support. Exact Figure 5 aggregates are in `figure_data/figure_05_panel_ab_retrieval_depth.csv`
(panels A and B) and `figure_data/figure_05_panel_c_unique_event.csv` (panel C).
Both declare their aggregation in their column names: `mean_recall_*` is the mean
over the three transfer routes into inD, `median_learned_minus_CV2` with
`min_*`/`max_*` is the median and range over the 12 contrasts, and the panel C
columns are route means. The comparator columns are route-invariant. The remaining
files are `results/final/posthoc_cv_score_views.csv`,
`results/final/posthoc_ap_sensitivity.csv`,
`results/final/posthoc_unique_event_recall5.csv`,
`results/final/posthoc_unique_event_contrasts.csv`, and
`results/final/posthoc_onset_branch.csv`.

The bounded capacity-matched conclusion is documented by
`results/final/posthoc_complementarity_verdict.json` and its gate-evaluation table.

The ACV score-level structure used for the sentinel-edge derivation is in
`results/final/posthoc_cv_score_views.csv`: `n_rows` minus `n_sentinel_rows` is
the ordered region, and the ordered-region positive mass is recovered by solving
the fractional-tie recall identity at every retrieval fraction in
`posthoc_recall_q_profile.csv` whose boundary falls inside the sentinel block.
