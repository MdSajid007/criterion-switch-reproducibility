# Criterion Dependence: Code and Results

This repository accompanies the manuscript *Criterion Dependence in Learned-versus-Analytic Model Comparison Under Fixed Predictions*. It provides the standalone HIGH geometry, analytic constant-velocity (ACV) score, tie-neutral retrieval rule, aggregate and contrast-level results, figure source data, and checks for the reported summaries.

## Verify the reported results

Use Python 3.11 or newer. The pinned NumPy version requires Python 3.11 or newer.

```text
python -m pip install -r requirements.txt
python scripts/verify_paper_results.py --verbose
python -m pytest -q
```

The verifier should finish with `33 checks, 0 failed` and `REPRODUCTION CHECK PASSED`. The test suite should report 68 passing tests. The verifier uses only the Python standard library; the standalone methods and tests use NumPy and pytest.

## Contents and scope

- `src/criterion_switch/geometry.py`: standalone HIGH geometry and ACV score.
- `src/criterion_switch/retrieval.py`: fixed-fraction, tie-neutral retrieval rule.
- `scripts/verify_paper_results.py`: checks the reported summaries against the released result tables.
- `results/final/`, `supplement_tables/`, and `figure_data/`: aggregate and contrast-level results and plotted values.
- `FROZEN_METHOD_CONTRACT.md`, `REPRODUCIBILITY.md`, and `DATA_LICENSE_BOUNDARY.md`: method and access boundaries.

This is a Level 1 release: it recomputes reported summaries from the distributed tables. It does not include the full source-data-dependent training and evaluation pipeline or the prepared per-observation inputs. See `REPRODUCIBILITY.md` for what can and cannot be rerun.

## Source datasets

No trajectory corpus is redistributed here. Obtain each dataset from its provider and follow its terms:

- ETH BIWI Walking Pedestrians annotations: https://ethz.ch/content/dam/ethz/special-interest/itet/cvl/vision-dam/documents/ewap_dataset_light.tgz
- UCY Crowds-by-Example data: https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data.zip
- inD access request and terms: https://levelxdata.com/ind-dataset/

The inD provider reviews access requests and limits free use to non-commercial purposes. The row-level derived tables are not distributed; see `DATA_LICENSE_BOUNDARY.md`.

## Rights

No public software or documentation licence has been applied to this repository. Contact the rights holders for reuse permissions. The source datasets have separate provider terms.
