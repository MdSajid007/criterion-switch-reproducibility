# Reproducibility levels

| Level | Scope | Status in this repository |
|---|---|---|
| 1 | Recompute manuscript checks from aggregate and contrast-level result artifacts | Supported: one command, 33 checks |
| 2 | Re-run evaluation from prepared trajectory-derived inputs | Not supported: those inputs are not distributed |
| 3 | Reconstruct and run from original source corpora | Possible in principle, unverified here, and not guaranteed bit-exact across environments |

The Level 1 verifier uses Python's standard library. NumPy is required by the standalone geometry and retrieval modules and their tests. No data download, network connection, accelerator, or project-relative environment variable is required.

The source in this repository is the standalone method core needed to test HIGH geometry, ACV scoring, and tie-neutral retrieval. The full source-data-dependent training and evaluation pipeline is not included, and a complete end-to-end rerun has not been verified outside the study environment.
