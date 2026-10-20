# LPBF motion-model decision sensitivity

Computational evidence and code for **Assessing decision sensitivity to
motion-model fidelity in multi-laser additive manufacturing planning**.

Authors: Yichuan Gao, Yongxiong Chen, Kaixin Wang.
Affiliation: Defense Innovation Institute, Academy of Military Science,
Beijing 100071, China. Corresponding author: Yichuan Gao.

## Scope

This study distinguishes execution-time prediction bias from the loss caused
by selecting a different scan plan. It compares declared kinematic evaluators
on common plans and information-isolated searches. It is a computational study,
not a validation of measured scanner dynamics, thermal quality, or production
savings. Established motion and optimization tools are not claimed as new
algorithms.

The primary fixed-portfolio comparison contains 468 zero-separation settings,
80 positive selection losses, a zero median, and an 11.04% maximum.
The dense audit contains 1,723,956 settings. The server study retains all
4,680 paired searches, ten seeds, 1/5/20-second budgets and LF-favorable cases.
Seeds and parameter settings are not independent manufacturing experiments.

## Download the complete version

Use the [v1.0.0 release](https://github.com/thaigaoyc-netizen/lpbf-motion-model-decision-sensitivity/releases/tag/v1.0.0):

1. `lpbf-reproduction-v1.0.0.zip` — source, constructed inputs, attributed
   released-layer workbooks, frozen cost records, full-precision tables,
   vector figures, protocols, and public-package checks.
2. `regular_results_20261018.tar.gz` — all saved server planning runs.
3. `gpu_atlas_validated_20261018.tar.gz` — complete dense-grid audit.

The two raw archives are unchanged from the completed experiments. Their
SHA-256 digests and the source ZIP digest are in `SHA256SUMS`.
The repository's automatically generated "Source code" archive alone is not
the complete dataset; download the three named release assets.

## Verify and prepare

Place the three assets and `SHA256SUMS` in one directory:

```sh
python3 verify_downloads.py
unzip lpbf-reproduction-v1.0.0.zip
mkdir -p lpbf-reproduction/server_supplement_20261018
cp regular_results_20261018.tar.gz gpu_atlas_validated_20261018.tar.gz lpbf-reproduction/server_supplement_20261018/
cd lpbf-reproduction
python3 verify_payload.py
```

Use Python 3.12 and a new environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r paper_server_revision_20261018/requirements-check.txt
```

Ruckig 0.15.3 may require a C++17 compiler, CMake and Ninja where no wheel
exists. GPU/CUDA is not required for saved-result checks or figure generation.
Legacy platform-specific wheels, credentials, SSH launch receipts, downloaded
literature and HTTP request traces are not included.

## Recheck saved evidence (no optimizer is launched)

```sh
.venv/bin/python paper_submission_final_20261017/validate_hermite.py
.venv/bin/python paper_submission_final_20261017/recheck_numerics.py
.venv/bin/python paper_submission_final_20261017/validate_statistics.py
.venv/bin/python paper_server_revision_20261018/prepare_results.py
.venv/bin/python paper_server_revision_20261018/audit_server_results.py
```

These commands check the cubic endpoint/domain implementation, controlled
kinematic identities, frozen decision statistics, search deadlines and saved
queue costs. The last command replays the complete server records, including
finite-jerk evaluations and dense-grid checks; it is slower than the summary
checks. It does not repeat the timed searches.

## Rebuild figures from frozen data

Run in a disposable copy, since generators replace their own output figures:

```sh
.venv/bin/python paper_submission_final_20261017/retained_figures.py
.venv/bin/python paper_submission_final_20261017/figure8.py
.venv/bin/python paper_submission_final_20261017/publication_visuals.py
.venv/bin/python paper_server_revision_20261018/make_figures.py
```

Figure generators retain all cases. PDF and SVG outputs are editable vectors.
Illustrative plate thickness and shading do not represent measured specimens.
`paper_computational_retarget_20261020/data/` includes the current runtime
summaries; its figure folder contains the current rendered outputs.

## Fresh optimization is a separate experiment

`server_supplement_20261018/engine.py`, `run_experiments.py`, and `preflight.py`
preserve the actual Linux CPU-affinity-based search implementation.
The archived protocols record the original hardware, budgets and execution
order. Timed search results on another machine need not match the frozen
queues or wall times. The original deployment layout and input-archive
checks must be reproduced explicitly before running those scripts; the
verification commands above do not initiate paid compute.

The CUDA implementation and independent checks are retained in
`gpu_atlas.py`, `export_gpu_atlas.py`, `validate_gpu_atlas.py` and the archived
`code/` members. Do not mistake a saved-result replay for a new GPU experiment.

## Provenance and packaging

`ORIGINAL_FILE_PROVENANCE.json` records copied-file origins and hashes.
`PUBLIC_PACKAGING_CHANGES.json` records the sole server-audit packaging
adapter: two original plan JSON files are read directly and checked by their
original SHA-256 values rather than extracted from a historical delivery ZIP.
No numerical equation, threshold, cost record or plan is changed by it.

Historical documents retained under dated directories are frozen protocol
dependencies, not the current submission manuscript. Historical local-delivery
manifests and instructions describe their original packages; use the root
`PUBLIC_MANIFEST.json` and this README for this public release.
A successful check receipt states the specific commands and environment
actually used. It is not evidence of measured machine validation.

## Third-party inputs and license

The two unchanged workbooks are from Yuxin Yang (2024),
[Zenodo 12747089](https://doi.org/10.5281/zenodo.12747089), licensed CC BY 4.0.
The original author does not endorse this derived study.

Original research code is licensed under MIT (`LICENSE`).
Original computed results and figures are licensed under CC BY 4.0
(`DATA_LICENSE.txt`). Third-party libraries retain their own licenses and are
installed separately. Historical manuscript text is not covered by these
code/data grants.
