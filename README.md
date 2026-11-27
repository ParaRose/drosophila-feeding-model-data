# Cross-connectome transfer of a Drosophila spiking feeding model: proofreading stability, reconstruction shifts, and homolog calibration

Supporting dataset for the V6 manuscript by Zihao Peng and Zhuoye Li (release `v6.0`; release `v5.0` is retained for provenance). Files in `data/` match the submission supplements. `S1_Text.pdf` describes the protocols and expanded disinhibition details; `S1_Fig.tif` and `S1_Table`–`S7_Table` report the supporting results.

| Machine-readable file | Contents |
| --- | --- |
| `data/S1_Table.tsv` | Behavioral labels and simulated MN9 responses by cell type and connectome |
| `data/S3_Table.tsv` | Rewired graph controls and left/right MN9 readouts |
| `data/S5_Table.tsv` | Candidate-specific local stimulation tests |
| `data/S7_Table.tsv` | Cross-connectome motif comparisons and shared intermediates |

The input connectomes are distributed by their original providers: [FlyWire](https://codex.flywire.ai/), [BANC v888](https://doi.org/10.7910/DVN/7WTH1N), and [MaleCNS](https://neuprint.janelia.org/). The present snapshot contains study outputs. `SHA256SUMS` lists checksums for all 28 archived files (`data/`, `results/`, and `code/`).

## Changes in v6.0 (retaining v5.0 for provenance)

- Rebuilt `data/S2_Table.pdf` (S2A main and threshold/denominator tables) directly from archived float64 arrays in `results/` with uniform 100,000-draw seed-42 stratified bootstrap 95% CIs, fixing S2A cell discrepancies and documenting the six-candidate BANC selection pool and separate pure cell-type check.
- Updated `data/S1_Text.pdf` (Part B, Section 3) to report unconstrained brain-side SELK activations alongside VNC-targeted suppression and fix phrasing in the premotor summary.
- Updated `data/S4_Table.pdf` and `data/S6_Table.pdf` headers/titles to match the V6 manuscript (`secondary baseline comparisons`).
- Added `code/verify_s2a_from_archived_jsons.py` to independently reproduce every S2A metric directly from `results/*.json`.
- All four machine-readable TSV files (`S1_Table.tsv`, `S3_Table.tsv`, `S5_Table.tsv`, `S7_Table.tsv`), `S1_Fig.tif`, and all eight simulation JSON files in `results/` are byte-identical to `v5.0`.

## Simulation results

The `results/` directory contains the eight study result JSON files, including the pre-submission calibration experiments and V5/V6 reviewer controls.

## Reproduce the published figures, TSV tables, and S2A verification

From the repository root, run the following commands in a Python 3.11+ environment (requires no raw connectome downloads):

```sh
python -m pip install numpy==1.26.4 scipy==1.11.4 matplotlib==3.8.0
python -c "from pathlib import Path; import shutil; d=Path('reproduce-v6'); d.mkdir(exist_ok=True); [shutil.copy2(p,d) for p in [*Path('code').glob('*.py'),*Path('results').glob('*.json')]]"
PYTHONHASHSEED=0 python reproduce-v6/english_figures.py
python code/verify_s2a_from_archived_jsons.py
```

The generated `supp_table1`, `supp_table2`, `supp_table3`, and `supp_table4` correspond respectively to `data/S1_Table.tsv`, `data/S5_Table.tsv`, `data/S7_Table.tsv`, and `data/S3_Table.tsv` (verified byte-identical).

## Run simulations from connectome inputs

The simulation scripts in `code/` correspond to the FlyWire v630 benchmark (`shiu_mn9_benchmark.py`), v783 transfer (`mn9_transfer_benchmark.py`), BANC (`mn9_banc_benchmark.py`), MaleCNS (`mn9_malecns_benchmark.py`), background-state/disinhibition scans (`mn9_disinhibition_state.py`, GPU version `mn9_disinhibition_colab_gpu.py`), and follow-up controls (`mn9_portability_hardening.py`, `mn9_pre_submission_ab.py`, `mn9_v5_reviewer_controls.py`). Local CPU stages read `PROJC_CACHE_DIR`; cloud stages read `RESEARCH_CACHE_DIR`. Raw connectomes are not bundled.
