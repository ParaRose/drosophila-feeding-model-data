# Transfer of a Drosophila spiking model of feeding across connectomes

Supporting dataset for the V5 manuscript by Zihao Peng and Zhuoye Li. Files in data/ match the submission supplements. S1 Text describes the protocols. S1 Figure and Tables S1–S7 report the supporting results.

| Machine-readable file | Contents |
| --- | --- |
| S1_Table.tsv | Behavioral labels and simulated MN9 responses by cell type and connectome |
| S3_Table.tsv | Rewired graph controls and left/right MN9 readouts |
| S5_Table.tsv | Candidate-specific local stimulation tests |
| S7_Table.tsv | Cross-connectome motif comparisons and shared intermediates |

The input connectomes are distributed by their original providers: [FlyWire](https://codex.flywire.ai/), [BANC v888](https://doi.org/10.7910/DVN/7WTH1N), and [MaleCNS](https://neuprint.janelia.org/). The present snapshot contains study outputs. SHA256SUMS lists the checksums of all nine supporting files.

## Simulation results

The results/ directory contains the eight study result JSON files, including the pre-submission calibration experiments and V5 reviewer controls. Source snapshot: b17701dc5fd643a30ce5c96d642061194e12ebb6.
## Reproduce the published figures and tables

From the repository root, run the following commands in a Python 3.11 environment. The figure build reads the eight archived JSON files and requires no raw connectome downloads. It writes four main SVG figures, one supplemental SVG, estimates.json and four TSVs to reproduce-v5/english_manuscript/.

~~~sh
python -m pip install numpy==1.26.4 scipy==1.11.4 matplotlib==3.8.0
python -c "from pathlib import Path; import shutil; d=Path('reproduce-v5'); d.mkdir(exist_ok=True); [shutil.copy2(p,d) for p in [*Path('code').glob('*.py'),*Path('results').glob('*.json')]]"
python reproduce-v5/english_figures.py
~~~

The generated supp_table1, supp_table2, supp_table3 and supp_table4 correspond respectively to S1, S5, S7 and S3 in data/. All four TSVs were verified field by field on Python 3.11.7.

## Run simulations from connectome inputs

The ten Python scripts in code/ are copied from the V5 analysis commit b17701dc5fd643a30ce5c96d642061194e12ebb6. The FlyWire v630 benchmark is shiu_mn9_benchmark.py. Run mn9_transfer_benchmark.py for the v783 transfer, mn9_banc_benchmark.py and mn9_malecns_benchmark.py for independent connectomes, and mn9_disinhibition_state.py for background-state tests. mn9_disinhibition_colab_gpu.py contains the GPU implementation. Subsequent controls are in mn9_portability_hardening.py, mn9_pre_submission_ab.py and mn9_v5_reviewer_controls.py. Consult each entry point for its input arguments and stage-specific options.

For simulations, prepare the versioned upstream connectivity, annotations, behavioral tables and stage-specific graph caches. The scripts use NumPy, SciPy, PyArrow and Numba; GPU stages use PyTorch. The local CPU stages read PROJC_CACHE_DIR (data/, source02/ and banc_ckpt/); cloud stages read RESEARCH_CACHE_DIR and the staging manifests referenced in the scripts. The raw connectomes, cloud caches and runtime-specific manifests are not bundled. The release verification regenerated the manuscript figures and tables from the archived outputs; full simulations on raw connectomes were not rerun.
