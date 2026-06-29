# Submission Reproducibility Guide

This folder contains instructions and helper scripts for reproducing the results in this branch.

## 1) Environment

From the project root directory `trace-bottleneck`:

```bash
conda env create -f environment.yml
conda activate c2bm
```

If your environment name differs, replace `c2bm` accordingly.

## 2) Connect LUMIERE Data Source

The project uses three data paths:

- raw scan-pair metadata root (`dataset.loader.root_dir`)
- processed image root (`dataset.loader.img_dir`)
- preprocessed cache directory (`dataset.loader.case_cache_dir`)

Export them once per shell:

```bash
export LUMIERE_ROOT=/abs/path/to/Lumiere
export LUMIERE_PROCESSED=/abs/path/to/processed_images
export LUMIERE_CACHE=/abs/path/to/cached_data_MNI
```

Optional helper (creates/updates shell exports in your current shell):

```bash
source trace-bottleneck/submission/scripts/set_data_env.sh \
  /abs/path/to/Lumiere \
  /abs/path/to/processed_images \
  /abs/path/to/cached_data_MNI
```

## 3) Main 5-Fold Training Run (TRACE)

```bash
python main.py \
  dataset=lumiere \
  model=c2bm \
  dataset.loader.graph_type=expert \
  dataset.loader.root_dir="$LUMIERE_ROOT" \
  dataset.loader.img_dir="$LUMIERE_PROCESSED" \
  dataset.loader.case_cache_dir="$LUMIERE_CACHE" \
  trainer.devices=1 \
  trainer.precision=bf16-mixed
```

Output is written under `outputs/YYYY-MM-DD/HH-MM-SS`.

## 4) Reproduce Added Ablations and Paper Artifacts

Assume `RUN_DIR=outputs/<date>/<time>` from the completed run.

### 4.1 Rule-based oracle on GT concepts

```bash
python rule_based_rano_oracle.py --run-dir "$RUN_DIR" --out-json paper_figures/rule_based_oracle_results.json
```

### 4.2 Rule-based oracle on predicted concepts

```bash
python predicted_concept_rano_oracle.py --run-dir "$RUN_DIR" --out-json paper_figures/predicted_concept_oracle_results.json
```

### 4.3 Ordered vs random intervention policy

```bash
python intervention_policy_ablation.py \
  --run-dir "$RUN_DIR" \
  --k-max 6 \
  --num-random 6 \
  --eval-batch-size 2 \
  --out-json paper_figures/intervention_policy_ablation_k6_r6.json \
  --out-csv paper_figures/intervention_policy_ablation_k6_r6.csv
```

### 4.4 Clinical time-gap CaCE recomputation (6 to 16 weeks)

```bash
python recompute_cace_timegap_clinical.py \
  --run-dir "$RUN_DIR" \
  --timegap-low-weeks 6 \
  --timegap-high-weeks 16 \
  --split val \
  --eval-batch-size 4 \
  --out-dir paper_figures
```

## 5) Notes

- Keep `dataset.loader.task_cardinality=4` for the reported main results.
- For GPU selection, prepend `CUDA_VISIBLE_DEVICES=<id>` to commands.

## 6) One-Patient Visit Trajectory Notebook

Notebook path:

```bash
submission/notebooks/patient_visit_trajectory.ipynb
```

It visualizes one patient from visit 1 to visit n with:
- follow-up enhancing/non-enhancing burden trends,
- follow-up SPD trend,
- percent-delta concept trends,
- RANO label trajectory,
- a single-visit transparent contribution breakdown,
- the expert causal DAG (paper Figure 7) annotated with the per-visit explanation.

Run it from the `trace-bottleneck` root:

```bash
jupyter notebook submission/notebooks/patient_visit_trajectory.ipynb
```

You can change `SELECTED_PATIENT_ID` inside the notebook to inspect different patients.

Full code and additional materials will be published upon acceptance.
