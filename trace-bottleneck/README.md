# TRACE Bottleneck

This is the anonymous repository for the submission of the paper titled
"TRACE: A Concept Bottleneck Model for Longitudinal 3D
Glioblastoma Response Assessment".

Run instructions:
- `submission/README_SUBMISSION.md`

Quick start:
1. Create environment:
     - `conda env create -f environment.yml`
     - `conda activate c2bm`
2. Set data paths with:
     - `source submission/scripts/set_data_env.sh <LUMIERE_ROOT> <LUMIERE_PROCESSED> <LUMIERE_CACHE>`
3. Run the main entrypoint:
     - `python main.py dataset=lumiere model=c2bm`

Patient-level visualization notebook:
- [submission/notebooks/patient_visit_trajectory.ipynb](submission/notebooks/patient_visit_trajectory.ipynb)

Full code and additional materials will be published upon acceptance.
