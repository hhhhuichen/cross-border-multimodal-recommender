# ACMR Research Code


The release contains the model, independent baselines, data-contract and
evaluation code, data-collection/build pipelines, and regression tests. Large
datasets, image/text features, checkpoints, generated result files, and local caches are intentionally excluded from this source release.

## Requirements

Python 3.10+ and the packages listed in [`requirements.txt`](requirements.txt).
PyTorch should be installed for the target CPU/GPU environment first, then:

```bash
python -m pip install -r requirements.txt
```

## Quick verification

The schema/model tests use synthetic fixtures and do not require the research
datasets:

```bash
python -m pytest -q
```

The same tests can be run directly when pytest is unavailable:

```bash
python test_data_schema_v2.py
python test_model_innovations.py
python test_experiment_validation.py
python test_smoke.py
python test_xmarket_schema_v2.py
```

## Synthetic smoke run

```bash
python train.py \
  --data synthetic \
  --model acmr \
  --residual none \
  --split-seed 20260801 \
  --train-seed 20260901 \
  --eval-k 10 20 \
  --market-aggregate macro \
  --ckpt-path /tmp/acmr-b0.pt \
  --result-path /tmp/acmr-b0.json
```

The independent baselines and ACMR ablations use the same entry point:

```bash
python train.py --model bpr_mf
python train.py --model lightgcn
python train.py --model vbpr
python train.py --model acmr --residual fused
python train.py --model acmr --residual decoupled
python train.py --model acmr --residual market_reliable
```

## Source layout

- `model.py`, `layers.py`: ACMR model and graph/multimodal layers.
- `baselines.py`: BPR-MF, standard LightGCN, and VBPR baselines.
- `data_contract.py`, `data_utils.py`: schema-v2 validation and data loading.
- `evaluation.py`, `experiment_utils.py`: full-catalog evaluation and result
  manifests.
- `off_pipeline/`: Open Food Facts collection and feature/data construction.
- `xmarket_pipeline/`: cross-market Electronics data construction.
- `research/`: reproducibility, screening, and experiment-validation scripts.
- `test_*.py`: regression and smoke tests.

## Dataset policy

No raw or processed dataset is bundled here. Before running a real-data
experiment, build the corresponding dataset with the pipeline scripts and
keep it outside version control (the default paths are ignored by Git). Use
the exact split seeds, train seeds, preprocessing settings, and feature-model
versions reported in the paper. Do not treat synthetic-user or simulated
feedback results as evidence of real purchasing behavior.

## Reproducibility

`train.py` writes an atomic JSON result manifest containing configuration,
seeds, data/source hashes, parameter counts, and validation/test metrics.
Preserve these manifests together with the code commit used for an experiment.

