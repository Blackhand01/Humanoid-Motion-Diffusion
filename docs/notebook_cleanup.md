# Notebook Cleanup

Primary notebook:

- `training_tutorial.ipynb`: Kaggle production tutorial for full AIST++ training, EMA/CFG showcase generation, and single ZIP export.

Removed obsolete notebooks:

- `phase1_aist_kaggle.ipynb`: superseded by the package-level AIST++ loader and `configs/testing.yaml`/`configs/kaggle_prod.yaml`.
- `test_evaluation_rigor.ipynb`: superseded by unit tests in `tests/test_evaluation_metrics.py` and `tests/test_failure_analysis.py`.
- `main_colab.ipynb`: superseded by the single Kaggle production tutorial.

These notebooks are no longer production entrypoints and should not be restored.
