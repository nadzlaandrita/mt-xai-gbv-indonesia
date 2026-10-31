"""Runs the validation experiments (single- vs multi-target, cluster bootstrap, synthetic data).

It reuses the data preparation and helper functions of pipeline.py, then executes validate.py.
Usage (from the repository root):
    RAW_PATH=data/raw/raw_dataset.csv OUT=results/validation python src/run_validate.py
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'pipeline.py'), encoding='utf-8').read().split("results, models, best_params")[0]
exec(src)
RF_WEB_PARAMS = dict(n_estimators=400, max_depth=10, min_samples_leaf=1, max_features='sqrt')
exec(open(os.path.join(HERE, 'validate.py'), encoding='utf-8').read())
