"""Code to reproduce our AutoGluon and data setup for the compeition.

Note, to increase this number, you need to monkey patch the local install of AutoGluon.
To do so, one needs to set the value in this line (https://github.com/autogluon/autogluon/blob/master/core/src/autogluon/core/models/greedy_ensemble/greedy_weighted_ensemble_model.py#L22) to 100.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

# -- Get Data
X_train = pd.read_csv("./train.csv")
X_test = pd.read_csv("./test.csv")
label = "class"

# -- Preprocessing
X_train = X_train.drop(columns=["id"])
X_test = X_test.drop(columns=["id"])

weird_columns = [
    "cap-shape",
    "cap-surface",
    "cap-color",
    "gill-attachment",
    "gill-spacing",
    "gill-color",
    "veil-type",
    "veil-color",
    "has-ring",
    "ring-type",
    "spore-print-color",
    "habitat",
    "does-bruise-or-bleed",
    "stem-root",
    "stem-surface",
    "stem-color",
]

for col in weird_columns:
    allowed_vals = X_test[col].unique()
    X_train.loc[~X_train[col].isin(allowed_vals), col] = np.nan
    X_test.loc[~X_test[col].isin(allowed_vals), col] = np.nan

cat_columns = [
    "cap-shape",
    "cap-surface",
    "cap-color",
    "does-bruise-or-bleed",
    "gill-attachment",
    "gill-spacing",
    "gill-color",
    "stem-root",
    "stem-surface",
    "stem-color",
    "veil-type",
    "veil-color",
    "has-ring",
    "ring-type",
    "spore-print-color",
    "habitat",
    "season",
]
X_train[cat_columns] = X_train[cat_columns].astype("category")
X_test[cat_columns] = X_test[cat_columns].astype("category")

# -- Get custom portfolio
from tabrepo_2024_custom import zeroshot2024

allowed_models = [
    "LR",
    "FASTAI",
    "NN_TORCH",
    "GBM",
    "CAT",
    "XGB",
    "RF",
    "XT",
]

for k in list(zeroshot2024.keys()):
    if k not in allowed_models:
        del zeroshot2024[k]

# -- Run AutoGluon
predictor = TabularPredictor(
    label=label,
    eval_metric="mcc",
    problem_type="binary",
    verbosity=2,
)

predictor.fit(
    time_limit=int(60 * 60 * 4),
    train_data=X_train,
    presets="best_quality",
    dynamic_stacking=False,
    hyperparameters=zeroshot2024,
    # Early Stopping
    ag_args_fit={
        "stopping_metric": "log_loss",
    },
    # Validation Protocol
    num_bag_folds=16,
    num_bag_sets=1,
    num_stack_levels=1,
)
predictor.fit_summary(verbosity=1)
predictions = predictor.predict(X_test)
