"""An example on how to use a distributed (prototype) version of AutoGluon.

1. Make sure to install this version of AutoGluon: https://github.com/LennartPurucker/autogluon/tree/distributed_autogluon
2. Start a ray cluster on your distributed compute resources and remember the head node's address.

Known Bugs and Problems:
* If the run does not use bagging, the code does not correctly set the CPUs and GPUs for fitting so far.
* Very likely other ray-related bug and import related bugs. This is not a stable solution.

Note: if you see a lot of errors, this is a result of canceling running ray jobs and not an actual error.

Then, run the following script to use its distributed capabilities.
"""

from __future__ import annotations

import os
from pathlib import Path

import ray

# --- Parameters ---
"""The Ray address to connect to. Provide "auto" to start a new cluster."""
ray_address: str | None = "TODO"

"""If True, we train all folds (and repeated folds) of a model for AutoGluon at the same time."""
autogluon_force_full_repeated_cross_validation: bool = True

"""If AutoGluon should be run in a distributed mode across several nodes."""
autogluon_distributed: bool = True

"""If AutoGluon is runs several different model fits in parallel. This will result in fitting all folds of a
all `autogluon_distributed_ray_worker` many models at the same time."""
autogluon_distributed_parallel_model_fit: str = "True"

"""If AutoGluon is runs several different model predictions in parallel. This results in predicting all models from
all bags in parallel with `autogluon_distributed_ray_worker` many bagged models at the same time."""
autogluon_distributed_parallel_model_predict: str = "True"

"""The number of works to start in the distributed AutoGluon setup. This is the number of models that are fitted in
parallel. Note, each worker requires at least one cpu (and GPU depending on the model)."""
autogluon_distributed_ray_worker: int = 10

"""The number of works to start in the distributed AutoGluon setup for predicting."""
autogluon_distributed_ray_worker_predict: int = 20


"""Whether to use a shared filesystem for the distributed AutoGluon setup.
This is necessary for some setups, e.g. SLURM. and I tested all of this with this set to True."""
autogluon_distributed_network_shared_filesystem: bool = True


# How many CPUs/GPUs to use per model. Can also be set per model via the hyperparameters.
num_cpus_per_fit = 1
num_gpus_per_fit = 0

# -- Setup
os.environ["AG_DISTRIBUTED_N_RAY_WORKERS"] = str(autogluon_distributed_ray_worker)
os.environ["AG_DISTRIBUTED_N_RAY_WORKERS_PREDICT"] = str(autogluon_distributed_ray_worker_predict)
os.environ["AG_DISTRIBUTED_FIT_MODELS_PARALLEL"] = autogluon_distributed_parallel_model_fit
os.environ["AG_DISTRIBUTED_PREDICT_MODELS_PARALLEL"] = autogluon_distributed_parallel_model_predict
os.environ["AG_DISTRIBUTED_RAY_WORKER_N_CPUS"] = "1"
os.environ["AG_IGNORE_RAY_VERSION"] = "True"

env_vars = {
    "AG_DISTRIBUTED_FIT_MODELS_PARALLEL": os.environ["AG_DISTRIBUTED_FIT_MODELS_PARALLEL"],
    "AG_DISTRIBUTED_N_RAY_WORKERS": os.environ["AG_DISTRIBUTED_N_RAY_WORKERS"],
    "AG_DISTRIBUTED_N_RAY_WORKERS_PREDICT": os.environ["AG_DISTRIBUTED_N_RAY_WORKERS_PREDICT"],
    "AG_DISTRIBUTED_PREDICT_MODELS_PARALLEL": os.environ["AG_DISTRIBUTED_PREDICT_MODELS_PARALLEL"],
    "AG_IGNORE_RAY_VERSION": os.environ["AG_IGNORE_RAY_VERSION"],
    "AG_DISTRIBUTED_RAY_WORKER_N_CPUS": os.environ["AG_DISTRIBUTED_RAY_WORKER_N_CPUS"],
}

if autogluon_distributed:
    os.environ["AG_DISTRIBUTED_MODE"] = "True"
    env_vars["AG_DISTRIBUTED_MODE"] = "True"

if autogluon_distributed_network_shared_filesystem:
    os.environ["AG_DISTRIBUTED_MODE_NFS"] = "True"
    env_vars["AG_DISTRIBUTED_MODE_NFS"] = "True"

if ray_address is not None:
    os.environ["RAY_ADDRESS"] = ray_address
    env_vars["RAY_ADDRESS"] = ray_address

working_dir = Path(__file__).parent
runtime_env = {
    "working_dir ": str(working_dir),
    "excludes": [
        "*",  # exclude everything
    ],
    "env_vars": env_vars,
}

ray.init(runtime_env=runtime_env, namespace="autogluon")


# --- Run AutoGluon ---
from autogluon.tabular import TabularDataset, TabularPredictor

data_root = "https://autogluon.s3.amazonaws.com/datasets/Inc/"
train_data = TabularDataset(data_root + "train.csv")
test_data = TabularDataset(data_root + "test.csv")

predictor = TabularPredictor(label="class", path=str(working_dir / "ag_path")).fit(
    train_data=train_data,
    time_limit=int(60 * 5),
    presets="best_quality",
    ag_args_fit={
        "num_cpus": num_cpus_per_fit,
        "num_gpus": num_gpus_per_fit,
    },
)
predictions = predictor.predict(test_data)
