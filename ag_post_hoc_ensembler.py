"""Code to use AutoGluon to do post hoc ensembling with output artifacts.


To adjust the number of decimals for rounding, set the `round_decimals` variable in
`autogluon.core.models.greedy_ensemble.ensemble_selection.EnsembleSelection` Line 112 to `8`.

"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
from autogluon.common.loaders import load_pd
from autogluon.common.savers import save_pd
from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.core.constants import BINARY
from autogluon.core.metrics import get_metric
from autogluon.core.models.greedy_ensemble.ensemble_selection import EnsembleSelection
from autogluon.core.utils.utils import get_pred_from_proba


@dataclass
class CompetitionInfo:
    label = "class"
    id_columns = ["id"]
    eval_metric = "mcc"
    problem_type = BINARY


class PostHocEnsemble:
    def __init__(
        self,
        comp_info: CompetitionInfo,
        y_pred_proba_multi_oof: pd.DataFrame,
        # y_pred_proba_multi_holdout: pd.DataFrame,
        y_pred_proba_multi_test: pd.DataFrame,
        gt_transform_oof: pd.DataFrame,
        # gt_transform_holdout: pd.DataFrame,
    ):
        self.comp_info = comp_info
        self.y_pred_proba_multi_oof = y_pred_proba_multi_oof
        # self.y_pred_proba_multi_holdout = y_pred_proba_multi_holdout
        self.y_pred_proba_multi_test = y_pred_proba_multi_test
        self.gt_transform_oof = gt_transform_oof
        # self.gt_transform_holdout = gt_transform_holdout
        assert self.gt_transform_oof[self.comp_info.id_columns].equals(
            self.y_pred_proba_multi_oof[self.comp_info.id_columns],
        )
        # assert self.gt_transform_holdout[self.comp_info.id_columns].equals(self.y_pred_proba_multi_holdout[self.comp_info.id_columns])
        self.ensemble_selection: EnsembleSelection = None
        self.models = None
        self.metric = get_metric(self.comp_info.eval_metric, problem_type=self.comp_info.problem_type)

    def models_all(self):
        models_all = list(self.y_pred_proba_multi_oof.columns)
        return [m for m in models_all if m not in self.comp_info.id_columns]

    def fit_weighted_ensemble(self, models: list | None = None):
        X = self.y_pred_proba_multi_oof.copy()
        X = X.drop(columns=self.comp_info.id_columns)
        if models is None:
            models = list(X.columns)
        X = X[models]

        y = self.gt_transform_oof[self.comp_info.label].copy()
        ensemble_selection = EnsembleSelection(
            ensemble_size=80,
            problem_type=self.comp_info.problem_type,
            metric=self.metric,
            # calibrate_decision_threshold=True,
        )

        X_in = X.values.T
        ensemble_selection.fit(
            predictions=X_in,
            labels=y,
        )

        self.models = list(X.columns)
        print(ensemble_selection.trajectory_)
        self.ensemble_selection = ensemble_selection

    def predict_proba(self, X):
        X = X[self.models]
        return self.ensemble_selection.predict_proba(X=X.values.T)

    def predict(self, X):
        X = X[self.models]
        return self.ensemble_selection.predict(X=X.values.T)

    def score_from_pred_proba(self, y_true, y_pred_proba, decision_threshold=0.5):
        y_pred = get_pred_from_proba(
            y_pred_proba,
            problem_type=self.comp_info.problem_type,
            decision_threshold=decision_threshold,
        )
        return self.metric(y_true=y_true, y_pred=y_pred)

    def weights(self):
        weights_series = pd.Series(dict(zip(self.models, self.ensemble_selection.weights_, strict=False))).sort_values(
            ascending=True,
        )
        print(weights_series)
        return weights_series

    def score_single_oof(self):
        models = self.models_all()
        X = self.y_pred_proba_multi_oof[models]
        y = self.gt_transform_oof[self.comp_info.label]
        scores = {}
        for m in models:
            scores[m] = self.score_from_pred_proba(y_true=y, y_pred_proba=X[m])

        return pd.Series(scores, name="score").sort_values(ascending=False)

    @classmethod
    def from_path(cls, path: str, comp_info: CompetitionInfo):
        y_pred_proba_multi_oof = load_pd.load(os.path.join(path, "y_pred_proba_multi_oof.csv"))
        # y_pred_proba_multi_holdout = load_pd.load(os.path.join(path, "y_pred_proba_multi_holdout.csv"))
        y_pred_proba_multi_test = load_pd.load(os.path.join(path, "y_pred_proba_multi_test.csv"))
        gt_transform_oof = load_pd.load(os.path.join(path, "gt_transform_oof.csv"))
        # gt_transform_holdout = load_pd.load(os.path.join(path, "gt_transform_holdout.csv"))
        return cls(
            comp_info=comp_info,
            y_pred_proba_multi_oof=y_pred_proba_multi_oof,
            # y_pred_proba_multi_holdout=y_pred_proba_multi_holdout,
            y_pred_proba_multi_test=y_pred_proba_multi_test,
            gt_transform_oof=gt_transform_oof,
            # gt_transform_holdout=gt_transform_holdout,
        )


def load_submission(post_hoc_ensemble: PostHocEnsemble, sub_path: str, sub_name: str):
    y_pred_proba_oof = load_pd.load(path=f"{sub_path}/oof_pred_proba.csv")
    y_pred_proba_test = load_pd.load(path=f"{sub_path}/pred_proba.csv")
    y_pred_test_submission = load_pd.load(path=f"{sub_path}/pred_submission.csv")

    print(y_pred_proba_oof)
    print(y_pred_proba_test)
    print(y_pred_test_submission)

    y_pred_proba_oof = y_pred_proba_oof.set_index("id")
    y_pred_proba_oof = y_pred_proba_oof["p"]
    y_pred_proba_test = y_pred_proba_test["p"]

    post_hoc_ensemble.y_pred_proba_multi_oof[sub_name] = y_pred_proba_oof
    post_hoc_ensemble.y_pred_proba_multi_test[sub_name] = y_pred_proba_test

    print(post_hoc_ensemble.y_pred_proba_multi_test)

    return y_pred_proba_oof, y_pred_proba_test, y_pred_test_submission


if __name__ == "__main__":
    set_logger_verbosity(4)
    comp_info = CompetitionInfo()

    data_path = "./data"

    # -- Load existing predictor for post hoc ensembling
    run_artifact_path = "path/to/fitted_predictor"
    post_hoc_ensemble = PostHocEnsemble.from_path(path=run_artifact_path, comp_info=comp_info)

    # -- Load other submission that we want to ensemble
    submissions_path = "../submission"
    sub_1_path = f"{submissions_path}/sub1"
    y_pred_proba_oof_2, y_pred_proba_test_2, y_pred_proba_test_submission_2 = load_submission(
        post_hoc_ensemble=post_hoc_ensemble,
        sub_path=sub_1_path,
        sub_name="Submission1",
    )

    scores = post_hoc_ensemble.score_single_oof()
    print(scores)

    post_hoc_ensemble.fit_weighted_ensemble(
        models=[
            "Submission1",
            "NeuralNetFastAI_BAG_L2",
            "WeightedEnsemble_L3",
            "RandomForestGini_BAG_L2",
            "ExtraTreesEntr_BAG_L2",
            "WeightedEnsemble_L2",
            "RandomForestEntr_BAG_L2",
            "LightGBMXT_BAG_L2",
            "LightGBMXT_BAG_L2",
            "XGBoost_BAG_L2",
            "LightGBMXT_BAG_L1",
        ],
    )

    weights = post_hoc_ensemble.weights()

    X = post_hoc_ensemble.y_pred_proba_multi_oof
    y_pred_proba = post_hoc_ensemble.predict_proba(X)
    y_pred = post_hoc_ensemble.predict(X)

    print(y_pred_proba)
    print(y_pred)

    score = post_hoc_ensemble.score_from_pred_proba(
        y_true=post_hoc_ensemble.gt_transform_oof[post_hoc_ensemble.comp_info.label],
        y_pred_proba=y_pred_proba,
    )
    print(f"Score OOF: {score}")

    X_test = post_hoc_ensemble.y_pred_proba_multi_test
    y_pred_proba_test = post_hoc_ensemble.predict_proba(X_test)
    y_pred_test = post_hoc_ensemble.predict(X_test)

    submission = X_test[post_hoc_ensemble.comp_info.id_columns].copy()
    submission[post_hoc_ensemble.comp_info.label] = y_pred_test
    print(submission)

    submission[post_hoc_ensemble.comp_info.label] = (
        submission[post_hoc_ensemble.comp_info.label].astype("object").map({1: "p", 0: "e"})
    )
    print(submission)

    save_pd.save(path="post_hoc_ensemble_submission.csv", df=submission)
