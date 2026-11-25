from sklearn.utils.estimator_checks import parametrize_with_checks

from compboost import TorchCompBoostRegressor

BASE_LEARNERS = [
    "linear",
    "polynomial",
    "tree",
    "bspline",
    ["linear", "bspline"],
]


@parametrize_with_checks(
    [
        TorchCompBoostRegressor(base_learner=base_learner, verbose=0)
        for base_learner in BASE_LEARNERS
    ]
)
def test_sklearn_compliance(estimator, check):
    check(estimator)
