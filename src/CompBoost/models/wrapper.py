import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from .ComponentwiseBoostingModel import ComponentwiseBoostingModel

class TorchCompBoostRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-Learn compatible wrapper for the PyTorch Component-wise Boosting Model.
    """
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        base_learner="polynomial",
        degree=3,
        bin_edges=None,
        device="cpu"
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_learner = base_learner
        self.degree = degree
        self.bin_edges = bin_edges
        self.device = device

    def fit(self, X, y):
        # 1. Scikit-learn validation
        X, y = check_X_y(X, y, y_numeric=True)

        # 2. Initialize PyTorch engine
        self.model_ = ComponentwiseBoostingModel(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            base_learner=self.base_learner,
            degree=self.degree,
            bin_edges=self.bin_edges,
            device=self.device
        )

        # 3. Fit the model
        self.model_.fit(X, y)
        
        # 4. Mark as fitted for scikit-learn
        self.is_fitted_ = True
        return self

    def predict(self, X):
        # 1. Scikit-learn validation
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        # 2. Predict using PyTorch engine
        preds = self.model_.predict(X)

        # 3. Ensure output is a standard numpy array
        if isinstance(preds, torch.Tensor):
            return preds.detach().cpu().numpy()
        return np.array(preds)