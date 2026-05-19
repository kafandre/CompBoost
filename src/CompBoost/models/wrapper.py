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
        base_learner="linear",
        poly_degree=2,
        tree_max_depth=1,
        n_bins=256,
        spline_degree=2,
        n_knots=10,
        loss='mse',
        use_momentum=False,
        momentum_decay=0.9,
        momentum_strength=1.0,
        random_state=None,
        eps_momentum=1e-6,
        eps_linear=1e-8,
        target_df=1.0,
        device="cpu"
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_learner = base_learner
        self.poly_degree = poly_degree
        self.tree_max_depth = tree_max_depth
        self.n_bins = n_bins
        self.spline_degree = spline_degree
        self.n_knots = n_knots
        self.loss = loss
        self.use_momentum = use_momentum
        self.momentum_decay = momentum_decay
        self.momentum_strength = momentum_strength
        self.random_state = random_state
        self.eps_momentum = eps_momentum
        self.eps_linear = eps_linear
        self.target_df = target_df
        self.device = device

    def fit(self, X, y):
        # 1. Scikit-learn validation
        X, y = check_X_y(X, y, y_numeric=True)

        # 2. Initialize PyTorch engine
        self.model_ = ComponentwiseBoostingModel(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            base_learner=self.base_learner,
            poly_degree=self.poly_degree,
            tree_max_depth=self.tree_max_depth,
            n_bins=self.n_bins,
            spline_degree=self.spline_degree,
            n_knots=self.n_knots,
            loss=self.loss,
            use_momentum=self.use_momentum,
            momentum_decay=self.momentum_decay,
            momentum_strength=self.momentum_strength,
            random_state=self.random_state,
            eps_momentum=self.eps_momentum,
            eps_linear=self.eps_linear,
            target_df=self.target_df,
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