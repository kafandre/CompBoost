import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from .ComponentwiseBoostingModel import ComponentwiseBoostingModel

class TorchCompBoostRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-Learn compatible wrapper for the PyTorch Component-wise Boosting Model.

    This regressor implements component-wise gradient boosting using tensor-vectorized
    PyTorch operations. It supports competing base learners and a novel momentum-based 
    feature selection regularizer.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of boosting iterations to perform.
    learning_rate : float, default=0.1
        Shrinks the contribution of each base learner by this value.
    base_learner : str or list of str, default="linear"
        The type of base learner(s) to use. Options are "linear", "polynomial", 
        "tree", and "bspline". If a list is provided (e.g., ["linear", "bspline"]), 
        the model operates in competing mode, selecting the best learner per iteration.
    poly_degree : int, default=2
        The degree of the polynomial if "polynomial" is in `base_learner`.
    tree_max_depth : int, default=1
        Maximum depth of the decision tree (currently acts as decision stumps).
    n_bins : int, default=256
        Number of bins used for histogram-based tree splitting.
    spline_degree : int, default=2
        Degree of the B-splines if "bspline" is in `base_learner`.
    n_knots : int, default=10
        Number of interior knots for B-splines.
    loss : str, default='mse'
        The loss function to optimize. Currently supports Mean Squared Error ('mse').
    use_momentum : bool, default=False
        Whether to use momentum-based feature selection to regularize the boosting path.
    momentum_decay : float, default=0.9
        Decay factor for the momentum tracker (requires `use_momentum=True`).
    momentum_strength : float, default=1.0
        Multiplier for the momentum penalty (requires `use_momentum=True`).
    random_state : int or None, default=None
        Seed for the random number generator for reproducible results.
    eps_momentum : float, default=1e-6
        Small constant added for numerical stability in momentum calculations.
    eps_linear : float, default=1e-8
        Small constant added to the diagonal of matrices for Ridge-like stabilization.
    target_df : float, default=1.0
        Target degrees of freedom used for penalization of complex base learners.
    device : str, default="cpu"
        The PyTorch device to run calculations on (e.g., "cpu", "cuda", "mps").
    """
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        base_learner="linear",
        poly_degree=2,
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
        device="cpu",
        verbose=10
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_learner = base_learner
        self.poly_degree = poly_degree
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
        self.verbose = verbose

    def fit(self, X, y, X_val=None, y_val=None):
        if self.loss != 'mse':
            raise ValueError(f"loss must be 'mse'. Got: {self.loss}")
        if (X_val is not None) != (y_val is not None):
            raise ValueError("Both X_val and y_val must be provided together for validation tracking.")
        # 1. Scikit-learn validation
        X, y = check_X_y(X, y, y_numeric=True)
        if X_val is not None and y_val is not None:
            X_val, y_val = check_X_y(X_val, y_val, y_numeric=True)

        # 2. Initialize PyTorch engine
        self.model_ = ComponentwiseBoostingModel(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            base_learner=self.base_learner,
            poly_degree=self.poly_degree,
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
            device=self.device,
            verbose=self.verbose
        )

        # 3. Fit the model
        self.model_.fit(X, y, X_val=X_val, y_val=y_val)
        
        # 4. Calculate feature importances and features in
        self.n_features_in_ = X.shape[1]
        importances = np.zeros(self.n_features_in_)
        selected = self.model_.history['selected_features']
        for idx in selected:
            importances[idx] += 1
        if len(selected) > 0:
            self.feature_importances_ = importances / len(selected)
        else:
            self.feature_importances_ = importances
        
        # 5. Mark as fitted for scikit-learn
        self.is_fitted_ = True
        return self

    def predict(self, X, use_best_model=False):
        # 1. Scikit-learn validation
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        # 2. Predict using PyTorch engine
        preds = self.model_.predict(X, use_best_model=use_best_model)

        # 3. Ensure output is a standard numpy array
        if isinstance(preds, torch.Tensor):
            return preds.detach().cpu().numpy()
        return np.array(preds)

    def to(self, device):
        """Moves the regressor's PyTorch engine and its parameters to the specified device."""
        self.device = str(device)
        if hasattr(self, 'model_'):
            self.model_.to(device)
        return self

    def save_model(self, path):
        """Saves the fitted regressor to disk using torch.save."""
        check_is_fitted(self, 'is_fitted_')
        import torch
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self, path)

    @staticmethod
    def load_model(path, map_location=None):
        """Loads a saved regressor from disk, mapping tensors to the specified device."""
        import torch
        if map_location is None:
            if not torch.cuda.is_available() and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                map_location = 'cpu'
        
        reg = torch.load(path, map_location=map_location, weights_only=False)
        if map_location is not None:
            reg.to(map_location)
        return reg