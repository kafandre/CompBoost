import pytest
import torch
import numpy as np
from compboost.models.ComponentwiseBoostingModel import ComponentwiseBoostingModel

@pytest.fixture
def synthetic_data():
    """Generates a simple non-linear dataset for testing."""
    np.random.seed(42)
    X = np.random.uniform(-2, 2, size=(100, 3))
    # True relationship: linear + quadratic + noise
    y = 1.5 * X[:, 0] + 2.0 * (X[:, 1] ** 2) + np.random.normal(0, 0.1, size=100)
    return X, y

def test_model_initialization():
    """Verifies default parameters and legacy mode targeting."""
    model = ComponentwiseBoostingModel(base_learner="linear")
    assert model.legacy_mode is True
    assert model.base_learner == "linear"

    model_competing = ComponentwiseBoostingModel(base_learner=["linear", "tree"])
    assert model_competing.legacy_mode is False
    assert model_competing.base_learner == "competing"

def test_2d_target_handling(synthetic_data):
    """Verifies that the model can handle 2D column-vector targets and raises ValueError for multi-output."""
    X, y = synthetic_data
    # 2D column vector target: shape (N, 1)
    y_2d = y.reshape(-1, 1)
    
    model = ComponentwiseBoostingModel(n_estimators=5, base_learner="linear")
    # This should not crash and should work successfully
    model.fit(X, y_2d)
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)
    
    # 2D target with multiple columns should raise ValueError
    y_multi = np.column_stack([y, y])
    with pytest.raises(ValueError, match="Multi-output targets are not supported"):
        model.fit(X, y_multi)

@pytest.mark.parametrize("base_learner", ["linear", "polynomial", "tree", "bspline", ["linear", "tree"]])
def test_base_learners_execution(synthetic_data, base_learner):
    """Ensures all single and competing base learners fit and predict without crashing."""
    X, y = synthetic_data
    model = ComponentwiseBoostingModel(n_estimators=10, base_learner=base_learner, random_state=42)
    
    # Check execution
    model.fit(X, y)
    preds = model.predict(X)
    
    assert isinstance(preds, torch.Tensor)
    assert preds.shape == (X.shape[0],)
    assert len(model.estimators_) == 10

def test_loss_reduction(synthetic_data):
    """Checks that the training MSE decreases over boosting iterations."""
    X, y = synthetic_data
    model = ComponentwiseBoostingModel(n_estimators=30, base_learner="polynomial", learning_rate=0.1)
    model.fit(X, y)
    
    train_loss = model.history['train_loss']
    assert train_loss[-1] < train_loss[0], "Training loss failed to decrease."

def test_validation_and_testing_tracking(synthetic_data):
    """Verifies data split evaluation and best iteration calculation."""
    X, y = synthetic_data
    X_tr, y_tr = X[:80], y[:80]
    X_va, y_va = X[80:], y[80:]
    
    model = ComponentwiseBoostingModel(n_estimators=20, base_learner="tree")
    model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
    
    assert len(model.history['val_loss']) == 20
    assert model.best_iteration_ > 0

def test_device_handling(synthetic_data):
    """Ensures tensor handling complies with target device settings."""
    X, y = synthetic_data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = ComponentwiseBoostingModel(n_estimators=5, device=device)
    model.fit(X, y)
    preds = model.predict(X)
    
    assert preds.device.type == device

def test_momentum_feature_selection(synthetic_data):
    """Ensures that momentum logic updates tracking dictionaries without crashing."""
    X, y = synthetic_data
    model = ComponentwiseBoostingModel(
        n_estimators=15, 
        base_learner="linear", 
        use_momentum=True, 
        momentum_decay=0.8,
        momentum_strength=1.5
    )
    model.fit(X, y)
    
    # Check that momentum dictionary was populated and updated
    assert len(model.feature_momentum) == X.shape[1]
    # At least one feature should have a non-zero momentum after 15 iterations
    assert any(val != 0.0 for val in model.feature_momentum.values())

def test_predict_use_best_model(synthetic_data):
    """Verifies that predicting with the best iteration restricts the estimator list."""
    X, y = synthetic_data
    X_tr, y_tr = X[:80], y[:80]
    X_va, y_va = X[80:], y[80:]
    
    # High learning rate to deliberately trigger overfitting / early stopping
    model = ComponentwiseBoostingModel(n_estimators=30, base_learner="tree", learning_rate=0.5)
    model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
    
    preds_full = model.predict(X_va, use_best_model=False)
    preds_best = model.predict(X_va, use_best_model=True)
    
    assert model.best_iteration_ > 0
    assert model.best_iteration_ <= len(model.estimators_)
    assert preds_full.shape == preds_best.shape
    
    # If best_iteration < n_estimators, the predictions should differ
    if model.best_iteration_ < model.n_estimators:
        assert not torch.allclose(preds_full, preds_best)

def test_save_and_load_model(synthetic_data, tmp_path):
    """Ensures the model can be pickled and unpickled while maintaining exact predictions."""
    X, y = synthetic_data
    model = ComponentwiseBoostingModel(n_estimators=5, base_learner="polynomial")
    model.fit(X, y)
    
    # tmp_path is a built-in pytest fixture that creates a temporary directory for the test
    file_path = tmp_path / "test_model.pkl"
    model.save_model(file_path)
    
    loaded_model = ComponentwiseBoostingModel.load_model(file_path)
    
    # Ensure predictions are exactly identical
    preds_orig = model.predict(X)
    preds_loaded = loaded_model.predict(X)
    
    assert torch.allclose(preds_orig, preds_loaded)

def test_device_migration_and_serialization(synthetic_data, tmp_path):
    """Verifies that the model can migrate device affinity and serialize/deserialize across devices."""
    X, y = synthetic_data
    model = ComponentwiseBoostingModel(n_estimators=5, base_learner=["linear", "polynomial"], target_df=1.0)
    model.fit(X, y)
    
    # Move model to CPU
    model.to("cpu")
    assert model.device == "cpu"
    
    # Verify internal tensors are on CPU
    if model.all_bin_edges is not None:
        assert model.all_bin_edges.device.type == "cpu"
    for est in model.estimators_:
        params = est['params']
        if isinstance(params, torch.Tensor):
            assert params.device.type == "cpu"
        elif isinstance(params, dict):
            for v in params.values():
                if isinstance(v, torch.Tensor):
                    assert v.device.type == "cpu"

    # Save using PyTorch serializer
    file_path = tmp_path / "test_device_model.pt"
    model.save_model(file_path)
    
    # Load using PyTorch deserializer explicitly mapping to 'cpu'
    loaded_model = ComponentwiseBoostingModel.load_model(file_path, map_location="cpu")
    assert loaded_model.device == "cpu"
    
    # Ensure predictions are identical
    preds_orig = model.predict(X)
    preds_loaded = loaded_model.predict(X)
    assert torch.allclose(preds_orig, preds_loaded)

def test_constant_features_bspline(synthetic_data):
    """Verifies that the model can handle constant features when using B-splines."""
    X, y = synthetic_data
    # Add a constant feature as the last column
    X_const = np.column_stack([X, np.ones(X.shape[0])])
    
    # Test competing mode with bspline and linear
    model_comp = ComponentwiseBoostingModel(n_estimators=5, base_learner=["bspline", "linear"], target_df=1.0)
    model_comp.fit(X_const, y)
    preds_comp = model_comp.predict(X_const)
    assert preds_comp.shape == (X.shape[0],)
    
    # Test legacy mode with bspline
    model_leg = ComponentwiseBoostingModel(n_estimators=5, base_learner="bspline")
    model_leg.fit(X_const, y)
    preds_leg = model_leg.predict(X_const)
    assert preds_leg.shape == (X.shape[0],)

def test_bspline_prediction_cropping(synthetic_data):
    """Verifies that the B-spline prediction path crops the design matrix if it has more columns than coeffs."""
    X, y = synthetic_data
    
    # 1. Test legacy mode cropping
    model_leg = ComponentwiseBoostingModel(n_estimators=1, base_learner="bspline")
    model_leg.fit(X, y)
    
    # Manually truncate the coefficients to trigger the cropping branch
    est_leg = model_leg.estimators_[0]
    orig_coeffs_leg = est_leg['params']['coeffs']
    est_leg['params']['coeffs'] = orig_coeffs_leg[:-1]
    
    preds_leg = model_leg.predict(X)
    assert preds_leg.shape == (X.shape[0],)
    
    # 2. Test competing mode cropping
    model_comp = ComponentwiseBoostingModel(n_estimators=1, base_learner=["bspline", "linear"])
    model_comp.fit(X, y)
    
    # Mock a bspline estimator with truncated beta coefficients
    n_basis = model_comp.n_knots + model_comp.spline_degree + 1
    est_comp = model_comp.estimators_[0]
    est_comp['learner'] = 'bspline'
    est_comp['params'] = {
        'beta': torch.zeros(n_basis - 1, device=model_comp.device),
        'beta_lin': torch.zeros(2, device=model_comp.device),
        'knots': np.linspace(-3, 3, model_comp.n_knots + 2 * model_comp.spline_degree + 2)
    }
    
    preds_comp = model_comp.predict(X)
    assert preds_comp.shape == (X.shape[0],)

def test_asymmetric_validation_split(synthetic_data):
    """Verifies that fit raises ValueError when validation or test splits are asymmetric."""
    X, y = synthetic_data
    model = ComponentwiseBoostingModel(n_estimators=5, base_learner="linear")
    
    # Asymmetric validation splits
    with pytest.raises(ValueError, match="Both X_val and y_val must be provided together"):
        model.fit(X, y, X_val=X)
        
    with pytest.raises(ValueError, match="Both X_val and y_val must be provided together"):
        model.fit(X, y, y_val=y)
        
    # Asymmetric test splits
    with pytest.raises(ValueError, match="Both X_test and y_test must be provided together"):
        model.fit(X, y, X_test=X)
        
    with pytest.raises(ValueError, match="Both X_test and y_test must be provided together"):
        model.fit(X, y, y_test=y)

def test_invalid_base_learner(synthetic_data):
    """Verifies that fit raises ValueError when base learner is invalid."""
    X, y = synthetic_data
    
    # Single invalid base learner string
    model = ComponentwiseBoostingModel(n_estimators=5, base_learner="invalid")
    with pytest.raises(ValueError, match="Invalid base_learner 'invalid'"):
        model.fit(X, y)
        
    # List containing invalid base learner
    model_list = ComponentwiseBoostingModel(n_estimators=5, base_learner=["linear", "invalid_item"])
    with pytest.raises(ValueError, match="Invalid base_learner 'invalid_item'"):
        model_list.fit(X, y)