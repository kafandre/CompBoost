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