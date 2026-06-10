import pytest
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from compboost.models.wrapper import TorchCompBoostRegressor

@pytest.fixture
def numpy_data():
    """Generates standard numpy array training sets."""
    X = np.random.randn(50, 4)
    y = np.dot(X, np.array([1.0, -2.0, 0.5, 0.0])) + np.random.normal(0, 0.05, size=50)
    return X, y

def test_wrapper_numpy_io(numpy_data):
    """Ensures input and outputs are clean standard NumPy structures without leaky Tensors."""
    X, y = numpy_data
    reg = TorchCompBoostRegressor(n_estimators=5, base_learner="linear")
    
    reg.fit(X, y)
    preds = reg.predict(X)
    
    assert isinstance(preds, np.ndarray)
    assert not isinstance(preds, type(X)) == False # Check strict numpy array type
    assert preds.ndim == 1
    assert preds.shape[0] == X.shape[0]

def test_scikit_learn_clone_compatibility():
    """Validates that clone() copies structural hyperparameters cleanly."""
    reg = TorchCompBoostRegressor(n_estimators=45, learning_rate=0.05, poly_degree=4)
    cloned_reg = clone(reg)
    
    assert cloned_reg.n_estimators == 45
    assert cloned_reg.learning_rate == 0.05
    assert cloned_reg.poly_degree == 4

def test_grid_search_integration(numpy_data):
    """Tests the wrapper within scikit-learn optimization pipelines."""
    X, y = numpy_data
    reg = TorchCompBoostRegressor()
    
    param_grid = {
        'n_estimators': [5, 10],
        'base_learner': ['linear', 'tree']
    }
    
    grid = GridSearchCV(estimator=reg, param_grid=param_grid, cv=2)
    
    # Should run cross-validation loops smoothly without syntax mismatch
    grid.fit(X, y)
    
    assert grid.best_params_['n_estimators'] in [5, 10]
    assert isinstance(grid.predict(X), np.ndarray)

def test_wrapper_device_and_serialization(numpy_data, tmp_path):
    """Verifies that the scikit-learn wrapper supports device migration and serialization/deserialization."""
    X, y = numpy_data
    reg = TorchCompBoostRegressor(n_estimators=5, base_learner=["linear", "tree"])
    reg.fit(X, y)
    
    # Move to CPU
    reg.to("cpu")
    assert reg.device == "cpu"
    assert reg.model_.device == "cpu"
    
    # Save
    file_path = tmp_path / "wrapper_model.pt"
    reg.save_model(file_path)
    
    # Load
    loaded_reg = TorchCompBoostRegressor.load_model(file_path, map_location="cpu")
    assert loaded_reg.device == "cpu"
    assert loaded_reg.model_.device == "cpu"
    
    # Predict and verify identical output
    preds_orig = reg.predict(X)
    preds_loaded = loaded_reg.predict(X)
    assert np.allclose(preds_orig, preds_loaded)

def test_wrapper_api_compatibility(numpy_data, capsys):
    """Verifies loss parameter validation, validation set fitting, and verbose printing."""
    X, y = numpy_data
    X_tr, y_tr = X[:40], y[:40]
    X_va, y_va = X[40:], y[40:]
    
    # 1. Test loss validation (must raise ValueError if not mse)
    with pytest.raises(ValueError, match="loss must be 'mse'"):
        TorchCompBoostRegressor(loss="absolute_error")
        
    # 2. Test fit with validation data
    reg = TorchCompBoostRegressor(n_estimators=12, verbose=5)
    reg.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
    
    # Ensure validation history is populated
    assert len(reg.model_.history['val_loss']) == 12
    assert reg.model_.best_iteration_ > 0
    
    # 3. Test verbose parameter and output capture
    captured = capsys.readouterr()
    assert "Iter 5/" in captured.out
    assert "Iter 10/" in captured.out
    assert "Iter 12/" not in captured.out