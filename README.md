# CompBoost 🚀

[![PyPI version](https://badge.fury.io/py/compboost.svg)](https://badge.fury.io/py/compboost)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CompBoost** is a high-performance, tensor-vectorized Component-wise Gradient Boosting (CWB) library built on PyTorch, with a seamless Scikit-Learn API. 

Rooted in academic research, CompBoost bridges the gap between highly interpretable statistical learning and the advanced optimization dynamics of modern boosting algorithms. It brings model-based gradient boosting—traditionally restricted to the R ecosystem (e.g., `mboost`)—into Python, supercharging it with GPU acceleration, parallelized tensor operations, and a fundamentally novel feature selection algorithm.

## 🌟 The Core Innovation: Momentum-Based Feature Selection

Traditional CWB algorithms are strictly greedy and "memoryless." At each iteration, they select the feature-learner combination that yields the steepest descent in the current empirical risk. While this provides implicit feature selection, it makes the algorithm highly susceptible to local noise, multicollinearity, and overfitting—often causing it to waste boosting updates on non-informative variables.

**CompBoost introduces a highly novel, Pareto-efficient regularizer: Momentum-Based Feature Selection.** Inspired by inertia-based optimizers in deep learning, CompBoost creatively translates the concept of momentum from continuous parameter space into discrete, functional coordinate selection. 

* **Temporal Filtering:** By applying an exponential moving average to the historical loss reductions of each feature, CompBoost injects *historical inertia* into the selection step. 
* **Noise Suppression:** To be selected, a feature must demonstrate a consistent history of error reduction. This acts as a powerful filter against random, local noise.
* **Proven Generalizability:** Extensive stress-testing under severe multicollinearity, low signal-to-noise ratios, and distributional shifts (concept drift and covariate shift) proves that this momentum mechanism is **Pareto-efficient**. It structurally improves feature selection stability and resilience to data drift *without* compromising predictive performance.

## 🔬 Scientific Rigor & Advanced Engineering

CompBoost is not just a wrapper; it is a mathematically rigorous recreation of boosting dynamics designed to solve complex real-world challenges, particularly in high-dimensional ($p \gg n$) and heterogeneous data environments.

* **Competing Base Learners without Bias:** In real-world data, different features require different functional approximations. CompBoost allows diverse base learners (Linear, Polynomial, Decision Stumps, B-Splines) to compete dynamically. To prevent the inherent selection bias toward complex learners, CompBoost utilizes **orthogonal decomposition** and exact **Ridge/P-Spline penalization** based on targeted degrees of freedom ($df$).
* **Vectorized Non-Parametric Learners:** Features like histogram-binning for decision stumps and Cox-de Boor recursive B-Spline basis matrix generation are fully vectorized using PyTorch tensors, enabling massive computational speedups.

Whether you are working with sparse genomic datasets requiring strictly additive interpretability, or heterogeneous tabular data requiring dynamic base-learner complexity, CompBoost provides a statistically rigorous, highly resilient, and blindingly fast solution.

## ✨ Key Features

* **Scikit-Learn Compatible**: Use it directly in `Pipeline`, `GridSearchCV`, and other standard sklearn workflows.
* **PyTorch Backend**: Vectorized mathematical operations and GPU support (`device="cuda"`).
* **Diverse Base Learners**: Supports Linear, Polynomial, Decision Trees, and B-Splines.
* **Competing Base Learners**: Pass a list of base learners (e.g., `["linear", "bspline"]`), and the algorithm will dynamically select the optimal learner for each feature at each iteration.
* **Choose optimizer**: Choose between standard gradient descent and the novel momentum-based feature selection algorithm.

## 📦 Installation

```bash
pip install compboost
```

## 🚀 Quick Start

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from compboost import TorchCompBoostRegressor

# Generate some non-linear data
np.random.seed(42)
X = np.random.uniform(-2, 2, size=(1000, 3))
y = 1.5 * X[:, 0] + 2.0 * (X[:, 1] ** 2) + np.random.normal(0, 0.1, size=1000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize the model with competing base learners and momentum
model = TorchCompBoostRegressor(
    n_estimators=150,
    learning_rate=0.1,
    base_learner=["linear", "polynomial"], # Competing mode
    poly_degree=2,
    use_momentum=True,
    momentum_decay=0.9,
    device="cpu" # Switch to "cuda" for GPU acceleration
)

# Fit and predict
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(f"Test MSE: {mean_squared_error(y_test, preds):.4f}")
```

## 📚 Citation & Background
This library implements the momentum-based feature selection regularizer and advanced CWB mechanics developed by André Kafanke. 

**A formal academic paper detailing the theoretical proofs and extensive benchmarking of this methodology is currently in preparation.** If you use this software in your research, please link back to this GitHub repository. A formal citation (BibTeX) will be provided here once the paper/preprint is published.

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.