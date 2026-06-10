"""
CompBoost: A high-performance, tensor-vectorized Component-wise Gradient Boosting library.
"""

from .models.wrapper import TorchCompBoostRegressor
__version__ = "0.1.0"
__all__ = ["TorchCompBoostRegressor"]