"""
YART: Your Another Reranker Trainer
Utilities for logging training metrics.
"""

from collections import defaultdict
from typing import Dict, List, Union

import numpy as np
import torch


class LogMetrics:
    """
    Class for collecting and computing metrics during training.
    """

    def __init__(self):
        self._init()

    def _init(self):
        self.metrics = defaultdict(list)

    def clear(self):
        """Clear all collected metrics."""
        self._init()

    def add(self, key: str, value: Union[float, torch.Tensor]):
        """
        Add a value to a specific metric.

        Args:
            key: Metric name
            value: Value to add
        """
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach().item()
        self.metrics[key].append(float(value))

    def add_dict(self, metrics: Dict[str, Union[float, torch.Tensor]]):
        """
        Add multiple metrics from a dictionary.

        Args:
            metrics: Dictionary of metrics
        """
        for key, value in metrics.items():
            self.add(key, value)

    def _process(self, np_func):
        """
        Process metrics with a NumPy function.

        Args:
            np_func: NumPy function to apply

        Returns:
            Dictionary of processed metrics
        """
        return {key: float(np_func(values)) for key, values in self.metrics.items()}

    def mean(self) -> Dict[str, float]:
        """Calculate mean of each metric."""
        return self._process(np.mean)

    def max(self) -> Dict[str, float]:
        """Calculate maximum of each metric."""
        return self._process(np.max)

    def min(self) -> Dict[str, float]:
        """Calculate minimum of each metric."""
        return self._process(np.min)

    def std(self) -> Dict[str, float]:
        """Calculate standard deviation of each metric."""
        return self._process(np.std)

    def median(self) -> Dict[str, float]:
        """Calculate median of each metric."""
        return self._process(np.median)

    def get_latest(self) -> Dict[str, float]:
        """Get the latest value for each metric."""
        return {
            key: values[-1] if values else float("nan")
            for key, values in self.metrics.items()
        }

    def get_all(self) -> Dict[str, List[float]]:
        """Get all values for each metric."""
        return dict(self.metrics)
