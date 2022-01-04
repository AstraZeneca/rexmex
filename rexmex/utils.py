from functools import wraps
from typing import Optional

import numpy as np

__all__ = [
    "binarize",
    "normalize",
    "Annotator",
]


def binarize(metric):
    """
    Binarize the predictions for a ground-truth - prediction vector pair.

    Args:
        metric (function): The metric function which needs a binarization pre-processing step.
    Returns:
        metric_wrapper (function): The function which wraps the metric and binarizes the probability scores.
    """

    @wraps(metric)
    def metric_wrapper(*args, **kwargs):
        # TODO: Move to optimal binning. Youden’s J statistic.
        y_score = args[1]
        y_score[y_score < 0.5] = 0
        y_score[y_score >= 0.5] = 1
        score = metric(*args, **kwargs)
        return score

    return metric_wrapper


def normalize(metric):
    """
    Normalize the predictions for a ground-truth - prediction vector pair.

    Args:
        metric (function): The metric function which needs a normalization pre-processing step.
    Returns:
        metric_wrapper (function): The function which wraps the metric and normalizes predictions.
    """

    @wraps(metric)
    def metric_wrapper(*args, **kwargs):
        y_true = args[0]
        y_score = args[1]
        y_mean = np.mean(y_true)
        y_std = np.std(y_true)
        y_true[:] = (y_true - y_mean) / y_std
        y_score[:] = (y_score - y_mean) / y_std
        score = metric(*args, **kwargs)
        return score

    return metric_wrapper


class Annotator:
    """A class to wrap annotations to make the registry pattern easier later."""

    def annotate(
        self,
        *,
        lower: float,
        upper: float,
        higher_is_better: bool,
        link: str,
        description: str,
        name: Optional[str] = None,
        lower_inclusive: bool = True,
        upper_inclusive: bool = True,
    ):
        """Annotate a classification function."""

        def _wrapper(func):
            func.name = name or func.__name__.replace("_", " ").title()
            func.lower = lower
            func.lower_inclusive = lower_inclusive
            func.upper = upper
            func.upper_inclusive = upper_inclusive
            func.higher_is_better = higher_is_better
            func.link = link
            func.description = description
            return func

        return _wrapper
