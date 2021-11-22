import numpy as np
from functools import wraps

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
        y_true = (y_true - y_mean)/y_std
        y_score = (y_score - y_mean)/y_std
        score = metric(*args, **kwargs)
        return score
    return metric_wrapper