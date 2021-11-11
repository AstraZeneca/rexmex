import numpy as np
from functools import wraps

def binarize(metric):
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
    @wraps(metric)
    def metric_wrapper(*args, **kwargs):    
        y_mean = np.mean(y_true)
        y_std = np.mean(y_true)
        y_true = (y_true - y_mean)/y_std
        y_scores = (y_scores - y_mean)/y_std
        score = metric(*args, **kwargs)
        return score
    return metric_wrapper