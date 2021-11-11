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