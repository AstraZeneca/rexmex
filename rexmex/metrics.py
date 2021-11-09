import pandas as pd
from functools import wraps
from typing import List, Dict
from abc import ABC, abstractmethod
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import f1_score, precision_score, recall_score


class MetricSet(ABC):
    """
    """
    
    def __init__(self):
        self._metrics = {}

    @abstractmethod
    def setup_basic_metrics(self):
        pass

    def _get_metrics(self, filter: List[str]=None) -> Dict:
        if filter is None:
            selected_metrics = self._metrics
        else:
            selected_metrics = {name: metric for name, metric in self._metrics.items()}
        return selected_metrics

    #def add_new_metrics(self, metrics: List[Tuple]):
    #    self._metrics = self._metrics + metrics
    #    return self._metrics

    def reset_metrics(self):
        self._metrics = {}
    
    def get_performance_metrics(self, y_true, y_score, filter: List[str]=None) -> pd.DataFrame:
        selected_metrics = self._get_metrics(filter)
        performance_metrics = {name: [metric(y_true, y_score)] for name, metric in selected_metrics.items()}
        performance_metrics = pd.DataFrame.from_dict(performance_metrics)
        return performance_metrics

class ClassificationMetricSet(MetricSet):
    """
    """
    def setup_basic_metrics(self):
        self._metrics["roc_auc"] = roc_auc_score
        self._metrics["pr_auc"] = pr_auc_score
        self._metrics["f1_score"] = binarize(f1_score)
        self._metrics["precision"] = binarize(precision_score)
        self._metrics["recall"] = binarize(recall_score)

class RankingMetricSet(MetricSet):

    def setup_basic_metrics(self):
        pass


class RatingMetricSet(MetricSet):

    def setup_basic_metrics(self):
        pass


class CoverageMetricSet(MetricSet):

    def setup_basic_metrics(self):
        pass

def pr_auc_score(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    return pr_auc

def binarize(metric):
    @wraps(metric)
    def metric_wrapper(*args, **kwargs):
        # TODO: Move to optimal binning.
        y_score = args[1]
        y_score[y_score < 0.5] = 0
        y_score[y_score >=0.5] = 1
        score = metric(*args, **kwargs)
        return score
    return metric_wrapper