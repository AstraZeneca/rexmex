import pandas as pd
from typing import List, Dict

from rexmex.utils import binarize, normalize

from scipy.stats.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

from rexmex.metrics import pr_auc_score
from rexmex.metrics import symmetric_mean_absolute_percentage_error, root_mean_squared_error

class MetricSet(dict):
    """
    A metric set is a special dictionary which contains metric name keys and evaluation metric function values.
    """
    def filter_metric_set(self, filter: List[str]=None):
        """
        A method to keep a list of metrics.

        Args:
            filter (str): A list of metric names to keep.
        Returns:
            self: The metric set after the metrics were filtered out.
        """
        for name, _ in self.items():
            if name not in filter:
                del self[name]
        return self

class ClassificationMetricSet(MetricSet):
    """
    A set of classification metrics with the following metrics included:
        **Bla**
        **Bla**
        **Bla**
        **Bla**
    """
    def __init__(self):
        self["roc_auc"] = roc_auc_score
        self["pr_auc"] = pr_auc_score
        self["average_precision"] = average_precision_score
        self["f1_score"] = binarize(f1_score)
        self["precision"] = binarize(precision_score)
        self["recall"] = binarize(recall_score)

class RatingMetricSet(MetricSet):
    """
    A set of rating metrics with the following metrics included:
        **Bla**
        **Bla**
        **Bla**
        **Bla**   
    """
    def __init__(self):
        self["mae"] = mean_absolute_error
        self["mse"] = mean_squared_error
        self["rmse"] = root_mean_squared_error
        self["mape"] = mean_absolute_percentage_error
        self["smape"] = symmetric_mean_absolute_percentage_error
        self["r_squared"] = r2_score
        self["pearson_correlation"] = pearsonr

    def normalize_metrics(self):
        for name, metric in self.items():
            self[name] = normalize(metric)

class CoverageMetricSet(MetricSet):
    """
    A set of coverage metrics with the following metrics included:
    
    """
    pass

class RankingMetricSet(MetricSet):
    """
    A set of ranking metrics with the following metrics included:
    
    """
    pass