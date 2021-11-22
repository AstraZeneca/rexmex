import pandas as pd
from typing import List, Dict, Tuple

from rexmex.utils import binarize, normalize

from scipy.stats.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy, balanced_accuracy
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

from rexmex.metrics import pr_auc_score
from rexmex.metrics import symmetric_mean_absolute_percentage_error, root_mean_squared_error

class MetricSet(dict):
    """
    A metric set is a special dictionary that contains metric
     name keys and evaluation metric function values.
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

    def add_metrics(self, metrics: List[Tuple]):
        for metric in metrics:
            metric_name, metric_function = metric
            self[metric_name] = metric_function
        return self

    def __repr__(self):
        """
        A representation of the MetricSet object.
        """
        return "MetricSet()"

    def __str__(self):
        """
        Printing the name of metrics when the print() function is called.
        """
        return {k for k in self.keys()}

    def __add__(self, other_metric_set):
        """
        Adding two metric sets together with the + syntactic sugar operator.
        """
        new_metric_set = self
        for name, metric in other_metric_set.items():
            new_metric_set[name] = metric
        return new_metric_set


class ClassificationMetricSet(MetricSet):
    """
    A set of classification metrics with the following metrics included:

    | **ROC AUC**
    | **PR AUC**
    | **Average Precision**
    | **F-1 Score**
    | **Precision**
    | **Recall**
    | **Accuracy**
    | **Balanced Accuracy**
    """
    def __init__(self):
        self["roc_auc"] = roc_auc_score
        self["pr_auc"] = pr_auc_score
        self["average_precision"] = average_precision_score
        self["f1_score"] = binarize(f1_score)
        self["precision"] = binarize(precision_score)
        self["recall"] = binarize(recall_score)
        self["accuracy"] = binarize(accuracy)
        self["balanced_accuracy"] = binarize(balanced_accuracy)

    def __repr__(self):
        """
        A representation of the ClassificationMetricSet object.
        """
        return "ClassificationMetricSet()"

class RatingMetricSet(MetricSet):
    """
    A set of rating metrics with the following metrics included:

    | **MAE** 
    | **MSE** 
    | **RMSE** 
    | **MAPE**
    | **SMAPE** 
    | **R Squared** 
    | **Pearson Correlation** 
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
        """
        A method to normalize a set of metrics.

        Returns:
            self: The metric set after the metrics were normalized.
        """
        for name, metric in self.items():
            self[name] = normalize(metric)
        return self

    def __repr__(self):
        """
        A representation of the RatingMetricSet object.
        """
        return "ClassificationMetricSet()"

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