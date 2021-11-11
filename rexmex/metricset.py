import pandas as pd
from typing import List, Dict
from abc import ABC, abstractmethod

from rexmex.utils import binarize, normalize

from scipy.stats.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

from rexmex.metrics import pr_auc_score
from rexmex.metrics import symmetric_mean_absolute_percentage_error, root_mean_squared_error

class MetricSet(ABC):
    """
    """   
    def __init__(self):
        """
        """
        self._metrics = {}

    @abstractmethod
    def setup_basic_metrics(self):
        """
        """
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
        self._metrics["average_precision"] = average_precision_score
        self._metrics["f1_score"] = binarize(f1_score)
        self._metrics["precision"] = binarize(precision_score)
        self._metrics["recall"] = binarize(recall_score)

class RatingMetricSet(MetricSet):
    """
    """
    def setup_basic_metrics(self):
        self._metrics["mae"] = mean_absolute_error
        self._metrics["mse"] = mean_squared_error
        self._metrics["rmse"] = root_mean_squared_error
        self._metrics["mape"] = mean_absolute_percentage_error
        self._metrics["smape"] = symmetric_mean_absolute_percentage_error
        self._metrics["r_squared"] = r2_score
        self._metrics["linear_correlation"] = pearsonr

class CoverageMetricSet(MetricSet):
    """
    """
    def setup_basic_metrics(self):
        pass


class RankingMetricSet(MetricSet):
    """
    """
    def setup_basic_metrics(self):
        pass
