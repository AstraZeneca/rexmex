import pandas as pd
from sklearn.metrics  import roc_auc_score

class MetricSet(object):
 
    def get_performance_metrics(self, y_true, y_score):
        performance_metrics = {name: metric(y_true, y_score) for name, metric in self._metrics}
        performance_metrics = pd.Series(performance_metrics)
        print(performance_metrics)
        return performance_metrics

class ClassificationMetricSet(MetricSet):

    def __init__(self):
        self._metrics = []
        self._metrics.append(("r1", roc_auc_score))
        self._metrics.append(("r2", roc_auc_score))
        