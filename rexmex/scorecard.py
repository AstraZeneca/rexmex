import pandas as pd

class ScoreCard(object):
    def __init__(self, metric_set: rexmex.metrics.MetricSet):
        
        self.metric_set = metric_set

    def calculate(self, scores: pd.DataFrame, groupping: List[str], filter List[Tuple[str]]) -> pd.DataFrame:
        performance_metrics = scores.groupby(groupping).apply(lambda group: self.metric_set.calculate(group.y_true, group.y_score, filter))
        return performance_metrics