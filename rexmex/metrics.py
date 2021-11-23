import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_curve, auc

def pr_auc_score(y_true: np.array, y_scores: np.array) -> float:
    """
    Calculate the precision recall area under the curve (PR AUC) for a ground-truth prediction vector pair.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_scores (array-like):  An N x 1 array of predicted values.
    Returns:
        pr_auc (float): The value of the precision-recall area under the curve.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    return pr_auc

def root_mean_squared_error(y_true: np.array, y_scores: np.array) -> float:
    """
    Calculate the root mean squared error (RMSE) for a ground-truth prediction vector pair.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_scores (array-like):  An N x 1 array of predicted values.
    Returns:
        rmse (float): The value of the root mean squared error.
    """
    rmse = mean_squared_error(y_true, y_scores)**0.5
    return rmse

def symmetric_mean_absolute_percentage_error(y_true: np.array, y_scores: np.array) -> float:
    """
    Calculate the symmetric mean absolute percentage error (SMAPE) for a ground-truth prediction vector pair.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_scores (array-like):  An N x 1 array of predicted values.
    Returns:
        smape (float): The value of the symmetric mean absolute percentage error .
    """
    smape = 100*np.mean(np.abs(y_scores - y_true)/((np.abs(y_scores)+np.abs(y_true))/2))
    return smape

def condition_positive(y_true: np.array) -> float:
    """
    Calculate the number of instances which are positive.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
    Returns:
        cp (float): The number of positive instances.
    """
    cp = np.sum(y_true)
    return cp

def condition_negative(y_true: np.array) -> float:
    """
    Calculate the number of instances which are negative.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
    Returns:
        cn (float): The number of negative instances.
    """
    cn = np.sum(1-y_true)
    return cn

def true_positive(y_true: np.array, y_scores: np.array) -> float:
    """
    Calculate the number of true positives.
    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_scores (array-like):  An N x 1 array of predicted values.
    Returns:
        tp (float): The number of true positives.
    """
    tp = np.sum(y_scores[y_true==1])
    return tp

def true_negative(y_true: np.array, y_scores: np.array) -> float:
    """
    Calculate the number of true negatives.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_scores (array-like):  An N x 1 array of predicted values.
    Returns:
        tn (float): The number of true negatives.
    """
    y_scores = 1 - y_scores
    tn = np.sum(y_scores[y_true==0])
    return tn

def false_positive(y_true: np.array, y_scores: np.array) -> float:
    """
    Calculate the number of false positives.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_scores (array-like):  An N x 1 array of predicted values.
    Returns:
        fp (float): The number of false positives.
    """
    fp = np.sum(y_scores[y_true==0])
    return fp

def false_negative(y_true: np.array, y_scores: np.array) -> float:
    """
    Calculate the number of false negatives.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_scores (array-like):  An N x 1 array of predicted values.
    Returns:
        fn (float): The number of false negatives.
    """
    y_scores = 1 - y_scores
    fn = np.sum(y_scores[y_true==1])
    return fn

def specificity(y_true: np.array, y_scores: np.array) -> float:
    """
    Calculate the specificity (same as selectivity and true negative rate).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_scores (array-like):  An N x 1 array of predicted values.
    Returns:
        tnr (float): The specificity score.
    """
    n = condition_negative(y_true)
    tn = true_negative(y_true, y_scores)
    tnr = tn / n
    return tnr

def selectivity(y_true: np.array, y_scores: np.array) -> float:
    """
    Calculate the selectivity (same as specificity and true negative rate).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_scores (array-like):  An N x 1 array of predicted values.
    Returns:
        tnr (float): The selectivity score.
    """
    tnr = specificity(y_true, y_scores)
    return tnr

def true_negative_rate(y_true: np.array, y_scores: np.array) -> float:
    """
    Calculate the true negative rate (same as specificity and selectivity).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_scores (array-like):  An N x 1 array of predicted values.
    Returns:
        tnr (float): The true negative rate.
    """
    tnr = specificity(y_true, y_scores)
    return tnr

def sensitivity(y_true: np.array, y_scores: np.array) -> float:
    """
    Calculate the sensitivity (same as recall, hit rate and true positive rate).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_scores (array-like):  An N x 1 array of predicted values.
    Returns:
        tpr (float): The sensitivity score.
    """
    p = condition_positive(y_true)
    tp = true_positive(y_true, y_scores)
    tpr = tp / p
    return tpr

def hit_rate(y_true: np.array, y_scores: np.array) -> float:
    """
    Calculate the hit rate (same as recall, sensitivity and true positive rate).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_scores (array-like):  An N x 1 array of predicted values.
    Returns:
        tpr (float): The hit rate.
    """
    tpr = sensitivity(y_true, y_scores)
    return tpr

def true_positive_rate(y_true: np.array, y_scores: np.array) -> float:
    """
    Calculate the true positive rate (same as recall, sensitivity and hit rate).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_scores (array-like):  An N x 1 array of predicted values.
    Returns:
        tpr (float): The true positive rate.
    """
    tpr = sensitivity(y_true, y_scores)
    return tpr

def positive_predictive_value(y_true: np.array, y_scores: np.array) -> float:
    """
    Calculate the positive predictive value (same as precision).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_scores (array-like):  An N x 1 array of predicted values.
    Returns:
        ppv (float): The positive predictive value.
    """
    tp = true_positive(y_true, y_scores)
    fp = false_positive(y_true, y_scores)
    ppv = tp/(tp+fp)
    return ppv

def negative_predictive_value(y_true: np.array, y_scores: np.array) -> float:
    """
    Calculate the negative predictive value (same as precision).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_scores (array-like):  An N x 1 array of predicted values.
    Returns:
        npv (float): The negative predictive value.
    """
    tn = true_negative(y_true, y_scores)
    fn = false_negative(y_true, y_scores)
    npv = tn/(tn+fn)
    return npv

def miss_rate(y_true: np.array, y_scores: np.array) -> float:
    """
    Calculate the miss rate (same as false negative rate).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_scores (array-like):  An N x 1 array of predicted values.
    Returns:
        fnr (float): The miss rate value.
    """
    fn = false_negative(y_true, y_scores)
    p = condition_positive(y_true)
    fnr = fn/p
    return fnr

def false_negative_rate(y_true: np.array, y_scores: np.array) -> float:
    """
    Calculate the false negative rate (same as miss rate).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_scores (array-like):  An N x 1 array of predicted values.
    Returns:
        fnr (float): The false negative rate value.
    """
    fnr = miss_rate(y_true, y_scores)
    return fnr

def fall_out(y_true: np.array, y_scores: np.array) -> float:
    """
    Calculate the fall out (same as false positive rate).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_scores (array-like):  An N x 1 array of predicted values.
    Returns:
        fpr (float): The fall out value.
    """
    fp = false_positive(y_true, y_scores)
    n = condition_negative(y_true)
    fpr = fp/n
    return fpr

def false_positive_rate(y_true: np.array, y_scores: np.array) -> float:
    """
    Calculate the false positive rate (same as fall out).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_scores (array-like):  An N x 1 array of predicted values.
    Returns:
        fpr (float): The false positive rate value.
    """
    fpr = fall_out(y_true, y_scores)
    return fpr

def false_discovery_rate(y_true: np.array, y_scores: np.array) -> float:
    """
    Calculate the false discovery rate.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_scores (array-like):  An N x 1 array of predicted values.
    Returns:
        fdr (float): The false discovery rate ralue.
    """
    fp = false_positive(y_true, y_scores)
    tp = true_positive(y_true, y_scores)
    fdr = fp/(fp+tp)
    return fdr

def false_omission_rate(y_true: np.array, y_scores: np.array) -> float:
    """
    Calculate the false omission rate.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_scores (array-like):  An N x 1 array of predicted values.
    Returns:
        fomr (float): The false omission rate ralue.
    """
    fn = false_negative(y_true, y_scores)
    tn = true_negative(y_true, y_scores)
    fomr = fn/(fn+tn)
    return fomr

def positive_likelihood_ratio():
    pass

def negative_likelihood_ratio():
    pass

def prevalence_threshold():
    pass

def threat_score():
    pass

def critical_success_index():
    pass

def prevalence():
    pass

def fowlkes_mallows_index():
    pass

def informedness():
    pass 

def markedness():
    pass

def diagnostic_odds_ratio():
    pass
