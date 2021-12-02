import numpy as np
import sklearn.metrics


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
    cn = np.sum(1 - y_true)
    return cn


def true_positive(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the number of true positives.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        tp (float): The number of true positives.
    """
    tp = np.sum(y_score[y_true == 1])
    return tp


def true_negative(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the number of true negatives.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        tn (float): The number of true negatives.
    """
    y_score = 1 - y_score
    tn = np.sum(y_score[y_true == 0])
    return tn


def false_positive(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the number of false positives.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        fp (float): The number of false positives.
    """
    fp = np.sum(y_score[y_true == 0])
    return fp


def false_negative(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the number of false negatives.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        fn (float): The number of false negatives.
    """
    y_score = 1 - y_score
    fn = np.sum(y_score[y_true == 1])
    return fn


def specificity(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the specificity (same as selectivity and true negative rate).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        tnr (float): The specificity score.
    """
    n = condition_negative(y_true)
    tn = true_negative(y_true, y_score)
    tnr = tn / n
    return tnr


def selectivity(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the selectivity (same as specificity and true negative rate).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        tnr (float): The selectivity score.
    """
    tnr = specificity(y_true, y_score)
    return tnr


def true_negative_rate(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the true negative rate (same as specificity and selectivity).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        tnr (float): The true negative rate.
    """
    tnr = specificity(y_true, y_score)
    return tnr


def sensitivity(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the sensitivity (same as recall, hit rate and true positive rate).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        tpr (float): The sensitivity score.
    """
    p = condition_positive(y_true)
    tp = true_positive(y_true, y_score)
    tpr = tp / p
    return tpr


def hit_rate(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the hit rate (same as recall, sensitivity and true positive rate).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        tpr (float): The hit rate.
    """
    tpr = sensitivity(y_true, y_score)
    return tpr


def true_positive_rate(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the true positive rate (same as recall, sensitivity and hit rate).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        tpr (float): The true positive rate.
    """
    tpr = sensitivity(y_true, y_score)
    return tpr


def positive_predictive_value(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the positive predictive value (same as precision).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        ppv (float): The positive predictive value.
    """
    tp = true_positive(y_true, y_score)
    fp = false_positive(y_true, y_score)
    ppv = tp / (tp + fp)
    return ppv


def negative_predictive_value(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the negative predictive value (same as precision).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        npv (float): The negative predictive value.
    """
    tn = true_negative(y_true, y_score)
    fn = false_negative(y_true, y_score)
    npv = tn / (tn + fn)
    return npv


def miss_rate(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the miss rate (same as false negative rate).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        fnr (float): The miss rate value.
    """
    fn = false_negative(y_true, y_score)
    p = condition_positive(y_true)
    fnr = fn / p
    return fnr


def false_negative_rate(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the false negative rate (same as miss rate).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        fnr (float): The false negative rate value.
    """
    fnr = miss_rate(y_true, y_score)
    return fnr


def fall_out(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the fall out (same as false positive rate).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        fpr (float): The fall out value.
    """
    fp = false_positive(y_true, y_score)
    n = condition_negative(y_true)
    fpr = fp / n
    return fpr


def false_positive_rate(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the false positive rate (same as fall out).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        fpr (float): The false positive rate value.
    """
    fpr = fall_out(y_true, y_score)
    return fpr


def false_discovery_rate(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the false discovery rate.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        fdr (float): The false discovery rate value.
    """
    fp = false_positive(y_true, y_score)
    tp = true_positive(y_true, y_score)
    fdr = fp / (fp + tp)
    return fdr


def false_omission_rate(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the false omission rate.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        fomr (float): The false omission rate value.
    """
    fn = false_negative(y_true, y_score)
    tn = true_negative(y_true, y_score)
    fomr = fn / (fn + tn)
    return fomr


def positive_likelihood_ratio(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the positive likelihood ratio.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
         (float): The positive likelihood ratio value.
    """
    tpr = true_positive_rate(y_true, y_score)
    fpr = false_positive_rate(y_true, y_score)
    lr_plus = tpr / fpr
    return lr_plus


def negative_likelihood_ratio(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the negative likelihood ratio.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        lr_minus (float): The negative likelihood ratio value.
    """
    fnr = false_negative_rate(y_true, y_score)
    tnr = true_negative_rate(y_true, y_score)
    lr_minus = fnr / tnr
    return lr_minus


def prevalence_threshold(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the prevalence threshold score.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        pthr (float): The prevalence threshold value.
    """
    fpr = false_positive_rate(y_true, y_score)
    tpr = true_positive_rate(y_true, y_score)
    pthr = (fpr ** 0.5) / (fpr ** 0.5 + tpr ** 0.5)
    return pthr


def threat_score(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the threat score.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        ts (float): The threat score value.
    """
    tp = true_positive(y_true, y_score)
    fn = false_negative(y_true, y_score)
    fp = false_positive(y_true, y_score)
    ts = tp / (tp + fn + fp)
    return ts


def critical_success_index(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the critical success index (same as the theat score).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        ts (float): The critical success index value.
    """
    ts = threat_score(y_true, y_score)
    return ts


def fowlkes_mallows_index(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the Fowlkes-Mallows index.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        fm (float): The the Fowlkes-Mallows index value.
    """
    ppv = positive_predictive_value(y_true, y_score)
    tpr = true_positive_rate(y_true, y_score)
    fm = (ppv * tpr) ** 0.5
    return fm


def informedness(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the informedness.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        bm (float): The informedness value.
    """
    tpr = true_positive_rate(y_true, y_score)
    tnr = true_negative_rate(y_true, y_score)
    bm = tpr + tnr - 1
    return bm


def markedness(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the markedness.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        mk (float): The markedness value.
    """
    ppv = positive_predictive_value(y_true, y_score)
    npv = negative_predictive_value(y_true, y_score)
    mk = ppv + npv - 1
    return mk


def diagnostic_odds_ratio(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the diagnostic odds ratio.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        dor (float): The diagnostic odds ratio value.
    """
    lr_minus = negative_likelihood_ratio(y_true, y_score)
    lr_plus = positive_likelihood_ratio(y_true, y_score)
    dor = lr_plus / lr_minus
    return dor


def roc_auc_score(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the AUC for a ground-truth prediction vector pair.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        auc (float): The value of the area under the curve.
    """
    auc = sklearn.metrics.roc_auc_score(y_true, y_score)
    return auc


def accuracy_score(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the accuracy score for a ground-truth prediction vector pair.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
         (float): The value of .
    """
    accuracy = sklearn.metrics.accuracy_score(y_true, y_score)
    return accuracy


def balanced_accuracy_score(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the balanced accuracy for a ground-truth prediction vector pair.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        balanced_accuracy (float): The value of the balanced accuracy score.
    """
    balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_true, y_score)
    return balanced_accuracy


def f1_score(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the F-1 score for a ground-truth prediction vector pair.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
         f1 (float): The value of the F-1 score.
    """
    f1 = sklearn.metrics.f1_score(y_true, y_score)
    return f1


def precision_score(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the precision for a ground-truth prediction vector pair.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        precision (float): The value of precision.
    """
    precision = sklearn.metrics.precision_score(y_true, y_score)
    return precision


def recall_score(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the recall for a ground-truth prediction vector pair.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        recall (float): The value of recall.
    """
    recall = sklearn.metrics.recall_score(y_true, y_score)
    return recall


def average_precision_score(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate for a ground-truth prediction vector pair.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        average_precision (float): The value of average precision.
    """
    average_precision = sklearn.metrics.average_precision_score(y_true, y_score)
    return average_precision


def matthews_correlation_coefficient(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate Matthew's correlation coefficient for a ground-truth prediction vector pair.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        mat_cor (float): The value of Matthew's correlation coefficient.
    """
    mat_cor = sklearn.metrics.matthews_corrcoef(y_true, y_score)
    return mat_cor


def pr_auc_score(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the precision recall area under the curve (PR AUC) for a ground-truth prediction vector pair.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        pr_auc (float): The value of the precision-recall area under the curve.
    """
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_score)
    pr_auc = sklearn.metrics.auc(recall, precision)
    return pr_auc
