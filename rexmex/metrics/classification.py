import numpy as np
import sklearn.metrics

from rexmex.utils import Annotator

classifications = Annotator()


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


@classifications.annotate(
    lower=0.0,
    upper=1.0,
    higher_is_better=True,
    description="TN / (TN + FP)",
    link="https://en.wikipedia.org/wiki/Specificity_(tests)",
)
def true_negative_rate(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the true negative rate (duplicated in :func:`specificity` and :func:`selectivity`).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        tnr (float): The true negative rate.
    """
    tnr = specificity(y_true, y_score)
    return tnr


@classifications.duplicate(true_negative_rate)
def specificity(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the specificity (duplicate of :func:`true_negative_rate`).

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


@classifications.duplicate(true_negative_rate)
def selectivity(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the selectivity (duplicate of :func:`true_negative_rate`).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        tnr (float): The selectivity score.
    """
    tnr = specificity(y_true, y_score)
    return tnr


@classifications.annotate(
    lower=0.0,
    upper=1.0,
    higher_is_better=True,
    description="TP / (TP + FN)",
    link="https://en.wikipedia.org/wiki/Sensitivity_(test)",
)
def true_positive_rate(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the true positive rate (duplicated in :func:`hit_rate`, :func:`sensitivity`, and :func:`recall_score`).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        tpr (float): The true positive rate.
    """
    tpr = sensitivity(y_true, y_score)
    return tpr


@classifications.duplicate(true_positive_rate)
def hit_rate(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the hit rate (duplicate of :func:`true_positive_rate`).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        tpr (float): The hit rate.
    """
    tpr = sensitivity(y_true, y_score)
    return tpr


@classifications.duplicate(true_positive_rate)
def sensitivity(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the sensitivity (duplicate of :func:`true_positive_rate`).

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


@classifications.duplicate(true_positive_rate, name="Recall", binarize=True)
def recall_score(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the recall for a ground-truth prediction vector pair.

    Duplicate of :func:`true_positive_rate`, but with alternate
    implementation from :mod:`sklearn`.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        recall (float): The value of recall.

    .. note::

        It's surprising that the sklearn implementation of TPR needs
        to be binarized but the rexmex implementation does not
    """
    recall = sklearn.metrics.recall_score(y_true, y_score)
    return recall


@classifications.annotate(
    lower=0.0,
    upper=1.0,
    higher_is_better=True,
    description="TP / (TP + FP)",
    link="https://en.wikipedia.org/wiki/Positive_predictive_value",
    binarize=True,
)
def positive_predictive_value(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the positive predictive value (duplicated in :func:`precision_score`).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        ppv (float): The positive predictive value.

    .. seealso::
        https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values
    """
    tp = true_positive(y_true, y_score)
    fp = false_positive(y_true, y_score)
    ppv = tp / (tp + fp)
    return ppv


@classifications.duplicate(positive_predictive_value, name="Precision")
def precision_score(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the precision for a ground-truth prediction vector pair.

    Duplicate of :func:`positive_predictive_value`, but with an
    alternate implementation using :mod:`sklearn`.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        precision (float): The value of precision.
    """
    precision = sklearn.metrics.precision_score(y_true, y_score)
    return precision


@classifications.annotate(
    lower=0.0,
    upper=1.0,
    higher_is_better=True,
    description="TN / (TN + FN)",
    link="https://en.wikipedia.org/wiki/Negative_predictive_value",
)
def negative_predictive_value(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the negative predictive value (duplicted in :func:`precision_score`).

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


@classifications.annotate(
    lower=0.0,
    upper=1.0,
    higher_is_better=False,
    description="FN / (FN + TP)",
    link="https://en.wikipedia.org/wiki/Type_I_and_type_II_errors#False_positive_and_false_negative_rates",
)
def false_negative_rate(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the false negative rate (duplicated in :func:`miss_rate`).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        fnr (float): The false negative rate value.
    """
    fnr = miss_rate(y_true, y_score)
    return fnr


@classifications.duplicate(false_negative_rate)
def miss_rate(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the miss rate (duplicate of :func:`false_negative_rate`).

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


@classifications.annotate(
    lower=0.0,
    upper=1.0,
    higher_is_better=False,
    description="FP / (FP + TN)",
    link="https://en.wikipedia.org/wiki/False_positive_rate",
)
def false_positive_rate(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the false positive rate (duplicated in :func:`false_positive_rate`).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        fpr (float): The false positive rate value.
    """
    fpr = fall_out(y_true, y_score)
    return fpr


@classifications.duplicate(false_positive_rate)
def fall_out(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the fall out (duplicate of :func:`false_positive_rate`).

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


@classifications.annotate(
    lower=0.0,
    upper=1.0,
    higher_is_better=False,
    description="FP / (FP + TP)",
    link="https://en.wikipedia.org/wiki/False_discovery_rate",
)
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


@classifications.annotate(
    lower=0.0,
    upper=1.0,
    higher_is_better=False,
    description="FN / (FN + TN)",
    link="https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values",
)
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


@classifications.annotate(
    lower=0.0,
    upper=float("inf"),
    upper_inclusive=False,
    higher_is_better=True,
    description="TPR / FPR",
    link="https://en.wikipedia.org/wiki/Positive_likelihood_ratio",
)
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


@classifications.annotate(
    lower=0.0,
    upper=float("inf"),
    upper_inclusive=False,
    higher_is_better=False,
    description="FNR / TNR",
    link="https://en.wikipedia.org/wiki/Negative_likelihood_ratio",
)
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


@classifications.annotate(  # TODO not confident about this annotation
    lower=0.0,
    upper=1.0,
    higher_is_better=False,
    description="√FPR / (√TPR + √FPR)",
    link="https://en.wikipedia.org/wiki/Prevalence_threshold",
)
def prevalence_threshold(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the prevalence threshold score.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        pthr (float): The prevalence threshold value.

    .. seealso::

        - https://en.wikipedia.org/wiki/Sensitivity_and_specificity#Prevalence_threshold
        - https://dx.doi.org/10.1371%2Fjournal.pone.0240215
    """
    fpr = false_positive_rate(y_true, y_score)
    tpr = true_positive_rate(y_true, y_score)
    pthr = (fpr ** 0.5) / (fpr ** 0.5 + tpr ** 0.5)
    return pthr


@classifications.annotate(
    lower=0.0,
    upper=1.0,
    higher_is_better=True,
    description="TP / (TP + FN + FP)",
    link="https://rexmex.readthedocs.io/en/latest/modules/root.html#rexmex.metrics.classification.threat_score",
)
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


@classifications.duplicate(threat_score)
def critical_success_index(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the critical success index (duplicate of :func:`threat_score`).

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        ts (float): The critical success index value.
    """
    ts = threat_score(y_true, y_score)
    return ts


@classifications.annotate(
    lower=0.0,
    upper=1.0,
    higher_is_better=True,
    description="√PPV x √TPR",
    link="https://en.wikipedia.org/wiki/Fowlkes%E2%80%93Mallows_index",
    binarize=True,
)
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


@classifications.annotate(
    lower=0.0,
    upper=1.0,
    higher_is_better=True,
    description="TPR + TNR - 1",
    link="https://en.wikipedia.org/wiki/Informedness",
)
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


@classifications.annotate(
    lower=0.0,
    upper=1.0,
    higher_is_better=True,
    description="PPV + NPV - 1",
    link="https://en.wikipedia.org/wiki/Markedness",
)
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


@classifications.annotate(
    lower=0.0,
    upper=float("inf"),
    upper_inclusive=False,
    higher_is_better=True,
    description="LR+/LR-",
    link="https://en.wikipedia.org/wiki/Diagnostic_odds_ratio",
)
def diagnostic_odds_ratio(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the diagnostic odds ratio.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        dor (float): The diagnostic odds ratio value.

    .. seealso::
        https://en.wikipedia.org/wiki/Diagnostic_odds_ratio
    """
    lr_minus = negative_likelihood_ratio(y_true, y_score)
    lr_plus = positive_likelihood_ratio(y_true, y_score)
    dor = lr_plus / lr_minus
    return dor


@classifications.annotate(
    name="AUC-ROC",
    lower=0.0,
    upper=1.0,
    higher_is_better=True,
    description="Area Under the ROC Curve",
    link="https://en.wikipedia.org/wiki/Receiver_operating_characteristic",
)
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


@classifications.annotate(
    name="Accuracy",
    lower=0.0,
    upper=1.0,
    higher_is_better=True,
    description="(TP + TN) / (TP + TN + FP + FN)",
    link="https://en.wikipedia.org/wiki/Accuracy",
    binarize=True,
)
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


@classifications.annotate(
    name="Balanced accuracy",
    lower=0.0,
    upper=1.0,
    higher_is_better=True,
    description="An adjusted version of the accuracy for imbalanced datasets",
    link="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html",
    binarize=True,
)
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


@classifications.annotate(
    name="F_1",
    lower=0.0,
    upper=1.0,
    higher_is_better=True,
    description="2TP / (2TP + FP + FN)",
    link="https://en.wikipedia.org/wiki/F1_score",
    binarize=True,
)
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


@classifications.annotate(
    name="Average precision",
    lower=0.0,
    upper=1.0,
    higher_is_better=True,
    description="A summary statistic over the precision-recall curve",
    link="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html",
)
def average_precision_score(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate for a ground-truth prediction vector pair.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        average_precision (float): The value of average precision.

    .. seealso::
        https://en.wikipedia.org/w/index.php?title=Information_retrieval&oldid=793358396#Average_precision
    """
    average_precision = sklearn.metrics.average_precision_score(y_true, y_score)
    return average_precision


@classifications.annotate(
    lower=-1.0,
    upper=1.0,
    higher_is_better=True,
    description="A balanced measure applicable even with class imbalance",
    link="https://en.wikipedia.org/wiki/Phi_coefficient",
    binarize=True,
)
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


@classifications.annotate(
    name="AUC-PR",
    lower=0.0,
    upper=1.0,
    higher_is_better=True,
    description="Area Under the Precision-Recall Curve",
    link="https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html",
)
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
