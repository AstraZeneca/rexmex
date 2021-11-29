Introduction by example
=======================

**rexmex** is recommender system evaluation metric library. It consists of utilities for recommender system evaluation. First, it provides a comprehensive collection of metrics for the evaluation of recommender systems. Second, it includes a variety of classes and methods for reporting and plotting the performance results. Implemented metrics cover a range of well-known metrics and newly proposed metrics from data mining conferences and prominent journals.


Overview
=======================
--------------------------------------------------------------------------------

We shortly overview the fundamental concepts and features of **rexmex** through simple examples. These are the following:

.. contents::
    :local:

Design philosophy
-----------------

**rexmex** is designed with the assumption that end users might want to use the evaluation metrics and utility functions without using the metric sets and score cards. Because of this, the evaluation metrics and utility functions (e.g. binarisation and normalisation) can be used independently from the **rexmex** library.


Synthetic toy datasets
------------------------------

**rexmex** is designed with the assumption that the predictions and the ground truth are stored in a  pandas ``DataFrame``. In our example we assume that this ``DataFrame`` has at least two columns ``y_true`` and ``y_score``. The first one contains the ground truth labels/ratings while the second one contains the predictions. Each row in the ``DataFrame`` is a single user - item like pairing of a source and target with ground truth and predictions. Additional columns represent groupings of the model predictions. Our library provides synthetic data which can be used for testing the library. The following lines import a dataset and print the head of the table.



.. jupyter-execute::

    from rexmex.dataset import DatasetReader

    reader = DatasetReader()
    scores = reader.read_dataset()

    print(scores.head())

Let us overview the structure of the ``scores DataFrame`` used in our example before we look at the core functionalities of the library. First of all we observe that: it is unindexed, has 6 columns and each row is a prediction. The first two columns ``source_id`` and ``target_id`` correspond to the user and item identifiers. The next two columns ``source_group`` and ``target_group`` help with the calculation of group performance metrics. Finally, ``y_true`` is a vector of ground truth values and ``y_score`` represents predicted probabilities. 

Evaluation metrics
------------------------------
The generic design **rexmex** involves classification metrics that exist on the appropriate namespace. For example the ``pr_auc_score`` is on the ``rexmex.metrics.classification`` namespace, because it is a classification metric. Functions that are on the same name space have the same signature. This specific function takes a target and prediction vector (we use the toy dataset) and outputs the precision recall area under the curve value as a ``float``.


.. jupyter-execute::

    from rexmex.metrics.classification import pr_auc_score

    pr_auc_value = pr_auc_score(scores["y_true"], scores["y_score"])
    print("{:.3f}".format(pr_auc_value))

Metric sets
------------------------------

A ``MetricSet()`` is a base class which inherits from ``dict`` and contains the name of the evaluation metrics and the evaluation metric functions as keys. Each of these functions should have the same signature. There are specialised ``MetricSet()`` variants which inherit from the base class such as the ``ClassificationMetricSet()``. The following example prints the classification metrics stored in this metric set.

.. jupyter-execute::

    from rexmex.metricset import ClassificationMetricSet

    metric_set = ClassificationMetricSet()
    metric_set.print_metrics()

Metric sets also allow the filtering of metrics which are interesting for a specific application. In our case we will only keep 3 of the metrics: ``roc_auc``, ``pr_auc`` and ``accuracy``.

.. jupyter-execute::

    metric_set.filter_metrics(["roc_auc", "pr_auc", "accuracy"])
    metric_set.print_metrics()


Score cards
-----------------------

Score cards allow the calculation of performance metrics for a whole metric set with ease. Let us create a scorecard and reuse the filtered metrics with the scorecard. We will calculate the performance metrics for the toy example. The ``ScoreCard()`` constructor uses the ``metric_set`` instance and the ``generate_report`` method uses the scores from earlier.  The result is a ``DataFrame`` of the scores.

.. jupyter-execute::

    from rexmex.scorecard import ScoreCard

    score_card = ScoreCard(metric_set)
    report = score_card.generate_report(scores)
    print(report)

The score cards allow the advanced reporting of the performance metrics. We could also group on the ``source_group`` and ``target_group`` keys and get specific subgroup performances. Just like this:

.. jupyter-execute::

    report = score_card.generate_report(scores, ["source_group", "target_group"])
    print(report)


Utility functions
------------------------------

A core idea of **rexmex** is the use of ``wrapper`` functions to help with recurring data manipulation. Our utility functions can be used to wrap the metrics when the predictions need to be transformed the ``y_score`` values are not binary. Because of this most evaluation metrics are not meaningful. However wrapping the classification metrics in the ``binarize`` function ensures that there is a binarization step. Let us take a look at this example snippet:

.. jupyter-execute::

    from rexmex.metrics.classification import accuracy_score
    from rexmex.utils import binarize

    new_accuracy_score = binarize(accuracy_score)
    accuracy_value = new_accuracy_score(scores.y_true, scores.y_score)
    print("{:.3f}".format(accuracy_value))
    



