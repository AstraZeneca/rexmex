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

.. jupyter-execute::
    from rexmex.metrics.classification import pr_auc_score

    pr_auc_value = pr_auc_score(scores["y_true"], scores["y_score"])
    print(pr_auc_value)

Metric sets
------------------------------

Score cards
------------------------------

Utility functions
------------------------------

