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

**rexmex** is designed with the assumption that the predictions and the ground truth are stored in a  pandas ``DataFrame``. In our example we assume that this ``DataFrame`` has at least two columns ``y_true`` and ``y_score``. The first one contains the ground truth labels/ratings while the second one contains the predictions. Each row in the ``dataframe`` is a single score (user - item) like pair of a source and target. Additional columns represent the source and target identifier and groupings of the model predictions.

.. code-block:: python



Evaluation metrics
------------------------------

.. code-block:: python

Metric sets
------------------------------

Score cards
------------------------------

Utility functions
------------------------------

