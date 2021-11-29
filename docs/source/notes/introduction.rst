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


Standardized dataset ingestion
------------------------------

**rexmex** is designed with the assumption that the 

.. code-block:: python

    import networkx as nx
    from karateclub import DeepWalk
    
    g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

    model = DeepWalk()
    model.fit(g)
    embedding = model.get_embedding()

Evaluation metrics
------------------------------

Metric sets
------------------------------

Score cards
------------------------------

Utility functions
------------------------------

